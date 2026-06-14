"""
LyraSVC 训练脚本
- 自动划分 train/val (90/10)
- 默认预加载全部数据到内存 (200MB)
- 随机截取固定长度片段训练
"""

import os
import random
import time
import glob
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from contextlib import nullcontext
from torch.utils.data import Dataset, DataLoader

from modules.lyra_model import LyraModel as LyraModelBase, EMA

def get_model_class(model_type: str = "base"):
    if model_type == "turbo":
        from modules.lyra_model_turbo import LyraModelTurbo
        return LyraModelTurbo
    return LyraModelBase


def load_config(config_path: str = "config/config.yaml") -> dict:
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# Dataset
# ============================================================

class SVCDataset(Dataset):
    """从预处理后的 data/ 目录加载 PPG + F0 + Mel，随机截取 segment_len 帧"""

    def __init__(self, data_dir: str, speakers: list, segment_len: int = 256,
                 preload: bool = True):
        self.data_dir = data_dir
        self.segment_len = segment_len
        self.preload = preload
        self.samples = []
        self._cache = {} if preload else None

        for spk_id, speaker in enumerate(speakers):
            spk_dir = os.path.join(data_dir, speaker)
            if not os.path.isdir(spk_dir):
                continue
            ppg_files = {f.replace('_ppg.npy', '') for f in os.listdir(spk_dir) if f.endswith('_ppg.npy')}
            f0_files = {f.replace('_f0.npy', '') for f in os.listdir(spk_dir) if f.endswith('_f0.npy')}
            mel_files = {f.replace('_mel.npy', '') for f in os.listdir(spk_dir) if f.endswith('_mel.npy')}
            common = ppg_files & f0_files & mel_files
            for name in sorted(common):
                self.samples.append((speaker, spk_id, name))

        if preload:
            print(f"  Preloading {len(self.samples)} samples into RAM...")
            for i in range(len(self.samples)):
                self._cache[i] = self._load_raw(i)
            total_mb = sum(v['mel'].nbytes + v['ppg'].nbytes + v['f0'].nbytes
                          for v in self._cache.values()) / 1e6
            print(f"  Done ({total_mb:.0f}MB)")

    def _load_raw(self, idx):
        speaker, spk_id, name = self.samples[idx]
        spk_dir = os.path.join(self.data_dir, speaker)
        ppg = np.load(os.path.join(spk_dir, f"{name}_ppg.npy")).astype(np.float32)
        f0 = np.load(os.path.join(spk_dir, f"{name}_f0.npy")).astype(np.float32)
        mel = np.load(os.path.join(spk_dir, f"{name}_mel.npy")).astype(np.float32)
        return {'ppg': ppg, 'f0': f0, 'mel': mel, 'spk_id': spk_id}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self._cache is not None:
            data = self._cache[idx]
            ppg, f0, mel, spk_id = data['ppg'], data['f0'], data['mel'], data['spk_id']
        else:
            data = self._load_raw(idx)
            ppg, f0, mel, spk_id = data['ppg'], data['f0'], data['mel'], data['spk_id']

        if ppg.ndim == 3:
            ppg = ppg[0]
        orig_mel_len = mel.shape[-1]
        if orig_mel_len < self.segment_len:
            pad = self.segment_len - orig_mel_len
            mel = np.pad(mel, ((0, 0), (0, pad)), mode='reflect')
            f0 = np.pad(f0, ((0, int(pad * len(f0) / max(1, orig_mel_len)))), mode='edge')
            ppg_pad = int(pad * ppg.shape[0] / max(1, orig_mel_len))
            ppg = np.pad(ppg, ((0, ppg_pad), (0, 0)), mode='edge')
        else:
            start = random.randint(0, orig_mel_len - self.segment_len)
            mel = mel[:, start:start + self.segment_len]
            ratio_f0 = len(f0) / orig_mel_len
            ratio_ppg = ppg.shape[0] / orig_mel_len
            f0_start = int(start * ratio_f0)
            f0_end = int((start + self.segment_len) * ratio_f0)
            ppg_start = int(start * ratio_ppg)
            ppg_end = int((start + self.segment_len) * ratio_ppg)
            f0 = f0[f0_start:f0_end]
            ppg = ppg[ppg_start:ppg_end]
            if len(f0) == 0:
                f0 = np.array([0.0])
            if len(ppg) == 0:
                ppg = np.zeros((1, ppg.shape[-1] if ppg.ndim > 1 else 2048), dtype=np.float32)

        mel = mel.transpose()
        f0 = np.atleast_1d(f0)
        return {
            'ppg': torch.from_numpy(ppg),
            'f0': torch.from_numpy(f0),
            'mel': torch.from_numpy(mel),
            'speaker_id': spk_id,
        }


def collate_batch(batch: list) -> dict:
    """Pad 到 batch 内最大长度"""
    ppg_list = [b['ppg'] for b in batch]
    f0_list = [b['f0'] for b in batch]
    mel_list = [b['mel'] for b in batch]
    spk = torch.tensor([b['speaker_id'] for b in batch])

    max_ppg = max(p.shape[0] for p in ppg_list)
    max_f0 = max(f.shape[0] for f in f0_list)
    max_mel = max(m.shape[0] for m in mel_list)

    ppg_batch = torch.zeros(len(batch), max_ppg, ppg_list[0].shape[-1])
    f0_batch = torch.zeros(len(batch), max_f0)
    mel_batch = torch.zeros(len(batch), max_mel, mel_list[0].shape[-1])

    for i, (p, f, m) in enumerate(zip(ppg_list, f0_list, mel_list)):
        ppg_batch[i, :p.shape[0]] = p
        f0_batch[i, :f.shape[0]] = f
        mel_batch[i, :m.shape[0]] = m

    return {'ppg': ppg_batch, 'f0': f0_batch, 'mel': mel_batch, 'speaker_id': spk}


def mel_snr(gt, pred):
    """信噪比 (dB): 信号能量 / 误差能量"""
    eps = 1e-8
    signal = (gt ** 2).flatten(1).sum(1)
    noise = ((gt - pred) ** 2).flatten(1).sum(1)
    return (10 * torch.log10(signal / (noise + eps) + eps)).mean().item()

def mel_psnr(gt, pred, max_val=17.0):
    """峰值信噪比 (dB): 峰值能量 / 误差能量, max_val=SPEC_RANGE"""
    eps = 1e-8
    mse = F.mse_loss(gt, pred, reduction='none').flatten(1).mean(1)
    return (20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse + eps))).mean().item()

def mel_sisnr(gt, pred):
    """SI-SNR (dB): 信号投影能量 / 垂直误差, 对音量偏移不敏感"""
    eps = 1e-8
    s = gt.flatten(1)       # (B, T*F)
    s_hat = pred.flatten(1)
    s_target = (s_hat * s).sum(1, keepdim=True) * s / (s * s).sum(1, keepdim=True).clamp(min=eps)
    e_noise = s_hat - s_target
    val = 10 * torch.log10((s_target ** 2).sum(1) / (e_noise ** 2).sum(1).clamp(min=eps) + eps)
    return val.mean().item()

def run_validation(model, val_loader, device, use_amp, amp_dtype, dpm_steps):
    model.eval()
    val_loss = 0.0
    val_recon = 0.0
    val_snr = 0.0
    val_psnr = 0.0
    val_sisnr = 0.0
    with torch.no_grad(), (torch.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()):
        for batch in val_loader:
            ppg = batch['ppg'].to(device)
            f0 = batch['f0'].to(device)
            mel = batch['mel'].to(device)
            spk = batch['speaker_id'].to(device)
            eps_loss, _, _ = model.train_loss(mel, ppg, f0, energy_mel=mel, speaker_ids=spk)
            val_loss += eps_loss.item()

            B = mel.shape[0]
            T_mel = mel.shape[1]
            gen_mel = model.sample(ppg, f0, mel, T_mel, speaker_ids=spk,
                                      dpm_steps=dpm_steps, cfg_scale=1.0, device=device)
            gt = mel.clamp(-12, 5)
            pred = gen_mel.clamp(-12, 5)
            val_recon += F.mse_loss(pred, gt).sqrt().item()
            val_snr += mel_snr(gt, pred)
            val_psnr += mel_psnr(gt, pred)
            val_sisnr += mel_sisnr(gt, pred)

    n = len(val_loader)
    val_loss /= n
    val_recon /= n
    val_snr /= n
    val_psnr /= n
    val_sisnr /= n
    return val_loss, val_recon, val_snr, val_psnr, val_sisnr


# ============================================================
# Training Loop
# ============================================================

def train(resume_from=None, model_type=None):
    config = load_config()

    # 模型类型: CLI 参数优先级 > config > 默认 base
    if model_type is None:
        model_type = config.get("model", {}).get("architecture", "base")
    LyraModel = get_model_class(model_type)

    data_cfg = config.get("data", {})
    data_dir = data_cfg.get("processed", "data")

    m_cfg = config.get("model", {})
    d_cfg = config.get("diffusion", {})
    t_cfg = config.get("train", {})
    inf_cfg = config.get("inference", {})

    # 收集说话人
    speakers = sorted(d for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d)))

    batch_size = t_cfg.get("batch_size", 4)
    grad_accum = t_cfg.get("grad_accum", 1)
    segment_len = t_cfg.get("segment_len", 128)
    num_workers = t_cfg.get("num_workers", 4)
    preload = t_cfg.get("preload", True)
    epochs = t_cfg.get("epochs", 200)
    val_split = t_cfg.get("val_split", 0.1)
    lr = t_cfg.get("learning_rate", 1e-4)
    precision = t_cfg.get("precision", "bf16")
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    amp_dtype = dtype_map.get(precision, torch.bfloat16)
    use_amp = precision != "fp32"
    weight_decay = t_cfg.get("weight_decay", 1e-5)
    beta1 = t_cfg.get("beta1", 0.9)
    beta2 = t_cfg.get("beta2", 0.999)
    grad_clip = t_cfg.get("grad_clip", 1.0)
    dpm_steps = inf_cfg.get("dpm_steps", 20)
    log_interval = t_cfg.get("log_interval", 500)
    val_interval = t_cfg.get("val_interval", 500)
    warmup_steps = t_cfg.get("warmup_steps", 1000)
    patience = t_cfg.get("patience", 30)  # 连续 N 次验证不创新低就停
    checkpoint_dir = t_cfg.get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Train/Val split
    random.seed(42)
    full_dataset = SVCDataset(data_dir, speakers, segment_len=segment_len, preload=preload)
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    split = int((1 - val_split) * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = torch.utils.data.Subset(full_dataset, train_idx)
    val_ds = torch.utils.data.Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_batch, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_batch, num_workers=num_workers, pin_memory=True)

    print(f"Speakers: {speakers}")
    print(f"Model: {model_type} ({LyraModel.__name__})")
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    print(f"Batch: {batch_size} x {grad_accum} accum, Seg: {segment_len} frames "
          f"(~{segment_len/86:.1f}s)")
    print(f"Dim: {m_cfg.get('hidden_dim', 256)}, Depth: {m_cfg.get('depth', 6)}")
    print(f"LR: {lr}, Epochs: {epochs}")

    # 模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LyraModel(
        num_speakers=len(speakers),
        ppg_dim=m_cfg.get("ppg_dim", 1280),
        hidden_dim=m_cfg.get("hidden_dim", 768),
        depth=m_cfg.get("depth", 12),
        num_heads=m_cfg.get("num_heads", 12),
        mlp_ratio=m_cfg.get("mlp_ratio", 4.0),
        mel_bins=m_cfg.get("mel_bins", 128),
        pitch_max_freq=m_cfg.get("pitch_max_freq", 2000.0),
        use_ref_spk=m_cfg.get("use_ref_spk", True),
        content_dim=m_cfg.get("content_dim", 1280),
        segment_len=segment_len,
        spec_min=m_cfg.get("spec_min", -12.0),
        spec_max=m_cfg.get("spec_max", 5.0),
        diffusion_timesteps=d_cfg.get("timesteps", 1000),
        beta_start=d_cfg.get("beta_start", 0.0001),
        beta_end=d_cfg.get("beta_end", 0.02),
        cfg_dropout_prob=d_cfg.get("cfg_dropout_prob", 0.1),
    ).to(device)

    use_8bit = t_cfg.get("optim_8bit", False)
    if use_8bit:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr, betas=(beta1, beta2),
                                         weight_decay=weight_decay)
        print(f"  Using 8-bit AdamW")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(beta1, beta2),
                                      weight_decay=weight_decay, fused=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    ema = EMA(model, decay=0.999)

    # 恢复训练
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    best_recon = float('inf')
    patience_counter = 0

    if resume_from and os.path.exists(resume_from):
        print(f"  Resuming from {resume_from}")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)

        # 说话人数变化: 保留 DiT 主干, 适配 speaker 表
        old_speakers = ckpt.get('speakers', [])
        if old_speakers != speakers:
            print(f"  Speakers changed, adapting...")
            old_weight = ckpt['model_state_dict']['speaker_enc.lookup.weight']
            new_weight = old_weight.mean(0, keepdim=True).repeat(len(speakers), 1)
            # 名字匹配上的保留原 embedding, 新的用均值
            for new_idx, name in enumerate(speakers):
                if name in old_speakers:
                    old_idx = old_speakers.index(name)
                    new_weight[new_idx] = old_weight[old_idx]
                    print(f"    {name}: kept from previous model")
                else:
                    print(f"    {name}: initialized from average")
            ckpt['model_state_dict']['speaker_enc.lookup.weight'] = new_weight
            # 优化器和 EMA 状态不适合迁移, 丢弃
            for key in ['optimizer_state_dict', 'scheduler_state_dict',
                        'scaler_state_dict', 'ema_shadow']:
                ckpt.pop(key, None)

        # 删除 rough_decoder 旧权重 (已废弃)
        rough_keys = [k for k in ckpt.get('model_state_dict', {}).keys() if k.startswith('rough_decoder.') or k.startswith('rough_mel_inject.')]
        for k in rough_keys:
            ckpt['model_state_dict'].pop(k, None)
        if rough_keys:
            print(f"  RoughMelDecoder dim changed, reinitializing from scratch")
            for key in ['optimizer_state_dict', 'scheduler_state_dict',
                        'scaler_state_dict', 'ema_shadow']:
                ckpt.pop(key, None)

        model.load_state_dict(ckpt['model_state_dict'], strict=False)

        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                scheduler.load_state_dict(ckpt.get('scheduler_state_dict', {}))
                if use_amp and 'scaler_state_dict' in ckpt:
                    scaler.load_state_dict(ckpt['scaler_state_dict'])
                if 'ema_shadow' in ckpt:
                    ema.shadow = ckpt['ema_shadow']
            except (ValueError, KeyError):
                print("  Optimizer/EMA state incompatible (architecture changed), starting fresh")
                ckpt.pop('optimizer_state_dict', None)
                ckpt.pop('ema_shadow', None)
            start_epoch = ckpt.get('epoch', 0)
            global_step = ckpt.get('global_step', 0)
            best_val_loss = ckpt.get('val_loss', float('inf'))
            best_recon = ckpt.get('best_recon', float('inf'))
            patience_counter = ckpt.get('patience_counter', 0)
        print(f"  Epoch {start_epoch}, step {global_step}, best_val_loss {best_val_loss:.4f}")

    # 强制使用 config 中的 LR (覆盖 resume 恢复的旧值)
    if resume_from:
        for pg in optimizer.param_groups:
            pg['lr'] = lr

    # 恢复训练时跳过 warmup
    if resume_from and global_step > 0:
        warmup_steps = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_start = time.time()
        total_batches = len(train_loader)
        last_step_time = time.time()
        recent_loss = 0.0
        recent_steps = 0

        for batch_idx, batch in enumerate(train_loader):
            ppg = batch['ppg'].to(device)
            f0 = batch['f0'].to(device)
            mel = batch['mel'].to(device)
            spk = batch['speaker_id'].to(device)

            amp_ctx = torch.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()
            with amp_ctx:
                eps_loss, _, _ = model.train_loss(
                    mel, ppg, f0, energy_mel=mel, speaker_ids=spk
                )
                total_loss = eps_loss
                loss = total_loss / grad_accum

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            recent_loss += total_loss.item() * grad_accum
            recent_steps += 1

            if (batch_idx + 1) % grad_accum == 0:
                # Warmup: 前 warmup_steps 步线性提升
                if global_step < warmup_steps:
                    for pg in optimizer.param_groups:
                        pg['lr'] = lr * (global_step + 1) / warmup_steps

                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                ema.update(model)
                global_step += 1

                # 步速: 最近一步的实际耗时
                now = time.time()
                step_time = now - last_step_time
                last_step_time = now

            cur_loss = recent_loss / max(1, recent_steps)
            elapsed = time.time() - epoch_start
            rate_str = f"{step_time:.1f}s/step" if step_time >= 1 else f"{1/max(step_time,0.01):.1f}step/s"
            print(f"\r\033[K  Epoch {epoch+1:3d} [{batch_idx+1:4d}/{total_batches}] "
                  f"step={global_step:5d} loss={cur_loss:7.4f} "
                  f"lr={optimizer.param_groups[0]['lr']:.2e} {rate_str}", end="", flush=True)

            if global_step % log_interval == 0 and global_step > 0:
                print(f"\n  --- step {global_step:5d} avg_loss: {cur_loss:.4f} "
                      f"ETA: {elapsed/(batch_idx+1)*(total_batches-batch_idx-1):.0f}s")
                recent_loss = 0.0
                recent_steps = 0

            # --- Step-based validation ---
            if global_step > 0 and global_step % val_interval == 0 and global_step >= warmup_steps:
                val_loss, val_recon, val_snr, val_psnr, val_sisnr = run_validation(
                    model, val_loader, device, use_amp, amp_dtype, dpm_steps)
                elapsed = time.time() - epoch_start
                print(f"  Step {global_step:5d} | loss: {val_loss:.4f} | recon: {val_recon:.3f} | "
                      f"SNR: {val_snr:.1f} | PSNR: {val_psnr:.1f} | SI-SNR: {val_sisnr:.1f} | "
                      f"best: {best_val_loss:.4f}")

                # Spot check: short segment (training length)
                try:
                    sample_data = next(iter(val_loader))
                    s_ppg = sample_data['ppg'][:1].to(device)
                    s_f0 = sample_data['f0'][:1].to(device)
                    s_mel = sample_data['mel'][:1].to(device)
                    s_spk = sample_data['speaker_id'][:1].to(device)
                    s_T = s_mel.shape[1]
                    if s_T > model.segment_len:
                        model.set_rope_scale(s_T / model.segment_len)
                    gen_mel = model.sample(s_ppg, s_f0, s_mel, s_T, s_spk,
                                          dpm_steps=8, cfg_scale=1.0, device=device)
                    gen_mel = gen_mel.clamp(-12, 5)
                    s_mel_clamped = s_mel.clamp(-12, 5)

                    # Frame energy correlation (0..1): tracks source dynamics
                    src_env = s_mel_clamped[0].norm(dim=-1)
                    gen_env = gen_mel[0].norm(dim=-1)
                    src_centered = src_env - src_env.mean()
                    gen_centered = gen_env - gen_env.mean()
                    denom = (src_centered.norm() * gen_centered.norm()).clamp(min=1e-8)
                    fcorr = (src_centered * gen_centered).sum() / denom

                    # Condition sensitivity: shuffle PPG → how much does output change?
                    shuffled_ppg = s_ppg[:, torch.randperm(s_ppg.shape[1]), :]
                    gen_shuffled = model.sample(shuffled_ppg, s_f0, s_mel, s_T, s_spk,
                                                dpm_steps=8, cfg_scale=1.0, device=device)
                    gen_shuffled = gen_shuffled.clamp(-12, 5)
                    csens = F.mse_loss(gen_mel, gen_shuffled).item()

                    print(f"  spot (T={s_T}): src_mean={s_mel_clamped.mean().item():.2f} src_std={s_mel_clamped.std().item():.2f} "
                          f"gen_mean={gen_mel.mean().item():.2f} gen_std={gen_mel.std().item():.2f}  "
                          f"fcorr={fcorr.item():.3f}  csens={csens:.4f}")
                except Exception as e:
                    print(f"  spot check failed: {e}")

                # Full-length spot check: test YaRN extrapolation
                try:
                    full_files = []
                    for spk in speakers:
                        spk_dir_full = os.path.join(data_dir, spk)
                        full_files.extend(glob.glob(os.path.join(spk_dir_full, "*_mel.npy")))
                    if full_files:
                        full_path = random.choice(full_files)
                        prefix = full_path.replace("_mel.npy", "")
                        full_mel = np.load(full_path).astype(np.float32)
                        full_ppg = np.load(prefix + "_ppg.npy").astype(np.float32)
                        full_f0 = np.load(prefix + "_f0.npy").astype(np.float32)
                        if full_ppg.ndim == 3:
                            full_ppg = full_ppg[0]
                        full_mel = full_mel.T  # (T, 128)
                        f_T = min(full_mel.shape[0], 1024)
                        full_mel = full_mel[:f_T]
                        rope_max = model.segment_len
                        if f_T > rope_max:
                            model.set_rope_scale(f_T / rope_max)

                        f_ppg_t = torch.from_numpy(full_ppg).float().unsqueeze(0).to(device)
                        f_f0_t = torch.from_numpy(full_f0).float().unsqueeze(0).to(device)
                        f_mel_t = torch.from_numpy(full_mel).float().unsqueeze(0).to(device)
                        f_spk = None
                        for si, spk_name in enumerate(speakers):
                            if spk_name in full_path:
                                f_spk = si
                                break
                        if f_spk is None:
                            f_spk = 0
                        spk_t = torch.tensor([f_spk], dtype=torch.long, device=device)

                        f_gen = model.sample(f_ppg_t, f_f0_t, f_mel_t, f_T, spk_t,
                                            dpm_steps=8, cfg_scale=1.0, device=device)
                        f_gen = f_gen.clamp(-12, 5)
                        f_mel_c = f_mel_t.clamp(-12, 5)

                        f_src_env = f_mel_c[0].norm(dim=-1)
                        f_gen_env = f_gen[0].norm(dim=-1)
                        f_sc = f_src_env - f_src_env.mean()
                        f_gc = f_gen_env - f_gen_env.mean()
                        f_fcorr = (f_sc * f_gc).sum() / (f_sc.norm() * f_gc.norm()).clamp(min=1e-8)

                        f_shuf = f_ppg_t[:, torch.randperm(f_ppg_t.shape[1]), :]
                        f_gen2 = model.sample(f_shuf, f_f0_t, f_mel_t, f_T, spk_t,
                                             dpm_steps=8, cfg_scale=1.0, device=device)
                        f_gen2 = f_gen2.clamp(-12, 5)
                        f_csens = F.mse_loss(f_gen, f_gen2).item()

                        # Reset RoPE scale for training
                        model.set_rope_scale(1.0)

                        print(f"  full (T={f_T}): src_mean={f_mel_c.mean().item():.2f} src_std={f_mel_c.std().item():.2f} "
                              f"gen_mean={f_gen.mean().item():.2f} gen_std={f_gen.std().item():.2f}  "
                              f"fcorr={f_fcorr.item():.3f}  csens={f_csens:.4f}")
                except Exception:
                    pass  # Full-length spot check is best-effort

                checkpoint_dict = {
                    'epoch': epoch + 1, 'global_step': global_step,
                    'model_state_dict': model.state_dict(), 'ema_shadow': ema.shadow,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss, 'best_recon': best_recon,
                    'patience_counter': patience_counter, 'speakers': speakers,
                }
                if use_amp:
                    checkpoint_dict['scaler_state_dict'] = scaler.state_dict()
                torch.save(checkpoint_dict, os.path.join(checkpoint_dir, "latest.pt"))

                improved = False
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_recon = val_recon
                    improved = True
                    torch.save(checkpoint_dict, os.path.join(checkpoint_dir, "best.pt"))
                    torch.save({'model_state_dict': ema.shadow, 'speakers': speakers},
                               os.path.join(checkpoint_dir, "best_ema.pt"))
                    print(f"    -> saved best.pt + best_ema.pt (val_loss={val_loss:.4f})")
                elif val_recon < best_recon:
                    best_recon = val_recon
                    improved = True
                    print(f"    -> new best recon={val_recon:.4f}")

                scheduler.step(val_loss)
                patience_counter = 0 if improved else patience_counter + 1
                if patience_counter >= patience:
                    print(f"\n  Early stopping: val_loss no improvement for {patience} validations (step {global_step})")
                    return

                model.train()

        # End of epoch - print summary
        elapsed = time.time() - epoch_start
        print(f"  Epoch {epoch+1:3d} complete | time: {elapsed:.0f}s | step: {global_step}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="从 checkpoint 恢复训练")
    parser.add_argument("--model", type=str, default=None,
                        help="模型架构: base (完整单流) / turbo (锚点压缩)，默认读 config")
    args = parser.parse_args()
    train(resume_from=args.resume, model_type=args.model)
