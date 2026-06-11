"""
LyraSVC 端到端推理
输入:  源音频 (wav)
流程:  提取 PPG → 提取 F0 → 提取 Mel (能量) → LyraModel 生成 → HiFiGAN 合成
输出:  转换后的 wav
"""

import os
import sys
import yaml
import torch
import numpy as np
import librosa
import soundfile as sf

from modules.lyra_model import LyraModel, ModelConfig, SPEC_MIN, SPEC_MAX
from modules.whisper_ppg import WhisperPPGExtractor
from modules.pitch import RMVPEExtractor
from modules.mel import MelExtractor
from modules.vocoder import load_vocoder, vocode


def load_model(checkpoint_path: str, device: str = "cuda"):
    cfg = ModelConfig.from_yaml()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    speakers = ckpt.get("speakers", ["default"])
    model = LyraModel(
        num_speakers=len(speakers),
        ppg_dim=cfg.ppg_dim,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        mel_bins=cfg.mel_bins,
        pitch_max_freq=cfg.pitch_max_freq,
        use_ref_spk=cfg.use_ref_spk,
        rough_decoder_hidden=cfg.rough_decoder_hidden,
        spec_min=cfg.spec_min,
        spec_max=cfg.spec_max,
        diffusion_timesteps=cfg.diffusion_timesteps,
        beta_start=cfg.diffusion_beta_start,
        beta_end=cfg.diffusion_beta_end,
        cfg_dropout_prob=cfg.cfg_dropout_prob,
    ).to(device)

    # 推理用 EMA 权重 (如果存在)
    has_ema = 'ema_shadow' in ckpt
    is_ema_file = not has_ema and 'model_state_dict' in ckpt and 'optimizer_state_dict' not in ckpt
    if has_ema:
        model.load_state_dict(ckpt['ema_shadow'])
        print(f"  Using EMA weights (from training checkpoint)")
    elif is_ema_file:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Using EMA weights (from standalone file)")
    else:
        if 'model_state_dict' not in ckpt:
            raise KeyError("Checkpoint has no model weights")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Warning: no EMA weights, using raw training weights")
    model.eval()
    print(f"  Model: {checkpoint_path} (epoch {ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('val_loss', 0):.4f})")
    return model, cfg, ckpt


def convert(
    model: LyraModel,
    source_audio: str,
    output_audio: str,
    speaker_id: int = 0,
    steps: int = None,
    device: str = "cuda",
    save_mel: str = None,
    cfg_scale: float = None,
):
    """端到端 SVC: 源音频 → 特征提取 → LyraModel → HiFiGAN → 输出"""

    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    inf_cfg = config.get("inference", {})
    dpm_steps = steps or inf_cfg.get("dpm_steps", 20)
    cfg_scale_val = cfg_scale if cfg_scale is not None else inf_cfg.get("cfg_scale", 1.5)

    # 1. 加载源音频
    wav_src, sr_src = librosa.load(source_audio, sr=None)
    print(f"  Source: {source_audio} ({sr_src}Hz, {len(wav_src)/sr_src:.1f}s)")

    # 2. 提取 PPG (Whisper)
    print("  Extracting PPG...")
    ppg_ext = WhisperPPGExtractor()
    ppg_res = ppg_ext.extract([source_audio])
    ppg = ppg_res[0].ppg  # (1, T_ppg, 2048) or (T_ppg, 2048)
    if ppg.ndim == 3:
        ppg = ppg[0]
    print(f"    PPG: {ppg.shape}")
    # 卸载 ASR 模型释放显存
    del ppg_ext
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. 提取 F0 (RMVPE)
    print("  Extracting F0...")
    pitch_ext = RMVPEExtractor()
    f0_res = pitch_ext.extract([source_audio])
    f0 = f0_res[0].f0
    print(f"    F0: {f0.shape}, range=[{f0[f0>1].min():.0f}-{f0[f0>1].max():.0f}] Hz")

    # 4. 提取 Mel (能量参考)
    print("  Extracting Mel...")
    mel_ext = MelExtractor()
    mel_res = mel_ext.extract([source_audio])
    energy_mel = mel_res[0].mel.cpu().numpy().T  # (128, T) → (T, 128)
    print(f"    Mel: {energy_mel.shape}")

    # 5. LyraModel 生成目标 mel
    T_mel = energy_mel.shape[0]
    print(f"  Generating mel ({dpm_steps} DPM-Solver steps)...")
    ppg_t = torch.from_numpy(ppg).float().unsqueeze(0).to(device)
    f0_t = torch.from_numpy(f0).float().unsqueeze(0).to(device)
    energy_t = torch.from_numpy(energy_mel).float().unsqueeze(0).to(device)
    spk_t = torch.tensor([speaker_id], dtype=torch.long, device=device)

    with torch.no_grad():
        mel_gen, rough_mel = model.sample(
            ppg_t, f0_t, energy_t, T_mel,
            speaker_ids=spk_t,
            dpm_steps=dpm_steps,
            cfg_scale=cfg_scale_val,
            device=device
        )
    mel_gen = mel_gen[0].cpu().float().numpy()
    mel_gen = np.clip(mel_gen, SPEC_MIN, SPEC_MAX)
    print(f"    Generated mel: {mel_gen.shape}, mean={mel_gen.mean():.3f} std={mel_gen.std():.3f} "
          f"range=[{mel_gen.min():.3f}, {mel_gen.max():.3f}]")

    if save_mel:
        np.save(save_mel, mel_gen)
        print(f"    Mel saved: {save_mel}")

    # 6. HiFiGAN 声码器 → 音频
    print("  Loading HiFiGAN vocoder...")
    voc_cfg = config.get("vocoder", {})
    model_path = voc_cfg.get("model", "Models/pc_nsf_hifigan/model.ckpt")
    config_path = voc_cfg.get("config", "Models/pc_nsf_hifigan/config.json")
    device_voc = voc_cfg.get("device", device)

    if not os.path.isabs(model_path):
        base = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base, model_path)
        config_path = os.path.join(base, config_path)

    hifigan = load_vocoder(config_path, model_path, device_voc)
    uv_orig = f0 == 0
    if uv_orig.any() and (~uv_orig).any():
        f0_filled = f0.copy()
        f0_filled[uv_orig] = np.interp(np.where(uv_orig)[0], np.where(~uv_orig)[0], f0[~uv_orig])
        f0_melrate = np.interp(np.linspace(0, 1, T_mel), np.linspace(0, 1, len(f0_filled)), f0_filled)
        uv_melrate = np.interp(np.linspace(0, 1, T_mel), np.linspace(0, 1, len(uv_orig.astype(float))),
                               uv_orig.astype(float)) > 0.5
        f0_melrate[uv_melrate] = 0.0
    else:
        f0_melrate = np.interp(np.linspace(0, 1, T_mel), np.linspace(0, 1, len(f0)), f0)
    wav_out = vocode(mel_gen, f0_melrate, hifigan, device_voc)
    print(f"    Output: {len(wav_out)/44100:.1f}s @ 44100Hz")

    sf.write(output_audio, wav_out, 44100)
    print(f"  Saved: {output_audio}")

    return wav_out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LyraSVC: Singing Voice Conversion")
    parser.add_argument("--source", required=True, help="源音频文件 (.wav)")
    parser.add_argument("--output", required=True, help="输出音频文件 (.wav)")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt", help="LyraModel 权重")
    parser.add_argument("--speaker", type=int, default=0, help="目标说话人 ID")
    parser.add_argument("--steps", type=int, default=None, help="DPM-Solver 采样步数")
    parser.add_argument("--cfg", type=float, default=None, help="CFG scale")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-mel", default=None, help="中间 mel 保存路径")
    args = parser.parse_args()

    model, cfg, ckpt = load_model(args.checkpoint, args.device)
    convert(model, args.source, args.output,
            speaker_id=args.speaker,
            steps=args.steps,
            cfg_scale=args.cfg,
            device=args.device,
            save_mel=args.save_mel)
