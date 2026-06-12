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
from modules.slicer import Slicer, cross_fade
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
        content_dim=cfg.content_dim,
        segment_len=cfg.segment_len,
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

    # 5. Slice by silence and process each segment
    spk_t = torch.tensor([speaker_id], dtype=torch.long, device=device)
    T_mel = energy_mel.shape[0]
    train_len = cfg.segment_len
    mel_hop = 512
    output_sr = 44100

    # Load HiFiGAN vocoder once
    print("  Loading HiFiGAN vocoder...")
    voc_cfg = config.get("vocoder", {})
    voc_model_path = voc_cfg.get("model", "Models/pc_nsf_hifigan/model.ckpt")
    voc_config_path = voc_cfg.get("config", "Models/pc_nsf_hifigan/config.json")
    device_voc = voc_cfg.get("device", device)
    if not os.path.isabs(voc_model_path):
        base = os.path.dirname(os.path.abspath(__file__))
        voc_model_path = os.path.join(base, voc_model_path)
        voc_config_path = os.path.join(base, voc_config_path)
    hifigan = load_vocoder(voc_config_path, voc_model_path, device_voc)

    # Detect silent boundaries using already-loaded source audio
    slicer = Slicer(sr=sr_src, threshold=-40, min_length=5000,
                    min_interval=300, hop_size=10, max_sil_kept=500)
    chunk_dict = slicer.slice(wav_src)

    # Build ordered list of (start_sample, end_sample, is_silent)
    seg_samples = []
    for k in sorted(chunk_dict.keys(), key=int):
        v = chunk_dict[k]
        tag = v["split_time"].split(",")
        s, e = int(tag[0]), int(tag[1])
        if e > s:
            seg_samples.append((s, e, v["slice"]))

    if not seg_samples:
        seg_samples = [(0, len(wav_src), False)]

    total_wav_samples = len(wav_src)

    # Convert sample boundaries → mel frame boundaries
    seg_mel = []
    for s, e, is_sil in seg_samples:
        ms = int(s * T_mel / total_wav_samples)
        me = int(e * T_mel / total_wav_samples)
        if me > ms:
            seg_mel.append((ms, me, is_sil))

    if not seg_mel:
        seg_mel = [(0, T_mel, False)]

    print(f"  Sliced into {len(seg_mel)} segments by silence")

    # Process each segment: model.sample → vocoder → collect audio
    rope_scale_now = 1.0
    seg_audio_pieces = []  # (output_sample_start, wav_array)
    seg_mel_pieces = []    # (mel_start, mel_gen) for non-silent segments
    current_sample = 0

    for seg_idx, (mel_start, mel_end, is_silent) in enumerate(seg_mel):
        seg_T = mel_end - mel_start
        expected_sample_pos = mel_start * mel_hop

        if is_silent:
            silent_samples = seg_T * mel_hop
            seg_audio_pieces.append((expected_sample_pos, np.zeros(silent_samples, dtype=np.float32)))
            current_sample = expected_sample_pos + silent_samples
            print(f"  Segment {seg_idx+1}/{len(seg_mel)}: silence {seg_T} frames")
            continue

        # Slice pre-extracted features to this segment
        ppg_seg = ppg[int(mel_start * ppg.shape[0] / T_mel):int(mel_end * ppg.shape[0] / T_mel)]
        f0_seg = f0[int(mel_start * len(f0) / T_mel):int(mel_end * len(f0) / T_mel)]
        mel_seg = energy_mel[mel_start:mel_end]

        # YaRN rope scale for this segment
        if seg_T > train_len:
            new_scale = seg_T / train_len
        else:
            new_scale = 1.0
        if new_scale != rope_scale_now:
            model.set_rope_scale(new_scale)
            rope_scale_now = new_scale

        print(f"  Segment {seg_idx+1}/{len(seg_mel)} ({seg_T} frames)...", end=" ")

        ppg_t = torch.from_numpy(ppg_seg).float().unsqueeze(0).to(device)
        f0_t = torch.from_numpy(f0_seg).float().unsqueeze(0).to(device)
        energy_t = torch.from_numpy(mel_seg).float().unsqueeze(0).to(device)

        with torch.no_grad():
            mel_gen_seg, _ = model.sample(
                ppg_t, f0_t, energy_t, seg_T,
                speaker_ids=spk_t, dpm_steps=dpm_steps,
                cfg_scale=cfg_scale_val, device=device)
        mel_gen_seg = mel_gen_seg[0].cpu().float().numpy()

        mel_gen_seg = np.clip(mel_gen_seg, SPEC_MIN, SPEC_MAX)
        seg_mel_pieces.append((mel_start, mel_gen_seg))
        print(f"ok")

        # Vocode this segment
        uv_seg = f0_seg == 0
        if uv_seg.any() and (~uv_seg).any():
            f0_filled = f0_seg.copy()
            f0_filled[uv_seg] = np.interp(np.where(uv_seg)[0], np.where(~uv_seg)[0], f0_seg[~uv_seg])
            f0_mr = np.interp(np.linspace(0, 1, seg_T), np.linspace(0, 1, len(f0_filled)), f0_filled)
            uv_mr = np.interp(np.linspace(0, 1, seg_T), np.linspace(0, 1, len(uv_seg.astype(float))),
                              uv_seg.astype(float)) > 0.5
            f0_mr[uv_mr] = 0.0
        else:
            f0_mr = np.interp(np.linspace(0, 1, seg_T), np.linspace(0, 1, len(f0_seg)), f0_seg)

        wav_seg = vocode(mel_gen_seg, f0_mr, hifigan, device_voc)
        seg_audio_pieces.append((expected_sample_pos, wav_seg))
        current_sample = expected_sample_pos + len(wav_seg)

    # 6. Cross-fade concatenate (DDSP-SVC style)
    if len(seg_audio_pieces) == 1:
        wav_out = seg_audio_pieces[0][1]
    else:
        result = np.zeros(0, dtype=np.float32)
        cur_len = 0
        for start_sample, wav_seg in seg_audio_pieces:
            silent_len = start_sample - cur_len
            if silent_len >= 0:
                result = np.concatenate([result, np.zeros(silent_len, dtype=np.float32), wav_seg])
            else:
                result = cross_fade(result, wav_seg, cur_len + silent_len)
            cur_len = start_sample + len(wav_seg)
        wav_out = result

    print(f"    Output: {len(wav_out)/output_sr:.1f}s @ {output_sr}Hz")

    if save_mel:
        if seg_mel_pieces:
            full_mel_save = np.zeros((T_mel, 128), dtype=np.float32)
            for mel_start, mel_gen_seg in seg_mel_pieces:
                seg_T = mel_gen_seg.shape[0]
                full_mel_save[mel_start:mel_start + seg_T] = mel_gen_seg
            np.save(save_mel, full_mel_save)
            print(f"    Mel saved: {save_mel}")
        else:
            print(f"    No mel to save (all segments are silent)")

    sf.write(output_audio, wav_out, output_sr)
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
