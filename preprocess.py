"""
LyraSVC 数据预处理 — 一键完成 PPG + F0 + Mel 提取
输出: data/<speaker>/*_ppg.npy, *_f0.npy, *_mel.npy
"""

import os
import sys
import yaml
import torch
import numpy as np

from modules.whisper_ppg import WhisperPPGExtractor
from modules.pitch import RMVPEExtractor
from modules.mel import MelExtractor


def load_config(config_path="config/config.yaml"):
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_audio(raw_dir: str, extensions=('.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac')):
    """收集 data_raw 下所有音频，按说话人分组"""
    audio_by_speaker = {}
    if not os.path.exists(raw_dir):
        return audio_by_speaker
    for speaker in sorted(os.listdir(raw_dir)):
        spk_dir = os.path.join(raw_dir, speaker)
        if not os.path.isdir(spk_dir):
            continue
        files = []
        for fn in sorted(os.listdir(spk_dir)):
            if fn.lower().endswith(extensions):
                files.append(os.path.join(spk_dir, fn))
        if files:
            audio_by_speaker[speaker] = files
    return audio_by_speaker


def preprocess():
    config = load_config()
    data_cfg = config.get("data", {})
    raw_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           data_cfg.get("raw", "data_raw"))
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           data_cfg.get("processed", "data"))

    audio_by_speaker = collect_audio(raw_dir)
    if not audio_by_speaker:
        print(f"No audio found in {raw_dir}")
        return

    # 收集全部音频路径
    all_audio = []
    for speaker, audio_list in audio_by_speaker.items():
        spk_out = os.path.join(out_dir, speaker)
        os.makedirs(spk_out, exist_ok=True)
        for path in audio_list:
            all_audio.append((path, speaker, spk_out))

    # 过滤已存在的文件 (增量预处理)
    new_audio = []
    for path, speaker, spk_out in all_audio:
        name = os.path.splitext(os.path.basename(path))[0]
        if not os.path.exists(os.path.join(spk_out, f"{name}_mel.npy")):
            new_audio.append((path, speaker, spk_out))
    if not new_audio:
        print("All files already preprocessed, skipping.")
        return
    skipped = len(all_audio) - len(new_audio)
    print(f"Found {len(all_audio)} files ({skipped} skipped, {len(new_audio)} new)")
    all_audio = new_audio
    total = len(all_audio)

    # 1. PPG (Whisper) — 边提取边存盘, 避免内存累积
    print(f"[1/3] PPG (Whisper, {total} files)...")
    ppg_ext = WhisperPPGExtractor()
    for i, (path, _, spk_out) in enumerate(all_audio):
        results = ppg_ext.extract([path])
        name = os.path.splitext(os.path.basename(path))[0]
        np.save(os.path.join(spk_out, f"{name}_ppg.npy"), results[0].ppg)
        del results
        print(f"\r  {i+1}/{total}", end="", flush=True)
    print()
    del ppg_ext
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[1/3] PPG done\n")

    # 2. F0 (RMVPE) — 一次加载，处理全部
    print(f"[2/3] F0 (RMVPE, {total} files)...")
    pitch_ext = RMVPEExtractor()
    results = pitch_ext.extract([p for p, _, _ in all_audio],
                                on_progress=lambda d, t: print(f"\r  {d}/{t}", end="", flush=True))
    print()
    for (path, _, spk_out), r in zip(all_audio, results):
        name = os.path.splitext(os.path.basename(path))[0]
        np.save(os.path.join(spk_out, f"{name}_f0.npy"), r.f0)
    del pitch_ext
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[2/3] F0 done\n")

    # 3. Mel (nvSTFT) — 无模型，处理全部
    print(f"[3/3] Mel (nvSTFT, {total} files)...")
    mel_ext = MelExtractor()
    results = mel_ext.extract([p for p, _, _ in all_audio],
                              on_progress=lambda d, t: print(f"\r  {d}/{t}", end="", flush=True))
    print()
    for (path, _, spk_out), r in zip(all_audio, results):
        name = os.path.splitext(os.path.basename(path))[0]
        np.save(os.path.join(spk_out, f"{name}_mel.npy"), r.mel.cpu().numpy())
    del mel_ext
    print(f"[3/3] Mel done\n")

    # 汇总
    for speaker in audio_by_speaker:
        spk_out = os.path.join(out_dir, speaker)
        ppg_files = {f for f in os.listdir(spk_out) if f.endswith('_ppg.npy')}
        f0_files = {f for f in os.listdir(spk_out) if f.endswith('_f0.npy')}
        mel_files = {f for f in os.listdir(spk_out) if f.endswith('_mel.npy')}
        print(f"  [{speaker}] Complete: {len(ppg_files)} PPG, {len(f0_files)} F0, {len(mel_files)} Mel")

    print("Done.")


if __name__ == "__main__":
    preprocess()
