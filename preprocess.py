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

    total = sum(len(v) for v in audio_by_speaker.values())
    print(f"Found {total} files across {len(audio_by_speaker)} speaker(s)")
    print()

    for speaker, audio_list in audio_by_speaker.items():
        spk_out = os.path.join(out_dir, speaker)
        os.makedirs(spk_out, exist_ok=True)
        n = len(audio_list)
        print(f"[{speaker}] {n} files")

        # 1. PPG (Qwen3-ASR, ~3.5GB)
        print(f"  [1/3] PPG ...")
        ppg_ext = WhisperPPGExtractor()
        results = ppg_ext.extract(audio_list,
                                  on_progress=lambda d, t: print(f"\r    {d}/{t}", end="", flush=True))
        print()
        for r in results:
            name = os.path.splitext(os.path.basename(r.audio_path))[0]
            np.save(os.path.join(spk_out, f"{name}_ppg.npy"), r.ppg)
        del ppg_ext
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  [1/3] PPG ✓")

        # 2. F0 (RMVPE)
        print(f"  [2/3] F0  ...")
        pitch_ext = RMVPEExtractor()
        def pitch_progress(done, total):
            print(f"\r    {done}/{total}", end="", flush=True)
        results = pitch_ext.extract(audio_list, on_progress=pitch_progress)
        print()
        for r in results:
            name = os.path.splitext(os.path.basename(r.audio_path))[0]
            np.save(os.path.join(spk_out, f"{name}_f0.npy"), r.f0)
        del pitch_ext
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  [2/3] F0  ✓")

        # 3. Mel (纯 CPU, 无模型)
        print(f"  [3/3] Mel ...")
        mel_ext = MelExtractor()
        results = mel_ext.extract(audio_list,
                                  on_progress=lambda d, t: print(f"\r    {d}/{t}", end="", flush=True))
        print()
        for r in results:
            name = os.path.splitext(os.path.basename(r.audio_path))[0]
            np.save(os.path.join(spk_out, f"{name}_mel.npy"), r.mel.cpu().numpy())
        del mel_ext
        print(f"  [3/3] Mel ✓")

        # 汇总
        ppg_files = {f for f in os.listdir(spk_out) if f.endswith('_ppg.npy')}
        f0_files = {f for f in os.listdir(spk_out) if f.endswith('_f0.npy')}
        mel_files = {f for f in os.listdir(spk_out) if f.endswith('_mel.npy')}
        print(f"  Complete: {len(ppg_files)} PPG, {len(f0_files)} F0, {len(mel_files)} Mel")
        print()

    print("Done.")


if __name__ == "__main__":
    preprocess()
