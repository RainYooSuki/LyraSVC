"""
LyraSVC 音高提取 — 标准 RMVPE (E2E0 + local_average_f0)
"""

import os
import torch
import numpy as np
import yaml
from typing import Optional, List, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from .rmvpe.inference import RMVPE


@dataclass
class PitchResult:
    audio_path: str
    f0: np.ndarray
    time: np.ndarray


class RMVPEExtractor:
    def __init__(self, config_path: Optional[str] = None, device: str = None):
        if config_path is None:
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(current_dir, "config", "config.yaml")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        pitch_cfg = config.get("pitch", {})
        model_path = pitch_cfg.get("model", "Models/rmvpe/model.pt")
        self.batch_size = pitch_cfg.get("batch_size", 1)
        self.num_workers = pitch_cfg.get("num_workers", 4)

        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), model_path)

        if device is None:
            device = pitch_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.rmvpe = RMVPE(model_path, hop_length=160)

        data_cfg = config.get("data", {})
        self.data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            data_cfg.get("processed", "data"),
        )
        self.raw_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            data_cfg.get("raw", "data_raw"),
        )

    @torch.no_grad()
    def extract(
        self,
        audio: Union[str, List[str]],
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        on_progress: Optional[callable] = None,
    ) -> List[PitchResult]:
        if isinstance(audio, str):
            audio = [audio]

        import librosa
        bs = batch_size or self.batch_size
        seg_len = self.rmvpe.seg_length  # 5120 samples at 16kHz

        # 1. Load all audio, pad to seg_len alignment
        padded_wavs = []
        for audio_path in audio:
            wav, sr = librosa.load(audio_path, sr=16000, mono=True)
            wav = wav.astype(np.float32)
            wav = wav / max(np.abs(wav).max(), 1.0)
            orig_len = len(wav)
            pad = (seg_len - (orig_len % seg_len)) % seg_len
            if pad > 0:
                wav = np.pad(wav, (0, pad))
            padded_wavs.append((orig_len, wav))

        # 2. Group by exact padded length, batch within groups
        from collections import defaultdict
        groups = defaultdict(list)
        for i, (orig_len, wav) in enumerate(padded_wavs):
            groups[len(wav)].append((i, orig_len, wav))

        results_list = [None] * len(padded_wavs)
        processed = 0

        for length, items in groups.items():
            wavs_in_group = [w for _, _, w in items]
            for batch_start in range(0, len(wavs_in_group), bs):
                batch_wavs = wavs_in_group[batch_start:batch_start + bs]
                batch_items = items[batch_start:batch_start + bs]

                f0s = self.rmvpe.infer_from_audio_batch(
                    batch_wavs, sample_rate=16000, device=self.device, thred=0.03, use_viterbi=False
                )

                for (idx, orig_len, _), f0 in zip(batch_items, f0s):
                    nf = orig_len // self.rmvpe.hop_length + 1
                    f0_trimmed = f0[:nf] if len(f0) > nf else f0
                    times = librosa.frames_to_time(np.arange(len(f0_trimmed)), sr=16000, hop_length=160)
                    results_list[idx] = PitchResult(audio_path=audio[idx], f0=f0_trimmed, time=times)

                if on_progress:
                    processed += len(batch_items)
                    on_progress(processed, len(padded_wavs))

        return results_list

    @staticmethod
    def get_frame_rate() -> float:
        return 100.0


if __name__ == "__main__":
    extractor = RMVPEExtractor()

    supported_extensions = ('.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac')

    audio_by_speaker = {}
    if os.path.exists(extractor.raw_dir):
        for speaker in os.listdir(extractor.raw_dir):
            speaker_dir = os.path.join(extractor.raw_dir, speaker)
            if not os.path.isdir(speaker_dir):
                continue
            files = []
            for filename in os.listdir(speaker_dir):
                if filename.lower().endswith(supported_extensions):
                    files.append(os.path.join(speaker_dir, filename))
            if files:
                audio_by_speaker[speaker] = files

    if not audio_by_speaker:
        print(f"No audio files found in: {extractor.raw_dir}")
    else:
        total = sum(len(v) for v in audio_by_speaker.values())
        print(f"Found {total} audio files across {len(audio_by_speaker)} speaker(s)")
        print()

        for speaker, audio_list in audio_by_speaker.items():
            output_dir = os.path.join(extractor.data_dir, speaker)
            os.makedirs(output_dir, exist_ok=True)

            pitch_results = extractor.extract(audio=audio_list)
            print()

            for r in pitch_results:
                name = os.path.splitext(os.path.basename(r.audio_path))[0]
                save_path = os.path.join(output_dir, f"{name}_f0.npy")
                np.save(save_path, r.f0)

            voiced = sum(np.sum(r.f0 > 1.0) for r in pitch_results)
            total_frames = sum(len(r.f0) for r in pitch_results)
            print(f"  [{speaker}] done: {len(pitch_results)} files, "
                  f"voiced={voiced}/{total_frames} "
                  f"({100 * voiced / max(1, total_frames):.0f}%)")
