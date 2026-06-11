import os
import sys
import numpy as np
import torch
import yaml
from typing import Optional, List, Union
from dataclasses import dataclass

from .nvSTFT import STFT


@dataclass
class MelResult:
    audio_path: str
    mel: np.ndarray


class MelExtractor:
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(current_dir, "config", "config.yaml")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        mel_cfg = config.get("mel", {})
        self.sample_rate = mel_cfg.get("sample_rate", 44100)
        self.n_fft = mel_cfg.get("n_fft", 2048)
        self.hop_length = mel_cfg.get("hop_length", 512)
        self.n_mels = mel_cfg.get("n_mels", 128)
        self.fmin = mel_cfg.get("fmin", 40)
        self.fmax = mel_cfg.get("fmax", 16000)

        self.stft = STFT(
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            win_size=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax,
        )

        data_cfg = config.get("data", {})
        self.data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            data_cfg.get("processed", "data"),
        )
        self.raw_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            data_cfg.get("raw", "data_raw"),
        )

    def extract(
        self,
        audio: Union[str, List[str]],
        on_progress: Optional[callable] = None,
    ) -> List[MelResult]:
        if isinstance(audio, str):
            audio = [audio]

        results = []
        for idx, audio_path in enumerate(audio):
            mel = self.stft(audio_path)  # 使用 STFT 自带的音频加载 (含归一化)
            results.append(MelResult(audio_path=audio_path, mel=mel))

            if on_progress:
                on_progress(idx + 1, len(audio))

        return results

    def get_frame_rate(self) -> float:
        return self.sample_rate / self.hop_length


if __name__ == "__main__":
    extractor = MelExtractor()

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
        print(f"Mel params: sr={extractor.sample_rate}, n_fft={extractor.n_fft}, "
              f"hop={extractor.hop_length}, n_mels={extractor.n_mels}")
        print(f"Frame rate: {extractor.get_frame_rate():.1f} Hz")
        print(f"Found {total} audio files across {len(audio_by_speaker)} speaker(s)")
        print()

        for speaker, audio_list in audio_by_speaker.items():
            output_dir = os.path.join(extractor.data_dir, speaker)
            os.makedirs(output_dir, exist_ok=True)

            def progress(done, _total):
                msg = f"  [{speaker}] extracting: {done}/{_total}"
                print(f"\r{msg:<50}", end="", flush=True)

            mel_results = extractor.extract(audio=audio_list, on_progress=progress)
            print()

            for r in mel_results:
                name = os.path.splitext(os.path.basename(r.audio_path))[0]
                save_path = os.path.join(output_dir, f"{name}_mel.npy")
                np.save(save_path, r.mel.cpu().numpy())

            shapes = [r.mel.shape for r in mel_results]
            avg_frames = sum(s[1] for s in shapes) / len(shapes)
            print(f"  [{speaker}] done: {len(mel_results)} files, "
                  f"avg_frames={avg_frames:.0f}")
