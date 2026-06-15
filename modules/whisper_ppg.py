import os
import torch
import numpy as np
import librosa
import yaml
from typing import Optional, List, Union, Callable
from dataclasses import dataclass

from transformers import WhisperProcessor, WhisperForConditionalGeneration


@dataclass
class WhisperPPGResult:
    audio_path: str
    ppg: np.ndarray


class WhisperPPGExtractor:
    """
    利用 Whisper Large v3 Turbo 的 Encoder 提取帧级 PPG 特征。
    取第 18 层中间隐藏状态 (0-indexed), 维度 (T, 1280), 帧率约 50 Hz。
    """

    def __init__(self, model_path: str = "Models/whisper-large-v3-turbo",
                 target_layer: int = 18, device: str = None):
        self.model_path = model_path
        self.target_layer = target_layer

        self.processor = WhisperProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device).eval()
        self.sr = 16000  # Whisper native sample rate

    @torch.no_grad()
    def extract(self, audio: Union[str, List[str]],
                on_progress: Optional[Callable] = None) -> List[WhisperPPGResult]:
        if isinstance(audio, str):
            audio = [audio]

        results = []
        for idx, path in enumerate(audio):
            wav, sr = librosa.load(path, sr=self.sr, mono=True)
            wav = wav.astype(np.float32)

            inputs = self.processor(wav, sampling_rate=self.sr, return_tensors="pt")
            inputs = {k: v.to(device=self.device, dtype=self.model.dtype) for k, v in inputs.items()}

            encoder_outputs = self.model.model.encoder(
                input_features=inputs["input_features"],
                output_hidden_states=True,
                return_dict=True,
            )

            hidden = encoder_outputs.hidden_states[self.target_layer + 1]
            ppg = hidden.squeeze(0).cpu().float().numpy()
            del encoder_outputs, hidden, inputs

            results.append(WhisperPPGResult(audio_path=path, ppg=ppg))

            if on_progress:
                on_progress(idx + 1, len(audio))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

    def get_frame_rate(self) -> float:
        return 50.0

    def get_feature_dim(self) -> int:
        return self.model.config.d_model


if __name__ == "__main__":
    extractor = WhisperPPGExtractor(
        model_path="Models/whisper-large-v3-turbo",
        target_layer=18,
    )
    print(f"Whisper PPG: dim={extractor.get_feature_dim()}, frame_rate={extractor.get_frame_rate()}")

    audio_dir = "data_raw/azi"
    files = sorted([os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')])[:2]

    for f in files:
        results = extractor.extract([f])
        ppg = results[0].ppg
        print(f"  {os.path.basename(f)}: ppg shape={ppg.shape}, mean={ppg.mean():.4f}, std={ppg.std():.4f}")

        out_dir = "data/azi"
        os.makedirs(out_dir, exist_ok=True)
        name = os.path.splitext(os.path.basename(f))[0]
        np.save(os.path.join(out_dir, f"{name}_ppg.npy"), ppg)
        print(f"    saved to data/azi/{name}_ppg.npy")
