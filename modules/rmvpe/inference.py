import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample
from .constants import *
from .model import E2E0, E2E
from .spec import MelSpectrogram 
from .utils import to_local_average_f0, to_viterbi_f0

class RMVPE:
    def __init__(self, model_path, hop_length=160):
        self.resample_kernel = {}
        model = E2E0(4, 1, (2, 2))
        try:
            ckpt = torch.load(model_path, map_location='cpu', weights_only=True)
        except Exception:
            ckpt = torch.load(model_path, map_location='cpu')
        state_dict = ckpt.get('model', ckpt)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        self.model = model
        self.hop_length = hop_length
        self.seg_length = 32 * hop_length
        self.mel_extractor = MelSpectrogram(N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length, None, MEL_FMIN, MEL_FMAX)
        self.resample_kernel = {}

    def mel2hidden(self, mel):
        with torch.no_grad():
            n_frames = mel.shape[-1]
            mel = F.pad(mel, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode='constant')
            hidden = self.model(mel)
            return hidden[:, :n_frames]

    def decode(self, hidden, thred=0.03, use_viterbi=False):
        if use_viterbi:
            f0 = to_viterbi_f0(hidden, thred=thred)
        else:
            f0 = to_local_average_f0(hidden, thred=thred)  
        return f0

    def infer_from_audio(self, audio, sample_rate=16000, device=None, thred=0.03, use_viterbi=False):
        f0s = self.infer_from_audio_batch([audio], sample_rate, device, thred, use_viterbi)
        return f0s[0]

    def infer_from_audio_batch(self, audios, sample_rate=16000, device=None, thred=0.03, use_viterbi=False):
        """Batched F0 extraction. audios: list of numpy arrays with shape (T,)"""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if isinstance(audios, np.ndarray):
            audios = [audios]

        orig_lens = [len(a) for a in audios]

        # Pad each audio to seg_length alignment first, then pad to max
        seg = self.seg_length
        padded_list = []
        for a in audios:
            t = torch.from_numpy(a).float()
            pad_len = (seg - (len(t) % seg)) % seg
            if pad_len > 0:
                t = F.pad(t, (0, pad_len))
            padded_list.append(t)

        max_len = max(len(p) for p in padded_list)
        audio = torch.zeros(len(audios), max_len, device=device)
        for i, p in enumerate(padded_list):
            audio[i, :len(p)] = p

        if sample_rate == 16000:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, 16000, lowpass_filter_width=128)
            self.resample_kernel[key_str] = self.resample_kernel[key_str].to(device)
            audio_res = self.resample_kernel[key_str](audio)

        B, T = audio_res.shape
        n_frames = T // self.hop_length + 1
        T1 = T + self.hop_length
        T_pad = seg * ((T1 - 1) // seg + 1) - T1
        audio_res = F.pad(audio_res, (0, T_pad))
        mel_extractor = self.mel_extractor.to(device)
        self.model = self.model.to(device)
        mel = mel_extractor(audio_res, center=True)
        with torch.no_grad():
            hidden = self.model(mel)

        f0s = []
        for i in range(B):
            nf_i = min(orig_lens[i] // self.hop_length + 1, hidden.shape[1])
            f0 = self.decode(hidden[i:i+1, :nf_i], thred=thred, use_viterbi=use_viterbi)
            f0s.append(f0)
        return f0s