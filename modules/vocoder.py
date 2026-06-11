"""
LyraSVC 声码器 — NSF-HiFiGAN (标准版 + mini_nsf 版)
Mel 谱 + F0 → 音频波形
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils.parametrizations import weight_norm
import torch.nn.utils.parametrize as parametrize

LRELU_SLOPE = 0.1


def init_weights(m, mean=0.0, std=0.01):
    if m.__class__.__name__.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


# ============================================================
# SineGen / SourceModuleHnNSF — 标准版 F0 谐波激励
# ============================================================

class SineGen(nn.Module):
    def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1, noise_std=0.003, voiced_threshold=0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    @torch.no_grad()
    def forward(self, f0, upp):
        f0 = f0.unsqueeze(-1)
        rad = f0 / self.sampling_rate * torch.arange(1, upp + 1, device=f0.device)
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0)
        rad += F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        rad = rad.reshape(f0.shape[0], -1, 1)
        rad = torch.multiply(rad, torch.arange(1, self.dim + 1, device=f0.device).reshape(1, 1, -1))
        rand_ini = torch.rand(1, 1, self.dim, device=f0.device)
        rand_ini[..., 0] = 0
        rad += rand_ini
        sines = torch.sin(2 * np.pi * rad) * self.sine_amp
        uv = (f0 > self.voiced_threshold).float()
        uv = F.interpolate(uv.transpose(2, 1), scale_factor=upp, mode='nearest').transpose(2, 1)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sines)
        return sines * uv + noise


class SourceModuleHnNSF(nn.Module):
    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshold=0):
        super().__init__()
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x, upp):
        return self.l_tanh(self.l_linear(self.l_sin_gen(x, upp)))


# ============================================================
# ResBlock1
# ============================================================

class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=d,
                               padding=get_padding(kernel_size, d))) for d in dilation
        ])
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))) for _ in dilation
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            parametrize.remove_parametrizations(l, 'weight')
        for l in self.convs2:
            parametrize.remove_parametrizations(l, 'weight')


# ============================================================
# Generator — 标准版 (带 SourceModuleHnNSF + noise_convs)
# ============================================================

class Generator(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.mini_nsf = getattr(h, 'mini_nsf', False)
        self.noise_sigma = getattr(h, 'noise_sigma', None)

        if self.mini_nsf:
            self.source_sr = h.sampling_rate / int(np.prod(h.upsample_rates[2:]))
            self.upp = int(np.prod(h.upsample_rates[:2]))
        else:
            self.source_sr = h.sampling_rate
            self.upp = int(np.prod(h.upsample_rates))
            self.m_source = SourceModuleHnNSF(sampling_rate=h.sampling_rate, harmonic_num=8)
            self.noise_convs = nn.ModuleList()

        self.conv_pre = weight_norm(Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        ch = h.upsample_initial_channel
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            ch //= 2
            self.ups.append(weight_norm(ConvTranspose1d(2 * ch, ch, k, u, padding=(k - u) // 2)))
            for kk, d in zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes):
                self.resblocks.append(ResBlock1(ch, kk, d))
            if not self.mini_nsf:
                if i + 1 < len(h.upsample_rates):
                    stride_f0 = int(np.prod(h.upsample_rates[i + 1:]))
                    self.noise_convs.append(Conv1d(
                        1, ch, kernel_size=stride_f0 * 2, stride=stride_f0, padding=stride_f0 // 2))
                else:
                    self.noise_convs.append(Conv1d(1, ch, kernel_size=1))
            elif i == 1:
                self.source_conv = Conv1d(1, ch, 1)
                self.source_conv.apply(init_weights)

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def fastsinegen(self, f0):
        n = torch.arange(1, self.upp + 1, device=f0.device)
        s0 = f0.unsqueeze(-1) / self.source_sr
        ds0 = F.pad(s0[:, 1:, :] - s0[:, :-1, :], (0, 0, 0, 1))
        rad = s0 * n + 0.5 * ds0 * n * (n - 1) / self.upp
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0)
        rad += F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        rad = rad.reshape(f0.shape[0], 1, -1)
        return torch.sin(2 * np.pi * rad)

    def forward(self, x, f0):
        if self.mini_nsf:
            har_source = self.fastsinegen(f0)
        else:
            har_source = self.m_source(f0, self.upp).transpose(1, 2)

        x = self.conv_pre(x)
        if self.noise_sigma is not None and self.noise_sigma > 0:
            x += self.noise_sigma * torch.randn_like(x)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            if not self.mini_nsf:
                x = x + self.noise_convs[i](har_source)
            elif i == 1:
                x = x + self.source_conv(har_source)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        return torch.tanh(x)

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            parametrize.remove_parametrizations(l, 'weight')
        for l in self.resblocks:
            l.remove_weight_norm()
        parametrize.remove_parametrizations(self.conv_pre, 'weight')
        parametrize.remove_parametrizations(self.conv_post, 'weight')


# ============================================================
# Loader
# ============================================================

def _config_to_h(config_path):
    with open(config_path, "r") as f:
        cfg = json.load(f)
    cfg['sampling_rate'] = cfg.get('sampling_rate', 44100)
    cfg['resblock'] = cfg.get('resblock', '1')
    cfg['mini_nsf'] = cfg.get('mini_nsf', False)
    cfg['noise_sigma'] = cfg.get('noise_sigma', None)
    cfg.pop('discriminator_periods', None)
    cfg.pop('pc_aug', None)

    class H:
        def __init__(self, d): self.__dict__.update(d)
        def __getattr__(self, name): return self.__dict__.get(name, None)
    return H(cfg)


def load_vocoder(config_path: str, checkpoint_path: str, device: str = "cuda"):
    h = _config_to_h(config_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = Generator(h)
    model.load_state_dict(ckpt["generator"])
    model.to(device)
    model.eval()
    model.remove_weight_norm()
    return model


def vocode(mel, f0, model, device="cuda"):
    mel_t = torch.from_numpy(mel).float().unsqueeze(0).to(device)
    mel_t = mel_t.transpose(1, 2)
    f0_t = torch.from_numpy(f0).float().unsqueeze(0).to(device)
    with torch.no_grad():
        audio = model(mel_t, f0_t)
    return audio.squeeze().cpu().numpy()
