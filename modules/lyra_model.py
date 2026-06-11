"""
LyraSVC Model — DDPM Diffusion + Official DiT Backbone + Condition Encoders
=============================================================================
Architecture:
  1. Condition Encoders (kept from original): Content, Pitch, Energy, Speaker
  2. RoughMelDecoder: content + pitch → coarse mel (zero-init output)
  3. Official Facebook DiT backbone: adaLN-Zero Transformer blocks
  4. Per-frame condition injection + global adaLN condition summary
  5. DDPM epsilon-prediction training + DDIM sampling with CFG
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from copy import deepcopy


# ============================================================
# 0. EMA class
# ============================================================

class EMA(nn.Module):
    """Exponential Moving Average: maintains a shadow copy of model weights."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self.shadow = deepcopy(model.state_dict())

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.state_dict().items():
            if param.dtype.is_floating_point:
                self.shadow[name] = self.shadow[name].to(param.device)
                self.shadow[name].lerp_(param, 1 - self.decay)

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=False)


# ============================================================
# 1. Condition Encoders (kept from original)
# ============================================================

class DurationAwareContentEncoder(nn.Module):
    """PPG → linear upsample → mel frame-rate content features."""
    def __init__(self, input_dim=1280, hidden_dim=256):
        super().__init__()
        self.pre_proj = nn.Conv1d(input_dim, hidden_dim, 1)
        self.post_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, ppg: torch.Tensor, target_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = ppg.transpose(1, 2)
        x = self.pre_proj(x)
        x = F.interpolate(x, size=target_len, mode='linear')
        x = self.post_conv(x)
        x = x.transpose(1, 2)
        return x, torch.zeros(ppg.shape[0], ppg.shape[1], device=ppg.device)


class ContinuousFreqEmbed(nn.Module):
    def __init__(self, dim: int, min_freq=32.7, max_freq=2000.0):
        super().__init__()
        self.dim = dim
        self.min_freq = min_freq
        self.max_freq = max_freq
        norm_factor = np.log2(max_freq / min_freq)
        self.register_buffer('inv_norm', torch.tensor(1.0 / norm_factor))
        self.linear = nn.Linear(dim, dim)

    def forward(self, f0_hz: torch.Tensor) -> torch.Tensor:
        log_f0 = torch.log2(torch.clamp(f0_hz, min=1.0) / self.min_freq)
        norm = log_f0 * self.inv_norm
        norm = norm.clamp(0, 1)
        half = self.dim // 2
        freqs = 2.0 ** torch.linspace(0, 7, half, device=f0_hz.device) * np.pi
        emb = torch.cat([torch.sin(norm.unsqueeze(-1) * freqs),
                         torch.cos(norm.unsqueeze(-1) * freqs)], dim=-1)
        return self.linear(emb)


class PitchEncoder(nn.Module):
    def __init__(self, hidden_dim=64, voiced_threshold=50.0, max_freq=2000.0):
        super().__init__()
        self.voiced_threshold = voiced_threshold
        self.freq_embed = ContinuousFreqEmbed(hidden_dim, max_freq=max_freq)

    def forward(self, f0: torch.Tensor, target_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.interpolate(f0.unsqueeze(1), size=target_len, mode='linear').squeeze(1)
        voiced = (x > self.voiced_threshold).float().unsqueeze(-1)
        return self.freq_embed(x) * voiced, voiced


class EnergyEncoder(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(1, hidden_dim, 5, padding=2),
            nn.GroupNorm(4, hidden_dim), nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2),
            nn.GroupNorm(4, hidden_dim), nn.SiLU(),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        energy = torch.sqrt(torch.mean(mel ** 2, dim=-1, keepdim=True) + 1e-8)
        energy = torch.log1p(energy)
        x = energy.transpose(1, 2)
        return self.proj(x).transpose(1, 2)


class SpeakerEncoder(nn.Module):
    def __init__(self, num_speakers: int, dim: int = 256, use_reference: bool = True):
        super().__init__()
        self.use_reference = use_reference
        self.lookup = nn.Embedding(num_speakers, dim)

        if use_reference:
            self.ref_conv = nn.Sequential(
                nn.Conv2d(1, 32, (3, 3), stride=(2, 1), padding=(1, 1)),
                nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 64, (3, 3), stride=(2, 1), padding=(1, 1)),
                nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 128, (3, 3), stride=(2, 1), padding=(1, 1)),
                nn.BatchNorm2d(128), nn.ReLU(),
                nn.Conv2d(128, 256, (3, 3), stride=(2, 1), padding=(1, 1)),
                nn.BatchNorm2d(256), nn.ReLU(),
            )
            self.attn_head = nn.Conv1d(256, 1, 1)
            self.ref_proj = nn.Linear(256, dim)
            self.gate = nn.Linear(dim * 2, dim)

    def _encode_ref(self, ref_mel: torch.Tensor) -> torch.Tensor:
        x = ref_mel.unsqueeze(1).transpose(2, 3)
        feats = self.ref_conv(x)
        pooled = feats.mean(dim=2)
        w = F.softmax(self.attn_head(pooled).squeeze(1), dim=1)
        emb = (pooled * w.unsqueeze(1)).sum(dim=-1)
        return self.ref_proj(emb)

    def forward(self, speaker_ids=None, ref_mel=None, target_len=None):
        if speaker_ids is not None:
            spk = self.lookup(speaker_ids)
        elif ref_mel is not None:
            B = ref_mel.shape[0]
            spk = torch.zeros(B, self.lookup.weight.shape[1], device=ref_mel.device)
        else:
            return None

        if self.use_reference and ref_mel is not None:
            ref_vec = self._encode_ref(ref_mel)
            g = self.gate(torch.cat([spk, ref_vec], dim=-1)).sigmoid()
            spk = g * spk + (1 - g) * ref_vec

        if target_len is not None:
            spk = spk.unsqueeze(1).expand(-1, target_len, -1)
        return spk


# ============================================================
# 2. RoughMelDecoder (zero-init output)
# ============================================================

class RoughMelDecoder(nn.Module):
    def __init__(self, content_dim=256, pitch_dim=64, hidden=256, mel_bins=128):
        super().__init__()
        self.content_proj = nn.Linear(content_dim, hidden)
        self.pitch_proj = nn.Linear(pitch_dim, hidden)
        self.convs = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv1d(hidden, mel_bins, 3, padding=1),
        )
        nn.init.zeros_(self.convs[-1].weight)
        nn.init.zeros_(self.convs[-1].bias)

    def forward(self, content, pitch):
        x = self.content_proj(content) + self.pitch_proj(pitch)
        x = x.transpose(1, 2)
        x = self.convs(x)
        x = x.transpose(1, 2)
        return x.clamp(-12.0, 5.0)


# ============================================================
# 3. DDPM helpers
# ============================================================

def extract(a, t, x_shape):
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# ============================================================
# 4. Spec normalization helpers
# ============================================================

SPEC_MIN, SPEC_MAX = -12.0, 5.0
SPEC_RANGE = SPEC_MAX - SPEC_MIN


def norm_spec(x):
    return (x - SPEC_MIN) / SPEC_RANGE * 2 - 1


def denorm_spec(x):
    return (x + 1) / 2 * SPEC_RANGE + SPEC_MIN


# ============================================================
# 5. Position embedding (1D sincos, adapted from official 2D)
# ============================================================

def get_1d_sincos_pos_embed(embed_dim, length):
    pos = np.arange(length, dtype=np.float32)
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    if embed_dim % 2:
        emb = np.concatenate([emb, np.zeros((length, 1), dtype=np.float32)], axis=1)
    return emb


# ============================================================
# 6. Official DiT components (from Facebook Research)
# ============================================================

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        B, T, D = x_norm.shape
        q = self.q_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        attn_out = self.out_proj(attn_out)
        x = x + gate_msa.unsqueeze(1) * attn_out
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# ============================================================
# 7. LyraModel — DDPM diffusion + Official DiT backbone
# ============================================================

class LyraModel(nn.Module):
    def __init__(
        self,
        num_speakers: int = 1,
        ppg_dim: int = 1280,
        hidden_dim: int = 1024,
        depth: int = 12,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        mel_bins: int = 128,
        pitch_max_freq: float = 2000.0,
        use_ref_spk: bool = True,
        rough_decoder_hidden: int = 256,
        spec_min: float = -12.0,
        spec_max: float = 5.0,
        diffusion_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        cfg_dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.mel_bins = mel_bins
        self.hidden_dim = hidden_dim
        self.timesteps = diffusion_timesteps
        self.spec_min = spec_min
        self.spec_max = spec_max

        # --- Condition encoders (kept from original) ---
        self.content_enc = DurationAwareContentEncoder(ppg_dim, 256)
        self.pitch_enc = PitchEncoder(64, max_freq=pitch_max_freq)
        self.energy_enc = EnergyEncoder(32)
        self.speaker_enc = SpeakerEncoder(num_speakers, 256, use_ref_spk)

        # --- Rough mel decoder ---
        self.rough_decoder = RoughMelDecoder(256, 64, rough_decoder_hidden, mel_bins)

        # --- Condition injection projections (to hidden_dim) ---
        self.content_inject = nn.Linear(256, hidden_dim)
        self.pitch_inject = nn.Linear(64, hidden_dim)
        self.energy_inject = nn.Linear(32, hidden_dim)
        self.speaker_inject = nn.Linear(256, hidden_dim)
        self.rough_mel_inject = nn.Linear(mel_bins, hidden_dim)

        # --- DiT components ---
        self.x_embedder = nn.Linear(mel_bins, hidden_dim)
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.y_embedder = LabelEmbedder(num_speakers, hidden_dim, cfg_dropout_prob)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_dim, mel_bins)

        # --- Position embedding (1D sincos, frozen) ---
        max_pos_len = 4096
        pos_embed_np = get_1d_sincos_pos_embed(hidden_dim, max_pos_len)
        self.register_buffer('pos_embed', torch.from_numpy(pos_embed_np).float().unsqueeze(0))

        # --- Build noise schedule ---
        self._build_noise_schedule(diffusion_timesteps, beta_start, beta_end)

        # --- Weight initialization ---
        self._init_weights()

    def _build_noise_schedule(self, timesteps, beta_start, beta_end):
        betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        self.register_buffer('betas', torch.from_numpy(betas).float())
        self.register_buffer('alphas_cumprod', torch.from_numpy(alphas_cumprod).float())
        self.register_buffer('sqrt_alphas_cumprod', torch.from_numpy(np.sqrt(alphas_cumprod)).float())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.from_numpy(np.sqrt(1 - alphas_cumprod)).float())

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def q_sample(self, x_start, t, noise):
        a = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        b = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return a * x_start + b * noise

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        ppg: torch.Tensor,
        f0: torch.Tensor,
        energy_mel: torch.Tensor,
        speaker_ids: Optional[torch.Tensor] = None,
        ref_mel: Optional[torch.Tensor] = None,
        force_uncond: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape

        # 1. Encode conditions
        content, _ = self.content_enc(ppg, T)
        pitch, _ = self.pitch_enc(f0, T)
        energy = self.energy_enc(energy_mel)
        if energy.shape[1] > T:
            energy = energy[:, :T]
        elif energy.shape[1] < T:
            energy = F.pad(energy, (0, 0, 0, T - energy.shape[1]))

        speaker_raw = self.speaker_enc(speaker_ids, ref_mel, T)
        if speaker_raw is None:
            speaker_raw = torch.zeros(B, T, 256, device=x.device)

        # 2. Rough mel (condition)
        rough_mel = self.rough_decoder(content, pitch)

        # 3. Build per-frame condition vector
        cond_inj = (
            self.content_inject(content)
            + self.pitch_inject(pitch)
            + self.energy_inject(energy)
            + self.speaker_inject(speaker_raw)
            + self.rough_mel_inject(rough_mel)
        )

        # 4. Timestep embedding
        t_emb = self.t_embedder(t)

        # 5. Speaker label embedding (for CFG)
        force_drop = torch.ones(B, device=x.device, dtype=torch.long) if force_uncond else None
        spk_emb = self.y_embedder(speaker_ids, self.training, force_drop_ids=force_drop)

        # 6. Combine: c = t + spk + cond_summary
        cond_summary = cond_inj.mean(dim=1)
        c = t_emb + spk_emb + cond_summary

        # 7. DiT forward
        x = self.x_embedder(x)
        x = x + self.pos_embed[:, :T, :]
        x = x + cond_inj

        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x, c)
        return x, rough_mel

    def train_loss(self, mel_real, ppg, f0, energy_mel, speaker_ids=None, ref_mel=None):
        B = mel_real.shape[0]
        mel_real_n = norm_spec(mel_real.clamp(SPEC_MIN, SPEC_MAX))

        t = torch.randint(0, self.timesteps, (B,), device=mel_real.device)
        noise = torch.randn_like(mel_real_n)
        x_noisy = self.q_sample(mel_real_n, t, noise)

        eps_pred, rough_mel = self(x_noisy, t, ppg, f0, energy_mel, speaker_ids, ref_mel)

        eps_loss = F.mse_loss(eps_pred, noise)
        rough_loss = F.mse_loss(rough_mel, mel_real.clamp(SPEC_MIN, SPEC_MAX))
        return eps_loss, rough_loss, eps_pred, noise

    @torch.no_grad()
    def sample(
        self,
        ppg: torch.Tensor,
        f0: torch.Tensor,
        energy_mel: torch.Tensor,
        length: int,
        speaker_ids: Optional[torch.Tensor] = None,
        ref_mel: Optional[torch.Tensor] = None,
        dpm_steps: int = 20,
        cfg_scale: float = 1.5,
        device: str = "cuda",
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B = ppg.shape[0]
        T_mel = length

        content, _ = self.content_enc(ppg, T_mel)
        pitch, _ = self.pitch_enc(f0, T_mel)
        energy = self.energy_enc(energy_mel)
        if energy.shape[1] > T_mel:
            energy = energy[:, :T_mel]
        elif energy.shape[1] < T_mel:
            energy = F.pad(energy, (0, 0, 0, T_mel - energy.shape[1]))

        rough_mel = self.rough_decoder(content, pitch)

        cond_inj = (
            self.content_inject(content) + self.pitch_inject(pitch)
            + self.energy_inject(energy) + self.rough_mel_inject(rough_mel)
        )

        speaker_raw = self.speaker_enc(speaker_ids, ref_mel, T_mel)
        if speaker_raw is not None:
            cond_inj = cond_inj + self.speaker_inject(speaker_raw)

        cond_summary = cond_inj.mean(dim=1)

        from modules.dpm_solver import NoiseScheduleVP, DPM_Solver

        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas)

        def model_fn(x, t_continuous):
            t_discrete = (t_continuous - 1.0 / self.timesteps) * self.timesteps

            def _forward(x, t_d, force_uncond):
                t_emb = self.t_embedder(t_d)
                force_drop = torch.ones(B, device=x.device, dtype=torch.long) if force_uncond else None
                spk_emb = self.y_embedder(speaker_ids, False, force_drop_ids=force_drop)
                c = t_emb + spk_emb + cond_summary

                x_proj = self.x_embedder(x) + self.pos_embed[:, :T_mel, :] + cond_inj
                for block in self.blocks:
                    x_proj = block(x_proj, c)
                return self.final_layer(x_proj, c)

            if cfg_scale != 1.0:
                eps_cond = _forward(x, t_discrete, force_uncond=False)
                eps_uncond = _forward(x, t_discrete, force_uncond=True)
                return eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            else:
                return _forward(x, t_discrete, force_uncond=False)

        dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")

        x_T = torch.randn(B, T_mel, self.mel_bins, device=device)

        mel = dpm_solver.sample(
            x_T, steps=dpm_steps,
            order=2, skip_type="time_uniform", method="multistep",
        )

        return denorm_spec(mel), rough_mel


# ============================================================
# 8. ModelConfig
# ============================================================

@dataclass
class ModelConfig:
    num_speakers: int = 1
    ppg_dim: int = 1280
    hidden_dim: int = 1024
    depth: int = 12
    num_heads: int = 16
    mlp_ratio: float = 4.0
    mel_bins: int = 128
    pitch_max_freq: float = 2000.0
    use_ref_spk: bool = True
    rough_decoder_hidden: int = 256
    spec_min: float = -12.0
    spec_max: float = 5.0
    diffusion_timesteps: int = 1000
    diffusion_beta_start: float = 0.0001
    diffusion_beta_end: float = 0.02
    cfg_dropout_prob: float = 0.1
    rough_loss_weight: float = 1.0
    dpm_steps: int = 20
    cfg_scale: float = 1.5
    learning_rate: float = 1e-4

    @classmethod
    def from_yaml(cls, config_path: str = "config/config.yaml") -> "ModelConfig":
        import yaml
        import os
        if not os.path.isabs(config_path):
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                config_path
            )
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        m = cfg.get("model", {})
        d = cfg.get("diffusion", {})
        t = cfg.get("train", {})
        inf = cfg.get("inference", {})
        return cls(
            num_speakers=m.get("num_speakers", 1),
            ppg_dim=m.get("ppg_dim", 1280),
            hidden_dim=m.get("hidden_dim", 1024),
            depth=m.get("depth", 12),
            num_heads=m.get("num_heads", 16),
            mlp_ratio=m.get("mlp_ratio", 4.0),
            mel_bins=m.get("mel_bins", 128),
            pitch_max_freq=m.get("pitch_max_freq", 2000.0),
            use_ref_spk=m.get("use_ref_spk", True),
            spec_min=m.get("spec_min", -12.0),
            spec_max=m.get("spec_max", 5.0),
            rough_decoder_hidden=m.get("rough_decoder_hidden", 256),
            diffusion_timesteps=d.get("timesteps", 1000),
            diffusion_beta_start=d.get("beta_start", 0.0001),
            diffusion_beta_end=d.get("beta_end", 0.02),
            cfg_dropout_prob=m.get("cfg_dropout_prob", 0.1),
            rough_loss_weight=t.get("rough_loss_weight", 1.0),
            learning_rate=t.get("learning_rate", 1e-4),
            dpm_steps=inf.get("dpm_steps", 20),
            cfg_scale=inf.get("cfg_scale", 1.5),
        )


# ============================================================
# 9. Quick test
# ============================================================

if __name__ == "__main__":
    cfg = ModelConfig.from_yaml()
    model = LyraModel(
        num_speakers=cfg.num_speakers,
        ppg_dim=cfg.ppg_dim,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        mel_bins=cfg.mel_bins,
        pitch_max_freq=cfg.pitch_max_freq,
        use_ref_spk=cfg.use_ref_spk,
        rough_decoder_hidden=cfg.rough_decoder_hidden,
        diffusion_timesteps=cfg.diffusion_timesteps,
        beta_start=cfg.diffusion_beta_start,
        beta_end=cfg.diffusion_beta_end,
        cfg_dropout_prob=cfg.cfg_dropout_prob,
    )
    model.cuda()

    B, T_ppg, T_f0, T_mel = 1, 30, 200, 170
    ppg = torch.randn(B, T_ppg, cfg.ppg_dim).cuda()
    f0 = torch.rand(B, T_f0).cuda() * 400 + 100
    mel_real = torch.randn(B, T_mel, 128).cuda()
    t = torch.randint(0, model.timesteps, (B,)).cuda()

    eps, rough = model(mel_real, t, ppg, f0, energy_mel=mel_real)
    print(f"Epsilon shape: {eps.shape}")
    print(f"Rough mel shape: {rough.shape}")

    eps_loss, rough_loss, _, _ = model.train_loss(
        mel_real, ppg, f0, energy_mel=mel_real
    )
    print(f"Eps loss: {eps_loss.item():.4f}, Rough loss: {rough_loss.item():.4f}")

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Params: {params:.1f}M")
