"""
LyraSVC Model Base
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
# 1. Condition Encoders
# ============================================================

class DurationAwareContentEncoder(nn.Module):
    """PPG -> linear upsample -> mel frame-rate content features."""
    def __init__(self, input_dim=1280, hidden_dim=1280):
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
        return x, torch.zeros(ppg.shape[0], target_len, device=ppg.device)


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
# 2. DDPM helpers
# ============================================================

def extract(a, t, x_shape):
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# ============================================================
# 3. Spec normalization helpers
# ============================================================

SPEC_MIN, SPEC_MAX = -12.0, 5.0
SPEC_RANGE = SPEC_MAX - SPEC_MIN


def norm_spec(x):
    return (x - SPEC_MIN) / SPEC_RANGE * 2 - 1


def denorm_spec(x):
    return (x + 1) / 2 * SPEC_RANGE + SPEC_MIN


# ============================================================
# 4. YaRN Rotary Position Embedding
# ============================================================

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def _yarn_find_correction_dim(num_rotations, dim, base, max_pos):
    return (dim * math.log(max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

def _yarn_find_correction_range(low_rot, high_rot, dim, base, max_pos):
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_pos))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_pos))
    return max(low, 0), min(high, dim - 1)

def _yarn_linear_ramp_mask(min_val, max_val, dim):
    if min_val == max_val:
        max_val += 0.001
    linear = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    return torch.clamp(linear, 0, 1)

def _yarn_get_mscale(scale=1.0):
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


class RotaryPositionEmbedding(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        max_pos: int = 4096,
        base: float = 10000.0,
        scale: float = 1.0,
        original_max_pos: int = 384,
        extrapolation_factor: float = 1.0,
        attn_factor: float = 1.0,
        beta_fast: int = 32,
        beta_slow: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.max_pos = max_pos
        self.base = base
        self.scale = scale
        self.original_max_pos = original_max_pos
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        pos_freqs = base ** (torch.arange(0, dim, 2).float() / dim)
        inv_freq_ext = 1.0 / pos_freqs
        inv_freq_int = 1.0 / (scale * pos_freqs)

        low, high = _yarn_find_correction_range(beta_fast, beta_slow, dim, base, original_max_pos)
        mask = (1 - _yarn_linear_ramp_mask(low, high, dim // 2)) * extrapolation_factor
        inv_freq = inv_freq_int * (1 - mask) + inv_freq_ext * mask

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mscale = float(_yarn_get_mscale(scale) * attn_factor)

        self._cached_seq_len = max_pos
        t = torch.arange(self._cached_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", (emb.cos() * self.mscale), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * self.mscale), persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len > self._cached_seq_len:
            self._cached_seq_len = seq_len
            t = torch.arange(self._cached_seq_len, dtype=torch.float32, device=device)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer("cos_cached", (emb.cos() * self.mscale).to(dtype), persistent=False)
            self.register_buffer("sin_cached", (emb.sin() * self.mscale).to(dtype), persistent=False)

        cos = self.cos_cached[:seq_len].to(device=device, dtype=dtype)
        sin = self.sin_cached[:seq_len].to(device=device, dtype=dtype)
        return cos[None, None, :, :], sin[None, None, :, :]


# ============================================================
# 5. RMSNorm
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


# ============================================================
# 6. Transformer block
# ============================================================

class TransformerBlock(nn.Module):
    """Single-stream transformer block with fused QKV, QK-Norm, and RMSNorm."""
    def __init__(self, dim, n_heads=12, ffn_mult=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.norm1 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.rotary_emb = None

        self.norm2 = RMSNorm(dim)
        hidden_dim = int(dim * ffn_mult)
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.w3 = nn.Linear(dim, hidden_dim)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        # Attention path
        x_norm = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        B, N, D = x_norm.shape
        qkv = self.qkv(x_norm).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(N, q.device, q.dtype)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q = self.q_norm(q)
        k = self.k_norm(k)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.permute(0, 2, 1, 3).reshape(B, N, D)
        x = x + gate_msa.unsqueeze(1) * self.proj(attn)

        # FFN path (SwiGLU)
        x_norm = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.w2(F.silu(self.w1(x_norm)) * self.w3(x_norm))
        return x


# ============================================================
# 7. Timestep & Label embedders
# ============================================================

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
        use_cfg_embedding = True
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


# ============================================================
# 8. FinalLayer
# ============================================================

class FinalLayer(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.linear = nn.Linear(dim, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.linear(x)


# ============================================================
# 9. LyraModel — DDPM diffusion + Single-Stream Transformer
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
        content_dim: int = 1280,
        segment_len: int = 384,
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

        # --- Condition encoders ---
        self.content_enc = DurationAwareContentEncoder(ppg_dim, content_dim)
        self.pitch_enc = PitchEncoder(64, max_freq=pitch_max_freq)
        self.energy_enc = EnergyEncoder(32)
        self.speaker_enc = SpeakerEncoder(num_speakers, 256, use_ref_spk)

        # --- Timestep and label embedders ---
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.y_embedder = LabelEmbedder(num_speakers, hidden_dim, cfg_dropout_prob)

        # --- YaRN RoPE ---
        # S3-DiT concatenates 5 modalities in a single stream: total seq_len = 5 * T_mel
        head_dim = hidden_dim // num_heads
        self.segment_len = segment_len
        self.rotary_emb = RotaryPositionEmbedding(
            dim=head_dim,
            max_pos=16384,
            base=10000.0,
            scale=1.0,
            original_max_pos=segment_len * 5,
        )

        # --- Token projections for each modality ---
        self.content_token = nn.Linear(content_dim, hidden_dim)
        self.pitch_token = nn.Linear(64, hidden_dim)
        self.energy_token = nn.Linear(32, hidden_dim)
        self.speaker_token = nn.Linear(256, hidden_dim)
        self.mel_token = nn.Linear(mel_bins, hidden_dim)

        # --- Learnable modality embeddings ---
        self.modal_embed = nn.Parameter(torch.zeros(1, 5, hidden_dim))
        nn.init.normal_(self.modal_embed, std=0.02)

        # --- Transformer blocks ---
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ffn_mult=mlp_ratio)
            for _ in range(depth)
        ])
        for block in self.blocks:
            block.rotary_emb = self.rotary_emb

        self.final_layer = FinalLayer(hidden_dim, mel_bins)

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

    def set_rope_scale(self, scale: float):
        head_dim = self.hidden_dim // self.blocks[0].n_heads
        old = self.rotary_emb
        new_rotary = RotaryPositionEmbedding(
            dim=head_dim,
            max_pos=16384,
            base=old.base,
            scale=scale,
            original_max_pos=self.segment_len * 5,
            extrapolation_factor=old.extrapolation_factor,
            attn_factor=old.attn_factor,
            beta_fast=old.beta_fast,
            beta_slow=old.beta_slow,
        ).to(next(self.parameters()).device)
        self.rotary_emb = new_rotary
        for block in self.blocks:
            block.rotary_emb = new_rotary

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
    ) -> torch.Tensor:
        B, T_mel, _ = x.shape

        # 1. Encode conditions
        content, _ = self.content_enc(ppg, T_mel)
        pitch, _ = self.pitch_enc(f0, T_mel)
        energy = self.energy_enc(energy_mel)
        if energy.shape[1] > T_mel:
            energy = energy[:, :T_mel]
        elif energy.shape[1] < T_mel:
            energy = F.pad(energy, (0, 0, 0, T_mel - energy.shape[1]))

        speaker = self.speaker_enc(speaker_ids, ref_mel, T_mel)
        if speaker is None:
            speaker = torch.zeros(B, T_mel, 256, device=x.device)

        # 2. Tokenize each modality
        mel_tokens = self.mel_token(x) + self.modal_embed[:, 0:1, :]
        content_tokens = self.content_token(content) + self.modal_embed[:, 1:2, :]
        pitch_tokens = self.pitch_token(pitch) + self.modal_embed[:, 2:3, :]
        energy_tokens = self.energy_token(energy) + self.modal_embed[:, 3:4, :]
        speaker_tokens = self.speaker_token(speaker) + self.modal_embed[:, 4:5, :]

        # 3. Frame-interleaved concatenation: [c0,p0,e0,s0,m0, c1,p1,e1,s1,m1, ...]
        tokens = torch.stack([content_tokens, pitch_tokens, energy_tokens,
                              speaker_tokens, mel_tokens], dim=2).reshape(B, T_mel * 5, self.hidden_dim)

        # 4. Timestep and speaker embedding for adaLN
        t_emb = self.t_embedder(t)
        force_drop = torch.ones(B, device=x.device, dtype=torch.long) if force_uncond else None
        spk_emb = self.y_embedder(speaker_ids, self.training, force_drop_ids=force_drop)
        c = t_emb + spk_emb

        # 5. Transformer blocks
        for block in self.blocks:
            tokens = block(tokens, c)

        # 6. Output only mel portion (modality index 4 in interleaved layout)
        out = self.final_layer(tokens, c)
        out = out[:, 4::5, :]  # every 5th token starting at index 4
        return out

    def train_loss(self, mel_real, ppg, f0, energy_mel, speaker_ids=None, ref_mel=None):
        B = mel_real.shape[0]
        mel_real_n = norm_spec(mel_real.clamp(SPEC_MIN, SPEC_MAX))

        t = torch.randint(0, self.timesteps, (B,), device=mel_real.device)
        noise = torch.randn_like(mel_real_n)
        x_noisy = self.q_sample(mel_real_n, t, noise)

        eps_pred = self(x_noisy, t, ppg, f0, energy_mel, speaker_ids, ref_mel)

        eps_loss = F.mse_loss(eps_pred, noise)
        return eps_loss, eps_pred, noise

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
    ) -> torch.Tensor:
        B = ppg.shape[0]
        T_mel = length

        content, _ = self.content_enc(ppg, T_mel)
        pitch, _ = self.pitch_enc(f0, T_mel)
        energy = self.energy_enc(energy_mel)
        if energy.shape[1] > T_mel:
            energy = energy[:, :T_mel]
        elif energy.shape[1] < T_mel:
            energy = F.pad(energy, (0, 0, 0, T_mel - energy.shape[1]))

        speaker = self.speaker_enc(speaker_ids, ref_mel, T_mel)
        if speaker is None:
            speaker = torch.zeros(B, T_mel, 256, device=device)

        content_t = self.content_token(content) + self.modal_embed[:, 1:2, :]
        pitch_t = self.pitch_token(pitch) + self.modal_embed[:, 2:3, :]
        energy_t = self.energy_token(energy) + self.modal_embed[:, 3:4, :]
        speaker_t = self.speaker_token(speaker) + self.modal_embed[:, 4:5, :]

        from modules.dpm_solver import NoiseScheduleVP, DPM_Solver

        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas)

        def model_fn(x, t_continuous):
            t_discrete = (t_continuous - 1.0 / self.timesteps) * self.timesteps

            def _forward(x, t_d, force_uncond):
                t_emb = self.t_embedder(t_d)
                force_drop = torch.ones(B, device=x.device, dtype=torch.long) if force_uncond else None
                spk_emb = self.y_embedder(speaker_ids, False, force_drop_ids=force_drop)
                c = t_emb + spk_emb

                mel_t = self.mel_token(x) + self.modal_embed[:, 0:1, :]
                tokens = torch.stack([content_t, pitch_t, energy_t, speaker_t, mel_t], dim=2).reshape(B, T_mel * 5, self.hidden_dim)
                for block in self.blocks:
                    tokens = block(tokens, c)
                out = self.final_layer(tokens, c)
                return out[:, 4::5, :]

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

        return denorm_spec(mel)


# ============================================================
# 10. ModelConfig
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
    content_dim: int = 1280
    spec_min: float = -12.0
    spec_max: float = 5.0
    diffusion_timesteps: int = 1000
    diffusion_beta_start: float = 0.0001
    diffusion_beta_end: float = 0.02
    cfg_dropout_prob: float = 0.1
    dpm_steps: int = 20
    cfg_scale: float = 1.5
    learning_rate: float = 1e-4
    segment_len: int = 384

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
            content_dim=m.get("content_dim", 1280),
            diffusion_timesteps=d.get("timesteps", 1000),
            diffusion_beta_start=d.get("beta_start", 0.0001),
            diffusion_beta_end=d.get("beta_end", 0.02),
            cfg_dropout_prob=m.get("cfg_dropout_prob", 0.1),
            learning_rate=t.get("learning_rate", 1e-4),
            dpm_steps=inf.get("dpm_steps", 20),
            cfg_scale=inf.get("cfg_scale", 1.5),
            segment_len=t.get("segment_len", 384),
        )


# ============================================================
# 11. Quick test
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
        content_dim=cfg.content_dim,
        segment_len=cfg.segment_len,
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
    spk_ids = torch.zeros(B, dtype=torch.long).cuda()

    eps = model(mel_real, t, ppg, f0, energy_mel=mel_real, speaker_ids=spk_ids)
    print(f"Epsilon shape: {eps.shape}")

    eps_loss, _, _ = model.train_loss(
        mel_real, ppg, f0, energy_mel=mel_real, speaker_ids=spk_ids
    )
    print(f"Eps loss: {eps_loss.item():.4f}")

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Params: {params:.1f}M")
