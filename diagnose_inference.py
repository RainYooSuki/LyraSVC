"""
Diagnose inference pipeline for DDPM-based LyraModel.
Checks rough mel, generated mel value distribution, clipping, and compares with source.
"""
import torch
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.lyra_model import LyraModel, ModelConfig, norm_spec, denorm_spec, SPEC_MIN, SPEC_MAX, SPEC_RANGE

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ─── 0. Load Model ───────────────────────────────────────────
cfg = ModelConfig.from_yaml("config/config.yaml")
ckpt = torch.load("checkpoints/best_ema.pt", map_location="cpu", weights_only=False)
speakers = ckpt.get("speakers", ["default"])

model = LyraModel(
    num_speakers=len(speakers),
    ppg_dim=cfg.ppg_dim, hidden_dim=cfg.hidden_dim,
    depth=cfg.depth, num_heads=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
    mel_bins=cfg.mel_bins,
    pitch_max_freq=cfg.pitch_max_freq, use_ref_spk=cfg.use_ref_spk,
    rough_decoder_hidden=cfg.rough_decoder_hidden,
    spec_min=cfg.spec_min, spec_max=cfg.spec_max,
    diffusion_timesteps=cfg.diffusion_timesteps,
    beta_start=cfg.diffusion_beta_start, beta_end=cfg.diffusion_beta_end,
    cfg_dropout_prob=cfg.cfg_dropout_prob,
).to(device)

model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
print(f"  diffusion_timesteps={cfg.diffusion_timesteps}, beta=[{cfg.diffusion_beta_start}, {cfg.diffusion_beta_end}]")

# ─── 1. Load preprocessed features ───────────────────────────
ppg = np.load("data/azi/azi2_1_ppg.npy")
f0 = np.load("data/azi/azi2_1_f0.npy")
energy_mel = np.load("data/azi/azi2_1_mel.npy")
energy_mel = energy_mel.T

print(f"\nFeatures:")
print(f"  PPG: {ppg.shape}, mean={ppg.mean():.4f}, std={ppg.std():.4f}")
print(f"  F0:  {f0.shape}, mean={f0.mean():.1f}, nonzero_mean={f0[f0>1].mean():.1f}")
print(f"  Mel (energy): {energy_mel.shape}, mean={energy_mel.mean():.4f}, std={energy_mel.std():.4f}, "
      f"min={energy_mel.min():.4f}, max={energy_mel.max():.4f}")

T_mel = energy_mel.shape[0]

# ─── 2. Convert to tensors ───────────────────────────────────
ppg_t = torch.from_numpy(ppg).float().unsqueeze(0).to(device)
f0_t = torch.from_numpy(f0).float().unsqueeze(0).to(device)
energy_t = torch.from_numpy(energy_mel).float().unsqueeze(0).to(device)
spk_t = torch.tensor([0], dtype=torch.long, device=device)

# ─── 3. Run DDPM sample ─────────────────────────────────────
dpm_steps = 4
print(f"\n{'='*60}")
print(f"DPM-Solver: dpm_steps={dpm_steps}")
print(f"{'='*60}\n")

with torch.no_grad():
    mel_gen, rough_mel = model.sample(
        ppg_t, f0_t, energy_t, T_mel,
        speaker_ids=spk_t, dpm_steps=dpm_steps, cfg_scale=1.0, device=device
    )

print(f"[Rough mel]")
print(f"  raw:            mean={rough_mel.mean():.4f}, std={rough_mel.std():.4f}, "
      f"min={rough_mel.min():.4f}, max={rough_mel.max():.4f}")
rough_norm = norm_spec(rough_mel)
print(f"  norm_spec:      mean={rough_norm.mean():.4f}, std={rough_norm.std():.4f}, "
      f"min={rough_norm.min():.4f}, max={rough_norm.max():.4f}")

print(f"\n[Generated mel (sample output)]")
print(f"  raw:            mean={mel_gen.mean():.4f}, std={mel_gen.std():.4f}, "
      f"min={mel_gen.min():.4f}, max={mel_gen.max():.4f}")

mel_norm = norm_spec(mel_gen)
print(f"  norm_spec:      mean={mel_norm.mean():.4f}, std={mel_norm.std():.4f}, "
      f"min={mel_norm.min():.4f}, max={mel_norm.max():.4f}")

# ─── 4. Denorm & clip (treat sample output as final) ─────────
denormed = denorm_spec(mel_norm)
print(f"\n[Denormed (before clip)]")
print(f"  mean={denormed.mean():.4f}, std={denormed.std():.4f}, "
      f"min={denormed.min():.4f}, max={denormed.max():.4f}")

clipped = torch.clamp(denormed, SPEC_MIN, SPEC_MAX)
print(f"\n[Clipped to [{SPEC_MIN}, {SPEC_MAX}]]")
print(f"  mean={clipped.mean():.4f}, std={clipped.std():.4f}, "
      f"min={clipped.min():.4f}, max={clipped.max():.4f}")

# Distribution
n_total = clipped.numel()
n_ceiling = (clipped >= (SPEC_MAX - 0.01)).sum().item()
n_floor = (clipped <= (SPEC_MIN + 0.01)).sum().item()
print(f"\n  Values at ceiling {SPEC_MAX}: {n_ceiling}/{n_total} ({100*n_ceiling/n_total:.1f}%)")
print(f"  Values at floor {SPEC_MIN}: {n_floor}/{n_total} ({100*n_floor/n_total:.1f}%)")

# Histogram bucket analysis
vals = clipped.cpu().flatten().numpy()
print(f"\n[Histogram buckets (denorm, clipped)]:")
step = (SPEC_MAX - SPEC_MIN) / 5
buckets = [(SPEC_MIN + i*step, SPEC_MIN + (i+1)*step) for i in range(5)]
for lo, hi in buckets:
    n = ((vals >= lo) & (vals < hi)).sum()
    print(f"  [{lo:5.1f}, {hi:5.1f}): {n:6d} ({100*n/len(vals):5.1f}%)")

# ─── 5. Compare with source ──────────────────────────────────
print(f"\n{'='*60}")
print("Comparison with source energy mel")
print(f"{'='*60}")
print(f"  Source mel (log-mel space): mean={energy_mel.mean():.4f}, std={energy_mel.std():.4f}, "
      f"min={energy_mel.min():.4f}, max={energy_mel.max():.4f}")
print(f"  Generated mel (clipped):   mean={clipped.mean():.4f}, std={clipped.std():.4f}, "
      f"min={clipped.min():.4f}, max={clipped.max():.4f}")

# Check normalization/denormalization consts
print(f"\n[Normalization constants]")
print(f"  SPEC_MIN={SPEC_MIN}, SPEC_MAX={SPEC_MAX}, RANGE={SPEC_RANGE}")
print(f"  norm_spec({SPEC_MIN}) = {norm_spec(torch.tensor(SPEC_MIN)).item():.4f}")
print(f"  norm_spec({SPEC_MAX})   = {norm_spec(torch.tensor(SPEC_MAX)).item():.4f}")
print(f"  denorm_spec(-1.0) = {denorm_spec(torch.tensor(-1.0)).item():.4f}")
print(f"  denorm_spec(0.0)  = {denorm_spec(torch.tensor(0.0)).item():.4f}")
print(f"  denorm_spec(1.0)  = {denorm_spec(torch.tensor(1.0)).item():.4f}")
