"""
DDPM for InSAR Phase Unwrapping — Configuration
"""
import os

# ── Paths ──────────────────────────────────────────────
DATA_DIR   = r"E:\projects\insar_ddpm_project_gpu_ready\DDPM2\data\taskbook_merged"
OUTPUT_DIR = r"E:\projects\insar_ddpm_project_gpu_ready\Easy_DDPM\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Data ───────────────────────────────────────────────
IMAGE_SIZE      = 256
NUM_TRAIN       = 300
NUM_VAL         = 80
NUM_TEST        = 80
USE_SINCOS      = True

# ── Diffusion ──────────────────────────────────────────
NUM_TIMESTEPS   = 200
BETA_START      = 1e-4
BETA_END        = 0.02

# ── Model (UNet) ───────────────────────────────────────
MODEL_CHANNELS      = 64
CHANNEL_MULTS       = [1, 2, 4]
NUM_RES_BLOCKS      = 2
TIME_EMB_DIM        = 128
ATTN_RESOLUTIONS    = [32]
DROPOUT             = 0.1
COND_CHANNELS       = 3       # sin(wrapped), cos(wrapped), coherence
OUT_CHANNELS        = 1       # unwrapped phase

# ── Training ───────────────────────────────────────────
BATCH_SIZE      = 4
LEARNING_RATE   = 1e-4
NUM_EPOCHS      = 80
EMA_DECAY       = 0.995
SAVE_EVERY      = 20
DEVICE          = "cuda"
NUM_WORKERS     = 2
USE_AMP         = True

# ── DDIM Acceleration ─────────────────────────────────
DDIM_STEPS      = 50
DDIM_ETA        = 0.0
USE_DDIM        = True

# ── Inference ──────────────────────────────────────────
INFER_NUM       = 8
