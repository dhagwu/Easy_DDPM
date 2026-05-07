"""
DDPM for InSAR Phase Unwrapping — Configuration (taskbook_merged dataset)
"""
import os

# ── Paths ──────────────────────────────────────────────
DATA_DIR   = r"E:\projects\insar_ddpm_project_gpu_ready\DDPM2\data\taskbook_merged"
OUTPUT_DIR = r"E:\projects\insar_ddpm_project_gpu_ready\Easy_DDPM\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Data ───────────────────────────────────────────────
IMAGE_SIZE      = 256          # crop from 512→256 for quality
NUM_TRAIN       = 300          # real + synthetic mixed
NUM_VAL         = 80
NUM_TEST        = 80
USE_SINCOS      = True         # sin/cos encoding of wrapped phase

# ── Diffusion ──────────────────────────────────────────
NUM_TIMESTEPS   = 200          # T (fewer steps = faster training)
BETA_START      = 1e-4
BETA_END        = 0.02
SCHEDULE        = "linear"

# ── Model (UNet) ───────────────────────────────────────
MODEL_CHANNELS      = 64
CHANNEL_MULTS       = [1, 2, 4]
NUM_RES_BLOCKS      = 2
TIME_EMB_DIM        = 128
ATTN_RESOLUTIONS    = [32]
DROPOUT             = 0.1
COND_CHANNELS       = 3       # sin(wrapped), cos(wrapped), coherence
COND_ENHANCED       = False   # use enhanced conditioning (coarse + gradients + mask)
COSINE_SCHEDULE     = False   # cosine noise schedule
MASKED_LOSS         = False   # only compute loss on valid pixels (mask > 0.5)
OUT_CHANNELS        = 1       # unwrapped phase

# ── Training (shared) ──────────────────────────────────
BATCH_SIZE      = 4           # reduced for 256×256 memory
USE_AMP         = True        # automatic mixed precision (fp16)
LEARNING_RATE   = 1e-4
EMA_DECAY       = 0.995
SAVE_EVERY      = 20
LOG_EVERY       = 5
DEVICE          = "cuda"
NUM_WORKERS     = 2

# ── DDPM Training ──────────────────────────────────────
NUM_EPOCHS_DDPM = 80

# ── UNet Regression Training ───────────────────────────
NUM_EPOCHS_UNET = 80

# ── Inference ──────────────────────────────────────────
SAMPLE_STEPS    = 200         # DDPM baseline steps
INFER_NUM       = 8

# ── DDIM Acceleration ─────────────────────────────────
DDIM_STEPS      = 50          # DDIM sampling steps (200→50, 4× speedup)
DDIM_ETA        = 0.0         # 0=deterministic, 1=DDPM-like stochastic
USE_DDIM        = True        # use DDIM for inference/evaluation

# ── Residual Prediction ───────────────────────────────
PREDICT_RESIDUAL = False      # True=predict residual (unwrapped-coarse), False=predict full phase

# ── Baselines ──────────────────────────────────────────
# methods to evaluate: "least_squares", "quality_guided", "unet", "ddpm"
BASELINE_METHODS = ["least_squares", "quality_guided", "unet", "ddpm"]
