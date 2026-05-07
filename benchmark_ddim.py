"""Speed-quality benchmark: DDPM vs DDIM at various step counts."""
import os, sys, time
import torch, numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from dataset import InSARDataset
from model import UNet, DDPM

device = torch.device("cuda")
ds = InSARDataset(cfg.DATA_DIR, split="val", image_size=cfg.IMAGE_SIZE,
                   num_samples=16, use_sincos=cfg.USE_SINCOS)

unet = UNet(
    in_channels=1, cond_channels=cfg.COND_CHANNELS, out_channels=cfg.OUT_CHANNELS,
    base_channels=cfg.MODEL_CHANNELS, channel_mults=cfg.CHANNEL_MULTS,
    num_res_blocks=cfg.NUM_RES_BLOCKS, time_emb_dim=cfg.TIME_EMB_DIM,
    dropout=0.0, attn_resolutions=cfg.ATTN_RESOLUTIONS, image_size=cfg.IMAGE_SIZE,
).to(device)
ckpt = torch.load(os.path.join(cfg.OUTPUT_DIR, "ddpm_best.pt"), map_location=device, weights_only=True)
unet.load_state_dict(ckpt["ema_unet_state"])
unet.eval()
ddpm = DDPM(unet, num_timesteps=cfg.NUM_TIMESTEPS, beta_start=cfg.BETA_START,
            beta_end=cfg.BETA_END, device=device).to(device)

step_counts = [200, 100, 50, 20, 10, 5]
results = []
baseline_time = None

for steps in step_counts:
    times, maes = [], []
    for idx in range(len(ds)):
        cond, target = ds[idx]
        cond_gpu = cond.unsqueeze(0).to(device)
        target_np = target.squeeze().numpy()
        data = np.load(ds.files[idx])
        mask = data["mask"].astype(np.float32)
        h, w = mask.shape
        t, l = (h - cfg.IMAGE_SIZE) // 2, (w - cfg.IMAGE_SIZE) // 2
        mask = mask[t:t + cfg.IMAGE_SIZE, l:l + cfg.IMAGE_SIZE]

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if steps == 200:
            pred = ddpm.sample(cond_gpu, progress=False).squeeze().cpu().numpy()
        else:
            pred = ddpm.ddim_sample(cond_gpu, ddim_steps=steps, eta=0.0, progress=False).squeeze().cpu().numpy()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        v = mask > 0.5
        valid_count = v.sum()
        if valid_count == 0:
            maes.append(0.0)
        else:
            mae_val = np.abs(pred[v] - target_np[v]).mean()
            if np.isnan(mae_val) or np.isinf(mae_val):
                mae_val = np.nan_to_num(mae_val, nan=99.0, posinf=99.0, neginf=99.0)
            maes.append(mae_val)

    avg_time = np.mean(times) * 1000
    avg_mae = np.mean(maes)
    if steps == 200:
        baseline_time = avg_time
    speedup = baseline_time / avg_time if baseline_time else float(steps) / 200
    results.append((steps, avg_time, avg_mae, speedup))

print(f"\n{'='*65}")
print(f"DDIM SPEED-QUALITY BENCHMARK (16 samples, {cfg.IMAGE_SIZE}x{cfg.IMAGE_SIZE})")
print(f"{'='*65}")
print(f"{'Method':<10s} {'Steps':>6s} {'Time':>10s} {'Masked MAE':>12s} {'vs DDPM':>10s}")
print(f"{'-'*55}")
for steps, t, mae, sp in results:
    label = "DDPM" if steps == 200 else "DDIM"
    print(f"{label:<10s} {steps:>6d} {t:>7.1f} ms {mae:>10.3f} rad {sp:>8.1f}x")

best_mae = results[0][2]
for steps, t, mae, sp in results[1:]:
    quality = mae / best_mae * 100
    print(f"  DDIM-{steps:<3d}: quality={quality:.0f}%, speed={sp:.1f}x, time={t:.0f}ms")
