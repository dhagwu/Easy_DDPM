"""
Inference: sample unwrapped phase from trained DDPM model.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import config as cfg
from dataset import InSARDataset
from model import UNet, DDPM


def load_model(checkpoint_path=None, device="cuda"):
    """Load trained model from checkpoint."""
    unet = UNet(
        in_channels=1,
        cond_channels=cfg.COND_CHANNELS,
        out_channels=cfg.OUT_CHANNELS,
        base_channels=cfg.MODEL_CHANNELS,
        channel_mults=cfg.CHANNEL_MULTS,
        num_res_blocks=cfg.NUM_RES_BLOCKS,
        time_emb_dim=cfg.TIME_EMB_DIM,
        dropout=0.0,  # no dropout for inference
        attn_resolutions=cfg.ATTN_RESOLUTIONS,
        image_size=cfg.IMAGE_SIZE,
    ).to(device)

    if checkpoint_path is None:
        for name in ["ddpm_best.pt", "best_model.pt", "final_model.pt"]:
            p = os.path.join(cfg.OUTPUT_DIR, name)
            if os.path.exists(p):
                checkpoint_path = p
                break

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # prefer EMA weights
    state = ckpt.get("ema_unet_state", ckpt.get("unet_state"))
    unet.load_state_dict(state)
    unet.eval()
    print(f"Loaded model from {checkpoint_path}")

    ddpm = DDPM(
        model=unet,
        num_timesteps=cfg.NUM_TIMESTEPS,
        beta_start=cfg.BETA_START,
        beta_end=cfg.BETA_END,
        device=device,
    ).to(device)
    return ddpm


@torch.no_grad()
def infer_and_visualize(ddpm, device="cuda", num_samples=8):
    """Run inference on test/val data and visualize results."""
    ds = InSARDataset(
        cfg.DATA_DIR, split="val",
        image_size=cfg.IMAGE_SIZE,
        num_samples=num_samples,
        use_sincos=cfg.USE_SINCOS,
    )

    fig, axes = plt.subplots(num_samples, 5, figsize=(16, 3.2 * num_samples))
    if num_samples == 1:
        axes = axes[None, :]

    titles = ["Wrapped", "Coherence", "Coarse Unwrapped", "DDPM Unwrapped", "True Unwrapped"]

    for idx in tqdm(range(num_samples), desc="Inference"):
        cond, target = ds[idx]
        cond = cond.unsqueeze(0).to(device)

        # DDPM sampling
        pred = ddpm.sample(cond, progress=False)  # [1, 1, H, W]
        pred = pred.squeeze().cpu().numpy()

        # Get wrapped, coherence for display
        cond_np = cond.squeeze(0).cpu().numpy()  # [3, H, W]

        if cfg.USE_SINCOS:
            wrapped_sin = cond_np[0]
            wrapped_cos = cond_np[1]
            wrapped = np.arctan2(wrapped_sin, wrapped_cos)  # reconstruct wrapped
            coherence = cond_np[2]
        else:
            wrapped = cond_np[0]
            coherence = cond_np[1]

        target_np = target.squeeze().numpy()

        # Load coarse unwrapped from the original npz
        sample_path = ds.files[idx]
        data = np.load(sample_path)
        coarse = data["coarse_unwrapped"].astype(np.float32)
        # same center crop
        h, w = coarse.shape
        top = (h - cfg.IMAGE_SIZE) // 2
        left = (w - cfg.IMAGE_SIZE) // 2
        coarse = coarse[top:top + cfg.IMAGE_SIZE, left:left + cfg.IMAGE_SIZE]

        axes[idx, 0].imshow(wrapped, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        axes[idx, 1].imshow(coherence, cmap="gray", vmin=0, vmax=1)
        axes[idx, 2].imshow(coarse, cmap="viridis")
        axes[idx, 3].imshow(pred, cmap="viridis")
        axes[idx, 4].imshow(target_np, cmap="viridis")

        for j, t in enumerate(titles):
            axes[idx, j].set_title(t, fontsize=10)
            axes[idx, j].axis("off")

    plt.tight_layout()
    save_path = os.path.join(cfg.OUTPUT_DIR, "inference_results.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {save_path}")

    # Compute metrics
    metrics = compute_metrics(ddpm, ds, device)
    return metrics


@torch.no_grad()
def compute_metrics(ddpm, ds, device="cuda"):
    """Compute MAE and RMSE between predicted and true unwrapped phase."""
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

    mae_total = 0.0
    rmse_total = 0.0
    count = 0

    for cond, target in tqdm(dl, desc="Computing metrics"):
        cond = cond.to(device)
        pred = ddpm.sample(cond, progress=False)
        pred = pred.cpu()
        diff = pred - target
        mae_total += diff.abs().sum().item()
        rmse_total += (diff ** 2).sum().item()
        count += target.numel()

    mae = mae_total / count
    rmse = np.sqrt(rmse_total / count)
    print(f"\nMetrics on {len(ds)} samples:")
    print(f"  MAE:  {mae:.4f} rad")
    print(f"  RMSE: {rmse:.4f} rad")

    # Also compare with coarse unwrapped MAE
    coarse_mae_total = 0.0
    for i in range(len(ds)):
        cond, target = ds[i]
        data = np.load(ds.files[i])
        coarse = data["coarse_unwrapped"].astype(np.float32)
        h, w = coarse.shape
        top = (h - cfg.IMAGE_SIZE) // 2
        left = (w - cfg.IMAGE_SIZE) // 2
        coarse = coarse[top:top + cfg.IMAGE_SIZE, left:left + cfg.IMAGE_SIZE]
        coarse_mae_total += np.abs(coarse - target.squeeze().numpy()).sum()
    coarse_mae = coarse_mae_total / count
    print(f"  Coarse MAE (baseline): {coarse_mae:.4f} rad")
    print(f"  Improvement: {(1 - mae/coarse_mae)*100:.1f}%")

    return {"mae": mae, "rmse": rmse, "coarse_mae": coarse_mae}


def main():
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ddpm = load_model(device=device)
    metrics = infer_and_visualize(ddpm, device=device, num_samples=cfg.INFER_NUM)

    # Save metrics
    metrics_path = os.path.join(cfg.OUTPUT_DIR, "metrics.txt")
    with open(metrics_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.6f}\n")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
