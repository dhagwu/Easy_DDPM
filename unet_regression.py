"""
UNet Regression baseline — direct phase unwrapping without diffusion.
"""
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import config as cfg
from dataset import make_dataloaders, InSARDataset


# ── Lightweight UNet (no time embedding) ───────────────

def conv3x3(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, 3, padding=1)


def conv1x1(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, 1)


class ResBlock2D(nn.Module):
    """Residual block without time conditioning."""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = conv3x3(in_ch, out_ch)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = conv3x3(out_ch, out_ch)
        self.skip = conv1x1(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = conv3x3(channels, channels)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class UNetRegressor(nn.Module):
    """Simple UNet for direct phase regression (no noise, no time emb)."""

    def __init__(self, in_channels=3, out_channels=1, base_channels=64,
                 channel_mults=(1, 2, 4), num_res_blocks=2, dropout=0.1):
        super().__init__()
        self.num_levels = len(channel_mults)

        self.conv_in = conv3x3(in_channels, base_channels)

        # Encoder
        ch = base_channels
        self.down_resblocks = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(ResBlock2D(ch, out_ch, dropout))
                ch = out_ch
            self.down_resblocks.append(level_blocks)
            self.down_attns.append(nn.ModuleList())  # no attention for simplicity
            if level != len(channel_mults) - 1:
                self.downsamples.append(Downsample2D(ch))

        # Middle
        self.mid_block1 = ResBlock2D(ch, ch, dropout)
        self.mid_block2 = ResBlock2D(ch, ch, dropout)

        # Decoder
        self.up_resblocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for level, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            if level != 0:
                self.upsamples.append(Upsample2D(ch))
            level_blocks = nn.ModuleList()
            skip_ch = base_channels * channel_mults[self.num_levels - 1 - level]
            level_blocks.append(ResBlock2D(ch + skip_ch, out_ch, dropout))
            ch = out_ch
            for _ in range(num_res_blocks):
                level_blocks.append(ResBlock2D(ch, out_ch, dropout))
                ch = out_ch
            self.up_resblocks.append(level_blocks)

        self.norm_out = nn.GroupNorm(min(32, ch), ch)
        self.conv_out = conv3x3(ch, out_channels)

    def forward(self, x):
        h = self.conv_in(x)

        # Encoder
        skips = []
        for level in range(self.num_levels):
            for block in self.down_resblocks[level]:
                h = block(h)
            skips.append(h)
            if level < len(self.downsamples):
                h = self.downsamples[level](h)

        # Middle
        h = self.mid_block1(h)
        h = self.mid_block2(h)

        # Decoder
        for level in range(self.num_levels):
            if level > 0:
                h = self.upsamples[level - 1](h)
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            for block in self.up_resblocks[level]:
                h = block(h)

        h = F.silu(self.norm_out(h))
        return self.conv_out(h)


# ── Training ───────────────────────────────────────────

def train_unet():
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Training UNet Regressor on {device}")

    dl_train, dl_val = make_dataloaders(cfg)
    print(f"Train: {len(dl_train.dataset)}, Val: {len(dl_val.dataset)}")

    model = UNetRegressor(
        in_channels=cfg.COND_CHANNELS,
        out_channels=cfg.OUT_CHANNELS,
        base_channels=cfg.MODEL_CHANNELS,
        channel_mults=cfg.CHANNEL_MULTS,
        num_res_blocks=cfg.NUM_RES_BLOCKS,
        dropout=cfg.DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    opt = AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-5)
    sched = CosineAnnealingLR(opt, T_max=cfg.NUM_EPOCHS_UNET)
    l1_loss = nn.L1Loss()

    best_val_loss = float("inf")
    log_file = os.path.join(cfg.OUTPUT_DIR, "unet_train_log.txt")
    start_time = time.time()

    for epoch in range(1, cfg.NUM_EPOCHS_UNET + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(dl_train, desc=f"UNet Epoch {epoch}/{cfg.NUM_EPOCHS_UNET}")
        for batch in pbar:
            cond, target, extra = [b.to(device) for b in batch]
            pred = model(cond)
            loss = F.mse_loss(pred, target) + 0.5 * l1_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= len(dl_train)

        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in dl_val:
                cond, target, extra = [b.to(device) for b in batch]
                pred = model(cond)
                val_loss += F.l1_loss(pred, target).item()
        val_loss /= len(dl_val)
        sched.step()

        elapsed = time.time() - start_time
        msg = f"Epoch {epoch:3d} | Train MSE+L1: {train_loss:.4f} | Val MAE: {val_loss:.4f} | Time: {elapsed/60:.1f}m"
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_name = "unet_residual_best.pt" if cfg.PREDICT_RESIDUAL else "unet_best.pt"
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, ckpt_name))

        if epoch % cfg.SAVE_EVERY == 0:
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, f"unet_epoch{epoch}.pt"))

    total_time = time.time() - start_time
    print(f"\nUNet training done! Total: {total_time/60:.1f} min, Best val MAE: {best_val_loss:.4f}")

    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "unet_final.pt"))
    return model


# ── Inference ──────────────────────────────────────────

@torch.no_grad()
def infer_unet(model, condition):
    """Single forward pass — orders of magnitude faster than DDPM."""
    model.eval()
    if isinstance(condition, torch.Tensor):
        condition = condition.unsqueeze(0) if condition.dim() == 3 else condition
    pred = model(condition)
    return pred.squeeze().cpu().numpy()


def load_unet(checkpoint_path=None, device="cuda"):
    """Load trained UNet regressor."""
    if checkpoint_path is None:
        ckpt_name = "unet_residual_best.pt" if cfg.PREDICT_RESIDUAL else "unet_best.pt"
        checkpoint_path = os.path.join(cfg.OUTPUT_DIR, ckpt_name)
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(cfg.OUTPUT_DIR, "unet_final.pt")

    # UNet baseline always uses 3-channel (trained before enhancement)
    model = UNetRegressor(
        in_channels=3,
        out_channels=cfg.OUT_CHANNELS,
        base_channels=cfg.MODEL_CHANNELS,
        channel_mults=cfg.CHANNEL_MULTS,
        num_res_blocks=cfg.NUM_RES_BLOCKS,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    return model


if __name__ == "__main__":
    train_unet()
