"""
Unified training script: DDPM or UNet regression baseline.
Usage: python train.py --method ddpm    (or --method unet)
"""
import os
import sys
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import config as cfg
from dataset import make_dataloaders


def ema_update(model, ema_model, decay):
    with torch.no_grad():
        for p_ema, p in zip(ema_model.parameters(), model.parameters()):
            p_ema.copy_(decay * p_ema + (1.0 - decay) * p)


def train_ddpm():
    from model import UNet, DDPM
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Training DDPM on {device}")

    dl_train, dl_val = make_dataloaders(cfg)
    print(f"Train: {len(dl_train.dataset)}, Val: {len(dl_val.dataset)}")

    unet = UNet(
        in_channels=1, cond_channels=cfg.COND_CHANNELS, out_channels=cfg.OUT_CHANNELS,
        base_channels=cfg.MODEL_CHANNELS, channel_mults=cfg.CHANNEL_MULTS,
        num_res_blocks=cfg.NUM_RES_BLOCKS, time_emb_dim=cfg.TIME_EMB_DIM,
        dropout=cfg.DROPOUT, attn_resolutions=cfg.ATTN_RESOLUTIONS,
        image_size=cfg.IMAGE_SIZE,
    ).to(device)

    ddpm = DDPM(unet, num_timesteps=cfg.NUM_TIMESTEPS, beta_start=cfg.BETA_START,
                beta_end=cfg.BETA_END, device=device,
                cosine_schedule=cfg.COSINE_SCHEDULE, masked_loss=cfg.MASKED_LOSS).to(device)

    ema_unet = UNet(
        in_channels=1, cond_channels=cfg.COND_CHANNELS, out_channels=cfg.OUT_CHANNELS,
        base_channels=cfg.MODEL_CHANNELS, channel_mults=cfg.CHANNEL_MULTS,
        num_res_blocks=cfg.NUM_RES_BLOCKS, time_emb_dim=cfg.TIME_EMB_DIM,
        dropout=cfg.DROPOUT, attn_resolutions=cfg.ATTN_RESOLUTIONS,
        image_size=cfg.IMAGE_SIZE,
    ).to(device)
    ema_unet.load_state_dict(unet.state_dict())

    ema_ddpm = DDPM(ema_unet, num_timesteps=cfg.NUM_TIMESTEPS, beta_start=cfg.BETA_START,
                    beta_end=cfg.BETA_END, device=device,
                    cosine_schedule=cfg.COSINE_SCHEDULE, masked_loss=cfg.MASKED_LOSS).to(device)

    print(f"Model params: {sum(p.numel() for p in unet.parameters()):,}")
    opt = AdamW(unet.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-5)
    sched = CosineAnnealingLR(opt, T_max=cfg.NUM_EPOCHS_DDPM)
    scaler = torch.cuda.amp.GradScaler() if cfg.USE_AMP else None

    best_val_loss = float("inf")
    log_file = os.path.join(cfg.OUTPUT_DIR, "ddpm_train_log.txt")
    start_time = time.time()

    for epoch in range(1, cfg.NUM_EPOCHS_DDPM + 1):
        unet.train()
        train_loss = 0.0
        pbar = tqdm(dl_train, desc=f"DDPM Epoch {epoch}/{cfg.NUM_EPOCHS_DDPM}")
        for batch in pbar:
            cond, target, extra = [b.to(device) for b in batch]
            if scaler:
                with torch.cuda.amp.autocast():
                    loss = ddpm(target, cond)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss = ddpm(target, cond)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                opt.step()
            opt.zero_grad(set_to_none=True)
            ema_update(unet, ema_unet, cfg.EMA_DECAY)
            train_loss += loss.item()
        train_loss /= len(dl_train)

        unet.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in dl_val:
                cond, target, extra = [b.to(device) for b in batch]
                val_loss += ema_ddpm(target, cond).item()
        val_loss /= len(dl_val)
        sched.step()

        msg = f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Time: {(time.time()-start_time)/60:.1f}m"
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_name = f"ddpm_{cfg.IMAGE_SIZE}_best.pt"
            torch.save({"unet_state": unet.state_dict(), "ema_unet_state": ema_unet.state_dict()},
                       os.path.join(cfg.OUTPUT_DIR, ckpt_name))

        if epoch % cfg.SAVE_EVERY == 0:
            torch.save({"unet_state": unet.state_dict(), "ema_unet_state": ema_unet.state_dict()},
                       os.path.join(cfg.OUTPUT_DIR, f"ddpm_epoch{epoch}.pt"))

    print(f"DDPM done! {cfg.NUM_EPOCHS_DDPM} epochs, best val: {best_val_loss:.4f}")


def train_unet():
    import unet_regression
    unet_regression.train_unet()


def main():
    method = "ddpm"
    if "--method" in sys.argv:
        idx = sys.argv.index("--method")
        method = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "ddpm"

    print(f"Method: {method}")
    if method == "ddpm":
        train_ddpm()
    elif method == "unet":
        train_unet()
    else:
        print(f"Unknown method: {method}. Use --method ddpm or --method unet")


if __name__ == "__main__":
    main()
