"""
Unified evaluation: 4 methods × 5 metrics, grouped analysis, rich visualization.
Metrics: masked_mae, masked_rmse, wrapped_mae, gradient_mae, gain_over_coarse
"""
import os, time
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from tqdm import tqdm

import config as cfg
from dataset import InSARDataset


# ── Metric Functions ────────────────────────────────────

def wrap_phase(x):
    """Wrap to [-pi, pi)."""
    return np.arctan2(np.sin(x), np.cos(x))

def masked_mae(pred, gt, mask):
    """MAE over valid pixels only."""
    v = mask > 0.5
    if v.sum() == 0:
        return np.nan
    return np.abs(pred[v] - gt[v]).mean()

def masked_rmse(pred, gt, mask):
    """RMSE over valid pixels only."""
    v = mask > 0.5
    if v.sum() == 0:
        return np.nan
    return np.sqrt(((pred[v] - gt[v]) ** 2).mean())

def wrapped_mae(pred, wrapped_obs, mask):
    """Wrapped-domain MAE: how well does pred re-wrap to the observation."""
    v = mask > 0.5
    if v.sum() == 0:
        return np.nan
    pred_wrapped = wrap_phase(pred)
    diff = wrap_phase(pred_wrapped - wrapped_obs)
    return np.abs(diff[v]).mean()

def gradient_mae(pred, gt):
    """Mean absolute difference of spatial gradients (x and y)."""
    gy_p, gx_p = pred[1:, :] - pred[:-1, :], pred[:, 1:] - pred[:, :-1]
    gy_g, gx_g = gt[1:, :] - gt[:-1, :], gt[:, 1:] - gt[:, :-1]
    return 0.5 * (np.abs(gy_p - gy_g).mean() + np.abs(gx_p - gx_g).mean())


# ── Cache ───────────────────────────────────────────────

def load_cache():
    p = os.path.join(cfg.OUTPUT_DIR, "eval_cache.npz")
    if os.path.exists(p):
        return dict(np.load(p, allow_pickle=True))
    return None

def save_cache(results):
    p = os.path.join(cfg.OUTPUT_DIR, "eval_cache.npz")
    np.savez(p, **results)


# ── Main ────────────────────────────────────────────────

def run_evaluation():
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = InSARDataset(cfg.DATA_DIR, split="val", image_size=cfg.IMAGE_SIZE,
                       num_samples=cfg.NUM_TEST, use_sincos=cfg.USE_SINCOS)
    n = len(ds)
    print(f"Evaluating on {n} validation samples")

    cache = load_cache()
    if cache is not None and len(cache.get("files", [])) == n:
        print("Using cached results...")
        results = cache
    else:
        # clear old cache (might be from different dataset)
        old_path = os.path.join(cfg.OUTPUT_DIR, "eval_cache.npz")
        if os.path.exists(old_path):
            os.remove(old_path)
        results = {"files": np.array(ds.files)}

    # ── Helper: load per-sample data ─────────────────────
    def sample_data(idx):
        cond, target = ds[idx]
        data = np.load(ds.files[idx])
        gt_unwrapped = target.squeeze().numpy()
        mask = _crop(data["mask"].astype(np.float32))
        wrapped_obs = _crop(data["wrapped"].astype(np.float32))
        coarse = _crop(data["coarse_unwrapped"].astype(np.float32))
        diff = float(data["difficulty_score"])
        return cond, gt_unwrapped, mask, wrapped_obs, coarse, diff

    # ── 1. Least-Squares ─────────────────────────────────
    if "ls_preds" not in results:
        print("\n[1/4] Least-Squares baseline...")
        ls_preds = np.zeros((n, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), dtype=np.float32)
        for idx in tqdm(range(n)):
            _, _, _, _, coarse, _ = sample_data(idx)
            ls_preds[idx] = coarse
        results["ls_preds"] = ls_preds
        save_cache(results)

    # ── 2. Quality-Guided ────────────────────────────────
    if "qg_preds" not in results:
        from baselines import quality_guided_unwrap
        print("\n[2/4] Quality-Guided Path-Following...")
        qg_preds = np.zeros((n, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), dtype=np.float32)
        for idx in tqdm(range(n)):
            _, _, mask, wrapped_obs, coarse, _, _ = sample_data(idx)
            try:
                qg_preds[idx] = quality_guided_unwrap(wrapped_obs, _crop(
                    np.load(ds.files[idx])["coherence"].astype(np.float32)), mask, reference_phase=coarse)
            except Exception:
                qg_preds[idx] = np.full_like(coarse, np.nan)
        results["qg_preds"] = qg_preds
        save_cache(results)

    # ── 3. UNet Regression ───────────────────────────────
    if "unet_preds" not in results:
        from unet_regression import load_unet
        print("\n[3/4] UNet Regression baseline...")
        model = load_unet(device=device)
        unet_preds = np.zeros((n, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), dtype=np.float32)
        for idx in tqdm(range(n)):
            cond, _, _, _, coarse, _ = sample_data(idx)
            with torch.no_grad():
                out = model(cond.unsqueeze(0).to(device)).squeeze().cpu().numpy()
            unet_preds[idx] = out
        results["unet_preds"] = unet_preds
        save_cache(results)

    # ── 4. DDPM / DDIM ────────────────────────────────────
    sample_mode = "DDIM" if cfg.USE_DDIM else "DDPM"
    cache_key = "ddpm_preds"
    if cache_key not in results:
        from model import UNet, DDPM
        print(f"\n[4/4] DDPM sampling ({sample_mode}, {cfg.DDIM_STEPS if cfg.USE_DDIM else cfg.NUM_TIMESTEPS} steps)...")
        ddpm = _load_ddpm(device)
        ddpm_preds = np.zeros((n, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), dtype=np.float32)
        for idx in tqdm(range(n)):
            cond, _, _, _, coarse, _ = sample_data(idx)
            cond_gpu = cond.unsqueeze(0).to(device)
            if cfg.USE_DDIM:
                out = ddpm.ddim_sample(cond_gpu, ddim_steps=cfg.DDIM_STEPS,
                                        eta=cfg.DDIM_ETA, progress=False).squeeze().cpu().numpy()
            else:
                out = ddpm.sample(cond_gpu, progress=False).squeeze().cpu().numpy()
            ddpm_preds[idx] = out
        results[cache_key] = ddpm_preds
        save_cache(results)

    # ── Metadata ─────────────────────────────────────────
    if "meta" not in results:
        print("\nCollecting metadata...")
        results["meta"] = np.array([ds.get_meta(i) for i in range(n)])
        save_cache(results)

    # ── Compute All Metrics Per-Sample ───────────────────
    print("\nComputing extended metrics...")
    method_keys = ["ls", "qg", "unet", "ddpm"]
    method_names = ["Least-Squares", "Quality-Guided", "UNet Reg.", "DDPM (Ours)"]
    metrics_list = ["masked_mae", "masked_rmse", "wrapped_mae", "gradient_mae", "gain_mae"]

    rows = []
    for idx in range(n):
        _, _, mask, wrapped_obs, coarse, diff, gt_unwrapped = sample_data(idx)
        meta = results["meta"][idx]
        row = {
            "sample": os.path.basename(results["files"][idx]),
            "source_domain": _m(meta, "source_domain"),
            "deformation_type": _m(meta, "deformation_type"),
            "noise_level": _m(meta, "noise_level"),
            "coherence_level": _m(meta, "coherence_level"),
            "gradient_level": _m(meta, "gradient_level"),
            "difficulty_score": diff,
            "coarse_mae": masked_mae(coarse, gt_unwrapped, mask),
        }
        for mk, mn in zip(method_keys, method_names):
            pred = results[f"{mk}_preds"][idx]
            mm = masked_mae(pred, gt_unwrapped, mask)
            mr = masked_rmse(pred, gt_unwrapped, mask)
            mw = wrapped_mae(pred, wrapped_obs, mask)
            mg = gradient_mae(pred, gt_unwrapped)
            gain = row["coarse_mae"] - mm  # positive = better than LS
            row[f"{mk}_masked_mae"] = mm
            row[f"{mk}_masked_rmse"] = mr
            row[f"{mk}_wrapped_mae"] = mw
            row[f"{mk}_gradient_mae"] = mg
            row[f"{mk}_gain_mae"] = gain
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg.OUTPUT_DIR, "per_sample.csv"), index=False)
    print(f"Saved per_sample.csv ({len(df)} rows)")

    # ── Overall Summary ──────────────────────────────────
    metric_labels = {
        "masked_mae": "Masked MAE (rad)",
        "masked_rmse": "Masked RMSE (rad)",
        "wrapped_mae": "Wrapped MAE (rad)",
        "gradient_mae": "Gradient MAE (rad/px)",
        "gain_mae": "Gain over LS (rad)",
    }
    summary_rows = []
    for mk, mn in zip(method_keys, method_names):
        r = {"Method": mn}
        for ml in metrics_list:
            col = f"{mk}_{ml}"
            v = df[col].dropna()
            r[f"{ml}_mean"] = v.mean()
            r[f"{ml}_std"] = v.std()
            r[f"{ml}_median"] = v.median()
        # Win rate over LS
        if mk != "ls":
            r["better_than_ls"] = (df[f"{mk}_masked_mae"] <= df["ls_masked_mae"]).mean() * 100
        else:
            r["better_than_ls"] = 100.0
        summary_rows.append(r)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(cfg.OUTPUT_DIR, "evaluation_summary.csv"), index=False)

    print("\n" + "=" * 72)
    print("MULTI-DIMENSIONAL COMPARISON")
    print("=" * 72)
    header = f"{'Method':<20s}"
    for ml in metrics_list:
        header += f" {metric_labels[ml]:>20s}"
    header += "  WinRate_vs_LS"
    print(header)
    print("-" * len(header))
    for _, r in summary.iterrows():
        line = f"{r['Method']:<20s}"
        for ml in metrics_list:
            line += f" {r[f'{ml}_mean']:>8.3f}±{r[f'{ml}_std']:<8.3f}"
        line += f"  {r['better_than_ls']:.0f}%"
        print(line)

    # ── Grouped Analysis ─────────────────────────────────
    group_cols = ["source_domain", "noise_level", "gradient_level", "deformation_type"]
    group_rows = []
    for gc in group_cols:
        for gv in df[gc].unique():
            sub = df[df[gc] == gv]
            if len(sub) < 3:
                continue
            for mk, mn in zip(method_keys, method_names):
                r2 = {"Group": gc, "Value": gv, "Method": mn, "N": len(sub)}
                for ml in metrics_list:
                    v = sub[f"{mk}_{ml}"].dropna()
                    r2[f"{ml}_mean"] = v.mean()
                    r2[f"{ml}_std"] = v.std()
                group_rows.append(r2)

    group_df = pd.DataFrame(group_rows)
    group_df.to_csv(os.path.join(cfg.OUTPUT_DIR, "evaluation_extended.csv"), index=False)

    # print grouped by masked_mae
    for gc in group_cols:
        print(f"\n{'─' * 60}")
        print(f"By {gc} (Masked MAE):")
        print(f"{'─' * 60}")
        sub = group_df[group_df["Group"] == gc]
        for gv in sorted(df[gc].dropna().unique()):
            vr = sub[sub["Value"] == gv]
            if len(vr) == 0:
                continue
            print(f"\n  {gv}:")
            for _, r in vr.iterrows():
                print(f"    {r['Method']:<20s}  masked_mae={r['masked_mae_mean']:.3f}±{r['masked_mae_std']:.3f}  gain={r['gain_mae_mean']:+.3f}")

    # ── Difficulty Analysis ──────────────────────────────
    print(f"\n{'─' * 60}")
    print("By Difficulty (masked_mae, tertiles of difficulty_score):")
    print(f"{'─' * 60}")
    dq = np.percentile(df["difficulty_score"].dropna(), [33.3, 66.7])
    for label, lo, hi in [("Easy", -999, dq[0]), ("Medium", dq[0], dq[1]), ("Hard", dq[1], 999)]:
        sub = df[(df["difficulty_score"] >= lo) & (df["difficulty_score"] < hi)]
        if len(sub) == 0:
            continue
        print(f"\n  {label} (n={len(sub)}, diff ∈ [{lo:.2f}, {hi:.2f}]):")
        for mk, mn in zip(method_keys, method_names):
            v = sub[f"{mk}_masked_mae"].dropna()
            print(f"    {mn:<20s}  masked_mae={v.mean():.3f}±{v.std():.3f}")

    # ── Visualization ────────────────────────────────────
    fig = plt.figure(figsize=(20, 14))
    colors = ["#7f7f7f", "#ff7f0e", "#2ca02c", "#1f77b4"]

    # 1. Radar chart (top left)
    ax_radar = fig.add_subplot(2, 3, 1, projection="polar")
    _plot_radar(ax_radar, summary, method_names, metrics_list, colors)

    # 2. Grouped bar: masked_mae by source_domain (top middle)
    ax = fig.add_subplot(2, 3, 2)
    _plot_grouped_bar(ax, group_df, "source_domain", method_names, colors, "masked_mae")

    # 3. Grouped bar: gain_mae by noise_level (top right)
    ax = fig.add_subplot(2, 3, 3)
    _plot_grouped_bar(ax, group_df, "noise_level", method_names, colors, "gain_mae")

    # 4. Difficulty scatter (bottom left)
    ax = fig.add_subplot(2, 3, 4)
    _plot_difficulty_scatter(ax, df, method_keys, method_names, colors)

    # 5. Gain boxplot (bottom middle)
    ax = fig.add_subplot(2, 3, 5)
    _plot_gain_boxplot(ax, df, method_keys, method_names, colors)

    # 6. Wrapped MAE by deformation_type (bottom right)
    ax = fig.add_subplot(2, 3, 6)
    _plot_grouped_bar(ax, group_df, "deformation_type", method_names, colors, "wrapped_mae")

    plt.tight_layout()
    fig_path = os.path.join(cfg.OUTPUT_DIR, "evaluation_summary.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved {fig_path}")

    # ── Difficulty scatter (separate detailed) ───────────
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
    for i, (mk, mn, c) in enumerate(zip(method_keys, method_names, colors)):
        ax = axes2[i // 2, i % 2]
        x = df["difficulty_score"]
        y = df[f"{mk}_masked_mae"]
        ax.scatter(x, y, c=c, alpha=0.6, s=20, edgecolors="none")
        # fit line
        valid = ~(x.isna() | y.isna())
        if valid.sum() > 2:
            z = np.polyfit(x[valid], y[valid], 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, np.polyval(z, xs), "--", color="black", alpha=0.5, lw=1.5)
        ax.set_title(mn, fontsize=12, color=c)
        ax.set_xlabel("Difficulty Score")
        ax.set_ylabel("Masked MAE (rad)")
    plt.tight_layout()
    fig2_path = os.path.join(cfg.OUTPUT_DIR, "evaluation_difficulty.png")
    plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
    print(f"Saved {fig2_path}")

    return df, summary, group_df


# ── Plot Helpers ─────────────────────────────────────────

def _plot_radar(ax, summary, method_names, metrics_list, colors):
    """Radar chart of 5 normalized metrics (lower=better, so invert for plot)."""
    mlabels = ["Masked MAE", "Masked RMSE", "Wrapped MAE", "Gradient MAE", "1/Gain"]
    n_metrics = len(metrics_list)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    # Normalize each metric to [0, 1] across methods (lower=better → 1, higher=worse → 0)
    for i, (ml, mlab) in enumerate(zip(metrics_list, mlabels)):
        vals = summary[f"{ml}_mean"].values
        if ml == "gain_mae":
            vmin, vmax = vals.min(), vals.max()
        else:
            vmin, vmax = vals.min(), vals.max()
        if vmax - vmin < 1e-8:
            normed = np.ones_like(vals)
        else:
            # invert: lower value → higher score
            if ml == "gain_mae":
                normed = (vals - vmin) / (vmax - vmin)  # higher gain = better
            else:
                normed = 1.0 - (vals - vmin) / (vmax - vmin)
        for j, (mn, c) in enumerate(zip(method_names, colors)):
            values = [summary[f"{m}_mean"].values[j] for m in metrics_list]
            vals_norm = []
            for k, (ml2, vl) in enumerate(zip(metrics_list, values)):
                all_v = summary[f"{ml2}_mean"].values
                if ml2 == "gain_mae":
                    vals_norm.append((vl - all_v.min()) / (all_v.max() - all_v.min() + 1e-8))
                else:
                    vals_norm.append(1.0 - (vl - all_v.min()) / (all_v.max() - all_v.min() + 1e-8))
            vals_norm += vals_norm[:1]
            ax.fill(angles, vals_norm, alpha=0.08, color=c)
            ax.plot(angles, vals_norm, "o-", color=c, lw=2, markersize=4, label=mn if i == 0 else "")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(mlabels, fontsize=9)
    ax.set_title("Normalized Performance Radar", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)


def _plot_grouped_bar(ax, group_df, group_col, method_names, colors, metric="masked_mae"):
    sub = group_df[group_df["Group"] == group_col]
    values = sorted(sub["Value"].unique())
    n_methods = len(method_names)
    n_vals = len(values)
    x = np.arange(n_vals)
    width = 0.8 / n_methods

    for i, (m, c) in enumerate(zip(method_names, colors)):
        means = []
        for v in values:
            row = sub[(sub["Value"] == v) & (sub["Method"] == m)]
            means.append(row[f"{metric}_mean"].values[0] if len(row) > 0 else 0)
        ax.bar(x + i * width, means, width, label=m, color=c, edgecolor="white")

    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels(values, fontsize=9)
    ylabel = {"masked_mae": "Masked MAE (rad)", "gain_mae": "Gain over LS (rad)",
              "wrapped_mae": "Wrapped MAE (rad)", "gradient_mae": "Gradient MAE (rad/px)"}
    ax.set_ylabel(ylabel.get(metric, metric))
    ax.set_title(f"{ylabel.get(metric, metric)} by {group_col}")
    ax.legend(fontsize=8)


def _plot_difficulty_scatter(ax, df, method_keys, method_names, colors):
    for mk, mn, c in zip(method_keys, method_names, colors):
        x = df["difficulty_score"]
        y = df[f"{mk}_masked_mae"]
        ax.scatter(x, y, c=c, alpha=0.5, s=16, label=mn, edgecolors="none")
    ax.set_xlabel("Difficulty Score")
    ax.set_ylabel("Masked MAE (rad)")
    ax.set_title("MAE vs Difficulty Score")
    ax.legend(fontsize=8)


def _plot_gain_boxplot(ax, df, method_keys, method_names, colors):
    gains = []
    labels = []
    col_list = []
    for mk, mn, c in zip(method_keys, method_names, colors):
        if mk == "ls":
            continue  # skip baseline
        g = df[f"{mk}_gain_mae"].dropna()
        gains.append(g)
        labels.append(mn)
        col_list.append(c)
    bp = ax.boxplot(gains, labels=labels, patch_artist=True)
    for patch, c in zip(bp["boxes"], col_list):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.4)
    ax.set_ylabel("Gain over LS (rad)  [positive = better]")
    ax.set_title("Improvement over Least-Squares")


# ── Utilities ───────────────────────────────────────────

def _crop(arr):
    h, w = arr.shape
    t = (h - cfg.IMAGE_SIZE) // 2
    l = (w - cfg.IMAGE_SIZE) // 2
    return arr[t:t + cfg.IMAGE_SIZE, l:l + cfg.IMAGE_SIZE]

def _m(meta, key):
    if isinstance(meta, dict):
        return meta.get(key, "unknown")
    return "unknown"

def _load_ddpm(device):
    from model import UNet, DDPM
    unet = UNet(
        in_channels=1, cond_channels=cfg.COND_CHANNELS, out_channels=cfg.OUT_CHANNELS,
        base_channels=cfg.MODEL_CHANNELS, channel_mults=cfg.CHANNEL_MULTS,
        num_res_blocks=cfg.NUM_RES_BLOCKS, time_emb_dim=cfg.TIME_EMB_DIM,
        dropout=0.0, attn_resolutions=cfg.ATTN_RESOLUTIONS, image_size=cfg.IMAGE_SIZE,
    ).to(device)
    ckpt_name = f"ddpm_{cfg.IMAGE_SIZE}_best.pt"
    ckpt_path = os.path.join(cfg.OUTPUT_DIR, ckpt_name)
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(cfg.OUTPUT_DIR, "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt.get("ema_unet_state", ckpt.get("unet_state"))
    unet.load_state_dict(state)
    unet.eval()
    return DDPM(unet, num_timesteps=cfg.NUM_TIMESTEPS, beta_start=cfg.BETA_START,
                beta_end=cfg.BETA_END, device=device).to(device)


if __name__ == "__main__":
    run_evaluation()
