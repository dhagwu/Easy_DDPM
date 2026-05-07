"""
Classic InSAR phase unwrapping baselines.
Ported from DDPM2/insar_diffusion/classic_baselines.py
"""
import heapq
import numpy as np
from scipy.ndimage import distance_transform_edt


# ── Utilities ──────────────────────────────────────────

def wrap_phase_np(x):
    """Wrap phase to [-pi, pi)."""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def wrapped_gradients(wrapped):
    """Compute wrapped x/y gradients."""
    h, w = wrapped.shape
    gx = np.zeros_like(wrapped)
    gy = np.zeros_like(wrapped)
    gx[:, :w - 1] = wrap_phase_np(wrapped[:, 1:] - wrapped[:, :w - 1])
    gy[:h - 1, :] = wrap_phase_np(wrapped[1:, :] - wrapped[:h - 1, :])
    return gx, gy


def nearest_fill(arr, valid_mask):
    """Fill invalid pixels with nearest valid value (distance transform)."""
    valid = valid_mask > 0.5
    if valid.all():
        return arr.copy()
    indices = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    if isinstance(indices, np.ndarray):
        indices = (indices[0], indices[1])
    return arr[indices[0], indices[1]]


def poisson_solve_periodic(div):
    """Solve Poisson equation with FFT under periodic BCs."""
    h, w = div.shape
    ky = 2.0 * np.pi * np.fft.fftfreq(h).reshape(-1, 1).astype(np.float64)
    kx = 2.0 * np.pi * np.fft.fftfreq(w).reshape(1, -1).astype(np.float64)
    denom = (2.0 * np.cos(ky) - 2.0) + (2.0 * np.cos(kx) - 2.0)
    div_hat = np.fft.fft2(div.astype(np.float64))
    denom[0, 0] = 1.0
    phi_hat = div_hat / denom
    phi_hat[0, 0] = 0.0
    return np.fft.ifft2(phi_hat).real.astype(np.float32)


# ── Least-Squares (FFT Poisson) ────────────────────────

def least_squares_unwrap(wrapped, valid_mask):
    """Least-squares phase unwrapping via FFT Poisson solver."""
    wrapped_fill = nearest_fill(wrapped, valid_mask)
    gx, gy = wrapped_gradients(wrapped_fill)
    div = np.zeros_like(wrapped_fill)
    div[:, :-1] += gx[:, :-1]
    div[:, 1:] -= gx[:, :-1]
    div[:-1, :] += gy[:-1, :]
    div[1:, :] -= gy[:-1, :]
    unwrapped = poisson_solve_periodic(div)
    valid = valid_mask > 0.5
    if valid.sum() > 0:
        unwrapped -= np.median(unwrapped[valid])
    return unwrapped.astype(np.float32)


# ── Quality-Guided Path-Following ──────────────────────

def _iter_neighbors(r, c, h, w):
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _select_seed(quality, valid, visited):
    remaining = np.where(valid & ~visited)
    if len(remaining[0]) == 0:
        return None
    idx = np.argmax(quality[remaining])
    return remaining[0][idx], remaining[1][idx]


def align_phase_to_reference(pred, reference, valid_mask):
    """Align predicted phase to reference by median offset."""
    valid = valid_mask > 0.5
    if valid.sum() == 0:
        return pred
    diff = reference[valid] - pred[valid]
    offset = np.median(diff)
    offset = np.round(offset / (2.0 * np.pi)) * (2.0 * np.pi)
    return pred + offset


def quality_guided_unwrap(wrapped, coherence, valid_mask, reference_phase=None):
    """
    Quality-guided path-following unwrapping.

    Uses coherence as quality map. Expands from highest-coherence seeds
    via priority queue, integrating wrapped phase differences along
    highest-quality edges first.

    Args:
        wrapped:         2D wrapped phase [rad], shape (H, W)
        coherence:       2D coherence map [0, 1], shape (H, W)
        valid_mask:      2D binary mask (>0.5 = valid)
        reference_phase: optional 2D reference for gauge alignment

    Returns:
        Unwrapped phase [rad], shape (H, W), float32
    """
    wrapped = wrap_phase_np(wrapped.astype(np.float64))
    coherence = coherence.astype(np.float64)
    valid_mask = valid_mask.astype(np.float64)

    h, w = wrapped.shape
    valid = (valid_mask > 0.5) & np.isfinite(wrapped) & np.isfinite(coherence)
    quality = coherence.copy()
    quality[~valid] = -1.0

    result = np.full((h, w), np.nan, dtype=np.float64)
    visited = np.zeros((h, w), dtype=bool)

    while True:
        seed = _select_seed(quality, valid, visited)
        if seed is None:
            break

        sr, sc = seed
        result[sr, sc] = wrapped[sr, sc]
        visited[sr, sc] = True

        heap = []
        for nr, nc in _iter_neighbors(sr, sc, h, w):
            if valid[nr, nc]:
                edge_quality = 0.5 * (coherence[sr, sc] + coherence[nr, nc])
                heapq.heappush(heap, (-edge_quality, sr, sc, nr, nc))

        while heap:
            _, pr, pc, r, c = heapq.heappop(heap)
            if visited[r, c]:
                continue

            delta = wrapped[r, c] - wrapped[pr, pc]
            delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
            result[r, c] = result[pr, pc] + delta
            visited[r, c] = True

            for nr, nc in _iter_neighbors(r, c, h, w):
                if valid[nr, nc] and not visited[nr, nc]:
                    edge_q = 0.5 * (coherence[r, c] + coherence[nr, nc])
                    heapq.heappush(heap, (-edge_q, r, c, nr, nc))

    # align each component
    if reference_phase is not None:
        result = align_phase_to_reference(result, reference_phase, valid_mask)
    else:
        v = valid & ~np.isnan(result)
        if v.sum() > 0:
            result -= np.median(result[v])

    result[~valid] = 0.0
    return result.astype(np.float32)


# ── Unified Interface ──────────────────────────────────

def run_baseline(method_name, wrapped, coherence, mask, coarse_unwrapped=None):
    """
    Run a classic baseline method on a single patch.

    Args:
        method_name:      "least_squares" or "quality_guided"
        wrapped:          2D wrapped phase [rad]
        coherence:        2D coherence [0, 1]
        mask:             2D binary mask
        coarse_unwrapped: 2D coarse unwrapped (used as ls result or qg reference)

    Returns:
        2D unwrapped phase [rad], float32
    """
    if method_name == "least_squares":
        if coarse_unwrapped is not None:
            return coarse_unwrapped.astype(np.float32)
        else:
            return least_squares_unwrap(wrapped, mask)

    elif method_name == "quality_guided":
        ref = coarse_unwrapped if coarse_unwrapped is not None else None
        return quality_guided_unwrap(wrapped, coherence, mask, reference_phase=ref)

    else:
        raise ValueError(f"Unknown baseline method: {method_name}")
