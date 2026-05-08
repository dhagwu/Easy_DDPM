"""
Conditional DDPM with a simple UNet backbone for InSAR phase unwrapping.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helpers ─────────────────────────────────────────────

def get_timestep_embedding(timesteps, dim, max_period=10000):
    """Sinusoidal timestep embeddings (transformer-style)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half)
    freqs = freqs.to(timesteps.device)
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def conv3x3(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, 3, padding=1)


def conv1x1(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, 1)


# ── Residual Block ──────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = conv3x3(in_ch, out_ch)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = conv3x3(out_ch, out_ch)
        self.skip = conv1x1(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


# ── Attention ───────────────────────────────────────────

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = conv1x1(channels, channels * 3)
        self.proj = conv1x1(channels, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        q = q.view(B, C, -1).permute(0, 2, 1)
        k = k.view(B, C, -1)
        v = v.view(B, C, -1).permute(0, 2, 1)
        scale = C ** -0.5
        attn = torch.bmm(q, k) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v).permute(0, 2, 1).view(B, C, H, W)
        return x + self.proj(out)


# ── Downsample / Upsample ───────────────────────────────

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = conv3x3(channels, channels)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# ── UNet ────────────────────────────────────────────────

class UNet(nn.Module):
    """Conditional UNet that predicts noise given x_t, t, and condition."""

    def __init__(self,
                 in_channels=1,
                 cond_channels=3,
                 out_channels=1,
                 base_channels=64,
                 channel_mults=(1, 2, 4),
                 num_res_blocks=2,
                 time_emb_dim=128,
                 dropout=0.1,
                 attn_resolutions=(16,),
                 image_size=128,
                 ):
        super().__init__()
        self.num_levels = len(channel_mults)
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.time_emb_dim = time_emb_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self.conv_in = conv3x3(in_channels + cond_channels, base_channels)

        # ── Encoder ──
        ch = base_channels
        curr_res = image_size
        self.down_resblocks = nn.ModuleList()   # per-level ResBlocks
        self.down_attns = nn.ModuleList()       # per-level attention
        self.downsamples = nn.ModuleList()      # per-level downsampler

        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            level_blocks = nn.ModuleList()
            level_attns = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(ResBlock(ch, out_ch, time_emb_dim, dropout))
                ch = out_ch
                if curr_res in attn_resolutions:
                    level_attns.append(AttentionBlock(ch))
            self.down_resblocks.append(level_blocks)
            self.down_attns.append(level_attns)
            if level != len(channel_mults) - 1:
                self.downsamples.append(Downsample(ch))
                curr_res //= 2

        # ── Middle ──
        self.mid_block1 = ResBlock(ch, ch, time_emb_dim, dropout)
        self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = ResBlock(ch, ch, time_emb_dim, dropout)

        # ── Decoder ──
        self.up_resblocks = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.decoder_skip_channels = []  # save for forward pass

        for level, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            if level != 0:
                self.upsamples.append(Upsample(ch))
                curr_res *= 2
            level_blocks = nn.ModuleList()
            level_attns = nn.ModuleList()
            skip_ch = base_channels * channel_mults[self.num_levels - 1 - level]
            self.decoder_skip_channels.append(skip_ch)
            # First ResBlock: concatenates skip, input has extra skip_ch channels
            level_blocks.append(ResBlock(ch + skip_ch, out_ch, time_emb_dim, dropout))
            ch = out_ch
            if curr_res in attn_resolutions:
                level_attns.append(AttentionBlock(ch))
            # Subsequent ResBlocks: standard, no skip concat
            for _ in range(num_res_blocks):
                level_blocks.append(ResBlock(ch, out_ch, time_emb_dim, dropout))
                ch = out_ch
                if curr_res in attn_resolutions:
                    level_attns.append(AttentionBlock(ch))
            self.up_resblocks.append(level_blocks)
            self.up_attns.append(level_attns)

        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = conv3x3(ch, out_channels)

    def forward(self, x, t, condition):
        t_emb = get_timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        h = torch.cat([x, condition], dim=1)
        h = self.conv_in(h)

        # ── Encoder ──
        skips = []
        for level in range(self.num_levels):
            for block in self.down_resblocks[level]:
                h = block(h, t_emb)
            for attn in self.down_attns[level]:
                h = attn(h)
            skips.append(h)
            if level < len(self.downsamples):
                h = self.downsamples[level](h)

        # ── Middle ──
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # ── Decoder ──
        for level in range(self.num_levels):
            if level > 0:
                h = self.upsamples[level - 1](h)
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            for block in self.up_resblocks[level]:
                h = block(h, t_emb)
            for attn in self.up_attns[level]:
                h = attn(h)

        h = self.norm_out(h)
        h = F.silu(h)
        return self.conv_out(h)


# ── DDPM ────────────────────────────────────────────────

class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model wrapper."""

    def __init__(self, model: UNet, num_timesteps=200, beta_start=1e-4, beta_end=0.02,
                 device="cuda"):
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps
        self.device = device

        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance", betas * (1.0 - alphas_cumprod) / (1.0 - alphas_cumprod))

    def q_sample(self, x0, t, noise=None):
        """Forward diffusion: sample x_t ~ q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * x0 + sqrt_one_minus * noise, noise

    def p_sample(self, xt, t, condition):
        """Reverse diffusion: p(x_{t-1} | x_t, condition)."""
        noise_pred = self.model(xt, t, condition)
        alpha = self.alphas[t][:, None, None, None]
        alpha_cumprod = self.alphas_cumprod[t][:, None, None, None]
        beta = self.betas[t][:, None, None, None]

        # estimate x_0
        x0_pred = (xt - torch.sqrt(1.0 - alpha_cumprod) * noise_pred) / torch.sqrt(alpha_cumprod)

        if t.min() == 0:
            # at t=0, return the clean estimate
            return x0_pred

        # posterior mean
        coef1 = beta * torch.sqrt(alpha) / (1.0 - alpha_cumprod)
        coef2 = (1.0 - alpha_cumprod / alpha) * torch.sqrt(alpha) / (1.0 - alpha_cumprod)
        mean = coef1 * x0_pred + coef2 * xt

        # posterior variance
        log_var = torch.log(beta)  # use β_t as variance (simplified)
        noise = torch.randn_like(xt)
        return mean + torch.exp(0.5 * log_var) * noise

    @torch.no_grad()
    def sample(self, condition, progress=True):
        """Full DDPM reverse diffusion (200 steps)."""
        b = condition.shape[0]
        xt = torch.randn(b, 1, condition.shape[2], condition.shape[3], device=condition.device)

        steps = range(self.num_timesteps - 1, -1, -1)
        if progress:
            from tqdm import tqdm
            steps = tqdm(steps, desc="Sampling")

        for i in steps:
            t = torch.full((b,), i, dtype=torch.long, device=condition.device)
            xt = self.p_sample(xt, t, condition)

        return xt

    @torch.no_grad()
    def ddim_sample(self, condition, ddim_steps=50, eta=0.0, progress=True):
        """DDIM accelerated sampling with configurable steps and stochasticity.

        Args:
            condition:  [B, C, H, W] conditioning tensor
            ddim_steps: number of DDIM steps (≤ num_timesteps)
            eta:        0.0 = deterministic DDIM, 1.0 = DDPM-like stochastic
            progress:   show tqdm progress bar

        Returns:
            x_0 prediction, shape [B, 1, H, W]
        """
        b = condition.shape[0]
        device = condition.device
        xt = torch.randn(b, 1, condition.shape[2], condition.shape[3], device=device)

        # uniform spacing of timesteps
        step_indices = torch.linspace(
            self.num_timesteps - 1, 0, ddim_steps, dtype=torch.long, device=device
        )

        steps_iter = step_indices
        if progress:
            from tqdm import tqdm
            steps_iter = tqdm(step_indices, desc=f"DDIM-{ddim_steps}")

        for k, i in enumerate(steps_iter):
            t = torch.full((b,), i.item(), dtype=torch.long, device=device)
            noise_pred = self.model(xt, t, condition)

            alpha_t = self.alphas_cumprod[t][:, None, None, None]

            # predict x_0 from current x_t and noise
            x0_pred = (xt - torch.sqrt(1.0 - alpha_t) * noise_pred) / torch.sqrt(alpha_t).clamp(min=1e-8)

            if k == len(step_indices) - 1:
                # last step: return clean estimate
                xt = x0_pred
            else:
                prev_i = step_indices[k + 1]
                alpha_prev = self.alphas_cumprod[prev_i].view(-1, 1, 1, 1)

                # DDIM sigma
                if eta > 0:
                    sigma = eta * torch.sqrt(
                        (1.0 - alpha_prev) / (1.0 - alpha_t).clamp(min=1e-8)
                    ) * torch.sqrt(1.0 - alpha_t / alpha_prev.clamp(min=1e-8))
                else:
                    sigma = torch.zeros_like(alpha_t)

                # deterministic part
                xt = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(
                    (1.0 - alpha_prev - sigma ** 2).clamp(min=0)
                ) * noise_pred

                # stochastic noise
                if eta > 0:
                    xt = xt + sigma * torch.randn_like(xt)

        return xt

    def forward(self, x0, condition):
        """Training: sample t, noise, compute loss."""
        b = x0.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        xt, noise = self.q_sample(x0, t, noise)
        noise_pred = self.model(xt, t, condition)
        return F.mse_loss(noise_pred, noise)
