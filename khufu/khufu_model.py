"""Khufu: a small diffusion model that generates images from its own weights.

Architecture: a tiny UNet (epsilon-prediction DDPM) with class conditioning,
sampled with DDIM. Fully convolutional, so it can generate at sizes other than
the training size (32/64/96/128). No external service is involved at any point.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

T = 1000  # diffusion steps (training)


def sinusoidal(t, dim):
    half = dim // 2
    freqs = torch.exp(torch.arange(half, dtype=torch.float32) * (-math.log(10000.0) / half))
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([args.sin(), args.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, time_dim, class_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, cout))
        self.class_mlp = nn.Sequential(nn.SiLU(), nn.Linear(class_dim, cout))
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, t_emb, y_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = h + self.class_mlp(y_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(F.silu(self.norm(x))).view(b, 3, c, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) / math.sqrt(c), dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).view(b, c, h, w)
        return x + self.proj(out)


class UNet(nn.Module):
    def __init__(self, base=32, n_classes=19, time_dim=128, class_dim=128, blocks=1):
        super().__init__()
        chs = [base, base * 2, base * 4, base * 4]
        self.time_dim = time_dim
        self.in_conv = nn.Conv2d(3, chs[0], 3, padding=1)
        self.class_emb = nn.Embedding(n_classes, class_dim)

        def RB(cin, cout):
            return ResBlock(cin, cout, time_dim, class_dim)

        def level(cin, cout):
            return nn.ModuleList([RB(cin, cout)] + [RB(cout, cout) for _ in range(blocks - 1)])

        self.d1 = level(chs[0], chs[0])
        self.d2 = level(chs[0], chs[1])
        self.d3 = level(chs[1], chs[2])
        self.d4 = level(chs[2], chs[3])
        self.a3 = AttnBlock(chs[2])
        self.a4 = AttnBlock(chs[3])
        self.mid = nn.ModuleList([RB(chs[3], chs[3]) for _ in range(blocks)])
        self.ma = AttnBlock(chs[3])
        self.u3 = level(chs[3] + chs[2], chs[2])
        self.u2 = level(chs[2] + chs[1], chs[1])
        self.u1 = level(chs[1] + chs[0], chs[0])
        self.out_norm = nn.GroupNorm(8, chs[0])
        self.out_conv = nn.Conv2d(chs[0], 3, 3, padding=1)

    def forward(self, x, t, y):
        t_emb = sinusoidal(t, self.time_dim)
        y_emb = self.class_emb(y)
        h = self.in_conv(x)
        skips = []
        for blk in self.d1:
            h = blk(h, t_emb, y_emb)
        skips.append(h)
        h = F.avg_pool2d(h, 2)
        for blk in self.d2:
            h = blk(h, t_emb, y_emb)
        skips.append(h)
        h = F.avg_pool2d(h, 2)
        for blk in self.d3:
            h = blk(h, t_emb, y_emb)
        h = self.a3(h)
        skips.append(h)
        h = F.avg_pool2d(h, 2)
        for blk in self.d4:
            h = blk(h, t_emb, y_emb)
        h = self.a4(h)
        for blk in self.mid:
            h = blk(h, t_emb, y_emb)
        h = self.ma(h)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = torch.cat([h, skips[2]], 1)
        for blk in self.u3:
            h = blk(h, t_emb, y_emb)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = torch.cat([h, skips[1]], 1)
        for blk in self.u2:
            h = blk(h, t_emb, y_emb)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = torch.cat([h, skips[0]], 1)
        for blk in self.u1:
            h = blk(h, t_emb, y_emb)
        return self.out_conv(F.silu(self.out_norm(h)))


def get_schedule(T=T):
    steps = torch.arange(T + 1, dtype=torch.float32)
    f = torch.cos(((steps / T) + 0.008) / 1.008 * math.pi / 2) ** 2
    betas = torch.clamp(1 - f[1:] / f[:-1], 0.0001, 0.02)
    alphas = 1 - betas
    acp = torch.cumprod(alphas, 0)
    return betas, alphas, acp


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}

    def apply_to(self, model):
        model.load_state_dict(self.shadow)


@torch.no_grad()
def sample_ddim(model, y, size=64, steps=50, seed=None, eta=0.0):
    """DDIM sampling. y: LongTensor (B,). Returns float tensor (B,3,size,size) in [0,1]."""
    _, _, acp = get_schedule()
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    x = torch.randn(len(y), 3, size, size, generator=g)
    ts = torch.linspace(T - 1, 0, steps + 1)
    for i in range(steps):
        t = int(ts[i])
        t_b = torch.full((len(y),), t, dtype=torch.long)
        eps = model(x, t_b, y)
        acp_t = float(acp[t])
        x0 = ((x - math.sqrt(1 - acp_t) * eps) / math.sqrt(acp_t)).clamp(-1, 1)
        if i == steps - 1:
            x = x0
            break
        t_next = int(ts[i + 1])
        acp_next = float(acp[t_next])
        sigma = eta * math.sqrt(max(0.0, (1 - acp_next) / (1 - acp_t) * (1 - acp_t / acp_next)))
        dir_xt = math.sqrt(max(0.0, 1 - acp_next - sigma ** 2)) * eps
        x = math.sqrt(acp_next) * x0 + dir_xt
        if sigma > 0:
            x = x + sigma * torch.randn_like(x, generator=g)
    return (x.clamp(-1, 1) + 1) / 2
