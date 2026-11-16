# -*- coding: utf-8 -*-
"""
khufu/model.py — شبكة خوفو: مُنبئ ضجيج (UNet مصغر) لتوليد الصور بنمط الانتشار
==============================================================================
الشبكة صغيرة عمدًا (~110 ألف معامل) عشان تتدرب على CPU بنواتين في دقايق —
المشاهد اللي خوفو بيتعلمها ناعمة ومتكررة، فمش محتاج شبكة ضخمة.

المدخل: صورة 24×24 مضافة عليها ضجيج + رقم الخطوة الزمنية + رقم الصنف.
المخرج: تقدير الضجيج نفسه — وتوليد الصورة بيتعمل بشيل الضجيج على خطوات
(20/50/100 — نفس أزرار الجودة في الواجهة).
"""

import math

import torch
import torch.nn as nn

from scenes import BASE, CLASSES


def timestep_embedding(t, dim):
    """تضمين جيبي للخطوة الزمنية (زي Transformer sinusoidal)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
    ang = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class Block(nn.Conv2d):
    """Conv + حقن (الزمن + الصنف) كـ bias بعد كل استدعاء."""

    def __init__(self, cin, cout, cond_dim, stride=1):
        super().__init__(cin, cout, 3, stride=stride, padding=1)
        self.cond = nn.Linear(cond_dim, cout)

    def forward(self, x, cond):
        h = super().forward(x)
        return h + self.cond(cond)[:, :, None, None]


class KhufuNet(nn.Module):
    """UNet مصغر: 24×24 → 12×12 → وسط → رجوع 24×24 مع skip."""

    def __init__(self, n_classes=len(CLASSES), base=32, mid=48, wide=64, tdim=64):
        super().__init__()
        cond = tdim                                  # (زمن + صنف) بيمروا مع بعض
        self.tmlp = nn.Sequential(nn.Linear(tdim, tdim * 2), nn.SiLU(),
                                  nn.Linear(tdim * 2, cond))
        self.cemb = nn.Embedding(n_classes, tdim)
        self.inp = Block(3, base, cond)
        self.down = Block(base, mid, cond, stride=2)
        self.mid1 = Block(mid, wide, cond)
        self.mid2 = Block(wide, mid, cond)
        self.up = nn.ConvTranspose2d(mid, base, 4, stride=2, padding=1)
        self.mrg = Block(base + base, base, cond)    # بعد دمج الـ skip
        self.out = Block(base, 3, cond)
        self.norms = nn.ModuleList([nn.GroupNorm(8, c) for c in
                                    (base, mid, wide, mid, base, base)])
        self.act = nn.SiLU()
        self._cond_dim = cond

    def forward(self, x, t, cls):
        cond = self.tmlp(timestep_embedding(t, self.tmlp[0].in_features)) \
            + self.cemb(cls)
        n0, n1, n2, n3, n4, n5 = self.norms
        h = self.act(n0(self.inp(x, cond)))          # 24×24 × base
        skip = h
        h = self.act(n1(self.down(h, cond)))         # 12×12 × mid
        h = self.act(n2(self.mid1(h, cond)))         # 12×12 × wide
        h = self.act(n3(self.mid2(h, cond)))
        h = self.up(h)                               # 24×24 × base
        h = self.act(n4(self.mrg(torch.cat([h, skip], dim=1), cond)))
        return self.out(h, cond)                     # 24×24 × 3 (تقدير الضجيج)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
