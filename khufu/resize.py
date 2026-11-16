# -*- coding: utf-8 -*-
"""
khufu/resize.py — مكبّر Lanczos3 بـ numpy خالص (مفيش PIL ولا مكتبات صور)
========================================================================
جولة خ2: خوفو بقى بيطلع صور حتى 750×750. القاعدة بتتولد 48×48 والمقاسات
الأكبر بتتكبّر هنا — Lanczos3 (sinc windowed sinc) بيحافظ على الحواف أنضف
من bilinear في التكبير الكبير (31× زي 24→750 كان هيطلع مهضوم، و15.6× من 48
بيطلع حاد ومرتاح للعين).

الفكرة: تكبير منفصل (أفقي ثم رأسي) بمصفوفة أوزان لكل محور — كل صف مخرج
جاي من تركيبة خطية موزونة لصفوف المصدر، والحواف بتتعامل بالتكرار مع تطبيع
الصف عشان السطوع ميتغيرش عند الحدود.
"""

import numpy as np

_A = 3.0                                   # نصف قطر نافذة Lanczos3


def _lanczos(x, a=_A):
    """نواة Lanczos: sinc(x)·sinc(x/a) لـ |x| < a، وصفر بره — و1 عند الصفر."""
    x = np.abs(np.asarray(x, dtype=np.float64))
    out = np.zeros_like(x)
    inside = x < a
    xi = x[inside]
    pi_x = np.pi * xi
    den = np.where(xi < 1e-8, 1.0, pi_x * pi_x)     # مقام آمن — عشان 0/0 ما يعملش nan
    out[inside] = np.where(xi < 1e-8, 1.0, (a * np.sin(pi_x) * np.sin(pi_x / a)) / den)
    return out


def _axis_matrix(src, dst, a=_A):
    """مصفوفة أوزان (dst × src) لمحور واحد: صف لكل بكسل مخرج.
    التكبير: نصف قطر a — التصغير: النصف قطر بيتوسع عشان ميعديش على تفاصيل."""
    scale = dst / src
    f = min(scale, 1.0)                    # معامل ضغط النواة عند التصغير
    centers = (np.arange(dst) + 0.5) / scale - 0.5
    W = np.zeros((dst, src))
    for i in range(dst):
        lo = int(np.floor(centers[i] - a / f))
        hi = int(np.ceil(centers[i] + a / f))
        js = np.arange(lo, hi + 1)
        wgt = _lanczos((js - centers[i]) * f, a)
        jc = np.clip(js, 0, src - 1)       # حواف مكررة (الكتلة بتتحفظ)
        np.add.at(W[i], jc, wgt)
    s = W.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return W / s


def _unsharp(img, amount=0.35):
    """ترويس حواف خفيف (unsharp mask): الصورة + amount × (الصورة − نسخة ناعمة منها).
    بيبرز الحواف والتفاصيل بعد التكبير الكبير — خ4: عشان الصور ما تطلعش "مش واضحة".
    ناعم بمصفوفة 3×3 حتمية بـ numpy خالص، والحدود بيتعامل معاها بالتكرار."""
    pad = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")
    blur = (pad[:-2, :-2] + pad[:-2, 1:-1] + pad[:-2, 2:]
            + pad[1:-1, :-2] + pad[1:-1, 1:-1] + pad[1:-1, 2:]
            + pad[2:, :-2] + pad[2:, 1:-1] + pad[2:, 2:]) / 9.0
    return img + amount * (img - blur)


def resize_lanczos(arr, size, a=_A, sharpen=True):
    """
    صورة (H, W, 3) — float في [0,1] أو uint8 — → (size, size, 3) float32 في [0,1].
    نفس المقاس = نفس المحتوى (بتحويل آمن للنوع). حتمية 100% لنفس المدخلات.
    sharpen=True (الافتراضي): التكبير بس بييجي بعده ترويس حواف خفيف — خ4 —
    والتصغير ونفس المقاس بيمروا زي ما هم عشان ما يتعملوش ripple.
    """
    src = np.asarray(arr)
    img = src.astype(np.float64)
    if src.dtype == np.uint8:
        img = img / 255.0
    h, w = img.shape[0], img.shape[1]
    if (h, w) == (size, size):
        return img.astype(np.float32)
    wx = _axis_matrix(w, size, a)          # تكبير أفقي
    hy = _axis_matrix(h, size, a)          # تكبير رأسي
    tmp = np.einsum("hwc,xw->hxc", img, wx)
    out = np.einsum("hxc,yh->yxc", tmp, hy)
    if size > max(h, w) and sharpen:       # التكبير بس بيتروش — خ4
        out = _unsharp(out)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
