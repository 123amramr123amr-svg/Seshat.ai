# -*- coding: utf-8 -*-
"""
khufu/generate.py — التوليد من نموذج خوفو + كائن الصورة
=======================================================
generate(cid, size, steps, seed) → (KhufuImage, cid)
  • cid: أي نص عربي/إنجليزي بيتفسر لأقرب صنف (خ4: مفيش رفض — nearest_class)،
    والأصناف نفسها في scenes.CLASS_IDS.
  • size: 32/64/96/128/256/512/750 — القاعدة بتتولد 96×96 (جولة خ4) — 750×750 أعلى دقة.
  • steps: 20/50/100 — خطوات شيل الضجيج (أكتر = أنعم وأدق).
  • seed: نفس البذرة = نفس الصورة بالحرف (على نفس الدقة).

جولة خ5 (الوضوح): توليد هجين — خوفو (DDPM) بيولّد "الأجواء" (تدرج الألوان والجو العام
بتاع المشهد من أوزانه هو)، والمولّد الإجرائي بيرسم نفس المشهد على دقة المخرج كاملة
بأشكال حادة واضحة، والاتنين بيتخلطوا: الرسمة الحادة هي الأساس والأجواء بتضيف تنفس
لوني طبيعي، وبعدها ترويس حواف خفيف. النتيجة: صور بوكّنها صور — الكائن ظاهر ومقروء
مش غيمة ألوان. حتمية 100% لنفس البذرة.

KhufuImage.save(path) بتكتب PNG بالإيد (png.py) — app.py بيستدعيها زي ما هي
(ده العقد اللي اتوصّل من ناحية seshat)، ومفيش PIL ولا مكتبات صور.
"""

import os
import secrets

import numpy as np
import torch

import png
import resize
import scenes

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "data", "khufu_latest.pt")

T_TOTAL = 1000
BETA_START, BETA_END = 1e-4, 0.02
_model_cache = {}

# خ5 — الرسم الهجين: وزن الرسمة الحادة ضد طبقة الأجواء، ودقة ضغط الأجواء.
# الأجواء بتتضغط على 24×24 وبعدها بتتكبر — الضباب القوي ده بيقطع الأشكال
# (مفيش قارب شبح بيظهر في مكان تاني) وبيسيب بس تدرج الألوان الجوي.
ATMO_PX = 24
CRISP_WEIGHT = 0.78


class KhufuImage:
    """صورة خوفو — مصفوفة uint8 (H, W, 3) + حفظ PNG من غير مكتبات."""

    def __init__(self, arr):
        self.arr = np.ascontiguousarray(arr, dtype=np.uint8)

    @property
    def size(self):
        return self.arr.shape[1], self.arr.shape[0]      # (عرض، طول) زي PIL

    def save(self, path):
        with open(path, "wb") as f:
            f.write(png.encode_png(self.arr))
        return path


def _tables():
    betas = torch.linspace(BETA_START, BETA_END, T_TOTAL)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)
    return alphas, abar


def _khufu_model_module():
    """تحميل model.py بتاع خوفو صراحةً من المجلد بتاعه — لو السيرفر الرئيسي
    حمّل model.py تاني (بتاع seshat النصي) قبلنا، الاستيراد بالاسم بيتلخبط.
    importlib بمسار مطلق بيلغي التعارض ده خالص."""
    global _KM
    if _KM is None:
        import importlib.util as _ilu
        _here = os.path.dirname(os.path.abspath(__file__))
        _spec = _ilu.spec_from_file_location("_khufu_model", os.path.join(_here, "model.py"))
        _KM = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_KM)
    return _KM


_KM = None


def load_model(path=CKPT):
    """تحميل النموذج المتدرب (مع كاش بسيط) — لو مفيش ملف بيشتغل بنموذج عشوائي صغير."""
    _m = _khufu_model_module()
    KhufuNet, grow_state = _m.KhufuNet, _m.grow_state
    key = path + ":" + str(int(os.path.getmtime(path)) if os.path.exists(path) else 0)
    if key in _model_cache:
        return _model_cache[key]
    net = KhufuNet()
    ema = None
    if os.path.exists(path):
        d = torch.load(path, map_location="cpu", weights_only=True)
        ema = d.get("ema") or d["weights"]
        # خ3: تكبير جدول الأصناف لو النموذج من جولة بعدد أصناف أقل
        net.load_state_dict(grow_state({k: torch.as_tensor(v) for k, v in ema.items()},
                                       len(scenes.CLASSES)))
    net.eval()
    _model_cache.clear()
    _model_cache[key] = net
    return net


def _atmosphere(img01, size):
    """طبقة الأجواء (خ5): مخرج خوفو متضغط قوي عشان يفضل منه تدرج الألوان بس —
    من غير أشكال ولا أجسام. حتمية لنفس المدخل."""
    small = resize.resize_lanczos(img01, ATMO_PX, sharpen=False)
    return resize.resize_lanczos(small, size, sharpen=False)


def _resolve(cid):
    if cid in scenes.CLASS_IDS:
        return cid
    # خ4: مفيش رفض تاني — أي نص بيتفسر لأقرب صنف (خوفو عمره ما يقول "مش عارف أرسم")
    return scenes.nearest_class(str(cid))


def generate(cid, size=64, steps=50, seed=None, _model=None):
    """العقد اللي app.py بيستدعيه: (KhufuImage, cid)."""
    cid = _resolve(cid)
    if int(size) not in scenes.SIZES:
        raise ValueError("الأبعاد لازم تكون %s" % (scenes.SIZES,))
    if int(steps) not in scenes.STEPS:
        raise ValueError("الخطوات لازم تكون %s" % (scenes.STEPS,))
    size, steps = int(size), int(steps)
    seed = int(seed) if seed not in (None, "") else secrets.randbelow(2**31)

    model = _model or load_model()
    alphas, abar = _tables()
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((1, 3, scenes.BASE, scenes.BASE), generator=g)
    cls = torch.tensor([scenes.CLASS_IDS.index(cid)])
    # انتبه: torch.unique بيرتب تصاعدي — فبنعمل flip عشان المسير يمشي تنازلي
    # (من 999 = ضوضاء كاملة → لـ 0 = صورة نضيفة)، مش العكس.
    ts = torch.linspace(0, T_TOTAL - 1, steps).round().long().unique().flip(0)
    for i, t in enumerate(ts):
        with torch.no_grad():
            eps = model(x, torch.full((1,), int(t)), cls)
        a = abar[int(t)]
        # خ2: تقييد تقدير الصورة النضيفة في المدى (clip denoised — هو et al. 2020).
        # من غيره خطوة تقدير وحشة في الضوضاء العالية بتخلي التوليد ينفجر
        # لأسود/أبيض كامل — كان بيحصل مع بذور معينة على 20 خطوة.
        x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
        if i < len(ts) - 1:
            anext = abar[int(ts[i + 1])]
            x = anext.sqrt() * x0 \
                + (1 - anext).sqrt() * torch.randn(x.shape, generator=g)
        else:
            x = x0                                      # آخر خطوة من غير ضجيج

    img01 = ((x.clamp(-1, 1) + 1) / 2).numpy()[0].transpose(1, 2, 0)
    if size != scenes.BASE:
        img01 = resize.resize_lanczos(img01, size, sharpen=False)

    # خ5 — الرسم الهجين: الرسمة الحادة (بنفس البذرة) هي الأساس، وأجواء خوفو
    # بتضيف تنفس لوني، وترويس خفيف بيبرز الحواف. الكائن بقى ظاهر ومقروء.
    crisp = np.clip(scenes.make_sample(cid, seed=seed, res=size), 0.0, 1.0)
    atmo = _atmosphere(img01, size)
    blended = CRISP_WEIGHT * crisp + (1.0 - CRISP_WEIGHT) * atmo
    blended = resize._unsharp(blended, amount=0.22)
    img01 = np.clip(blended, 0.0, 1.0)
    return KhufuImage(np.clip(img01 * 255 + 0.5, 0, 255)), cid
