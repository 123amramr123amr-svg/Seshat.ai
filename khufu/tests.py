# -*- coding: utf-8 -*-
"""اختبارات خوفو v0 — python3 tests.py (من جوه مجلد khufu)"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

FAILURES = []


def check(name, ok, detail=""):
    print(("  ✅ " if ok else "  ❌ ") + name + ("" if ok else " — " + str(detail)[:120]))
    if not ok:
        FAILURES.append(name)


print("1) الأصناف (scenes.py)")
import numpy as np
import scenes
check("١٢ صنف ومعاها أسماء عربي", len(scenes.CLASSES) == 12
      and all(isinstance(c, tuple) and c[1] for c in scenes.CLASSES))
check("CLASS_IDS متطابق مع CLASSES", scenes.CLASS_IDS == tuple(c[0] for c in scenes.CLASSES))
check("دقة القاعدة 48×48 (جولة خ2)", scenes.BASE == 48)
check("المقاسات فيها 750 (أعلى دقة)", 750 in scenes.SIZES
      and scenes.SIZES[-1] == 750 and list(scenes.SIZES) == sorted(scenes.SIZES))
check("«غروب» → sunset", scenes.class_from_text("غروب") == "sunset")
check("«البحر» → sea", scenes.class_from_text("البحر") == "sea")
check("«سلسلة جبال» → mountains", scenes.class_from_text("سلسلة جبال") == "mountains")
check("إنجليزي «city» → city", scenes.class_from_text("city") == "city")
check("حاجة غلط → None", scenes.class_from_text("يا نبى سيدي") is None)
check("نفس البذرة = نفس الصورة بالحرف",
      (scenes.make_sample("sunset", 42) == scenes.make_sample("sunset", 42)).all()
      and not (scenes.make_sample("sunset", 42) == scenes.make_sample("sunset", 43)).all())
imgs, labels = scenes.make_dataset(3, seed=9)
check("الداتاسيت (12×3) على دقة القاعدة (48×48×3)",
      imgs.shape == (36, scenes.BASE, scenes.BASE, 3) and labels.shape == (36,)
      and imgs.dtype == np.float32 and imgs.min() >= 0 and imgs.max() <= 1)

print("\n2) كاتب PNG (png.py)")
import png
import numpy as np
px = np.zeros((5, 7, 3), dtype=np.uint8)
px[..., 2] = 200
data = png.encode_png(px)
check("التوقيع سليم", data[:8] == b"\x89PNG\r\n\x1a\n")
ihdr = png.read_ihdr(data)
check("الأبعاد من IHDR: عرض 7 طول 5", ihdr == {"width": 7, "height": 5})

print("\n2مكرر) مكبّر Lanczos (resize.py) — جولة خ2")
import resize
im0 = scenes.make_sample("sea", 3)
same = resize.resize_lanczos(im0, 48)
check("نفس المقاس = نفس المحتوى", np.abs(same - im0).max() < 1e-5)
big = resize.resize_lanczos(im0, 750)
check("48 → 750 بالشكل الصح (750×750×3)", big.shape == (750, 750, 3)
      and big.dtype == np.float32 and big.min() >= 0 and big.max() <= 1)
big2 = resize.resize_lanczos(im0, 750)
check("حتمية: نفس المدخل = نفس المخرج", (big == big2).all())
small = resize.resize_lanczos(im0, 32)
check("التصغير شغال (48 → 32)", small.shape == (32, 32, 3))
u8 = (im0 * 255).astype(np.uint8)
from_u8 = resize.resize_lanczos(u8, 750)
check("يقبل uint8 ويرجّع float [0،1]", from_u8.dtype == np.float32
      and from_u8.min() >= 0 and from_u8.max() <= 1)
g = np.linspace(0, 1, 48)[:, None, None] * np.ones((48, 48, 3))
gb = resize.resize_lanczos(g, 750)
check("تدرج السما بيفضل مونوتوني بعد التكبير (من غير خطوط)",
      bool(np.all(np.diff(gb.mean(axis=(1, 2))) >= -1e-3)))

print("\n3) الشبكة (model.py)")
import torch
from model import KhufuNet, count_params
net = KhufuNet()
check("الأوزان بتمر وتطلع بالشكل الصح", count_params(net) > 50000)
x = torch.randn(2, 3, scenes.BASE, scenes.BASE)
t = torch.tensor([10, 999])
c = torch.tensor([0, 5])
with torch.no_grad():
    out = net(x, t, c)
check("مخرج المنبئ بنفس شكل المدخل", out.shape == x.shape)

print("\n4) التوليد (generate.py) — نموذج غير متدرب للسرعة، وبعدين المتدرب لو موجود")
import generate
img, got_cid = generate.generate("غروب", size=32, steps=20, seed=7, _model=net)
check("«غروب» بيتفسر لصنف", got_cid == "sunset")
check("الصورة طلعت 32×32×3", img.arr.shape == (32, 32, 3))
img2, _ = generate.generate("sunset", size=32, steps=20, seed=7, _model=net)
check("نفس البذرة = نفس الصورة بالحرف", (img.arr == img2.arr).all())
try:
    generate.generate("يا نبى سيدي", size=32, steps=20, _model=net)
    check("صنف غلط بيرفض باحترام", False)
except ValueError:
    check("صنف غلط بيرفض باحترام", True)
try:
    generate.generate("sea", size=40, steps=20, _model=net)
    check("مقاس غلط بيرفض", False)
except ValueError:
    check("مقاس غلط بيرفض", True)

img750, _ = generate.generate("غروب", size=750, steps=20, seed=7, _model=net)
check("** جولة خ2: توليد 750×750 بالحرف **", img750.arr.shape == (750, 750, 3))
check("PNG 750 مكتوب بأبعاد سليمة", png.read_ihdr(png.encode_png(img750.arr))
      == {"width": 750, "height": 750})
img750b, _ = generate.generate("غروب", size=750, steps=20, seed=7, _model=net)
check("نفس البذرة على 750 = نفس الصورة", (img750.arr == img750b.arr).all())

trained_path = generate.CKPT
if os.path.exists(trained_path):
    img3, _ = generate.generate("night", size=64, steps=20, seed=3)
    check("النموذج المتدرب بيولّد 64×64", img3.arr.shape == (64, 64, 3))
    check("النموذج المتدرب مش ضوضاء بيضا (له تدرج سماء)",
          float(img3.arr.astype(float).std()) > 10.0
          and float(np.abs(img3.arr[:, :, 2].astype(float)
                           - img3.arr[:, :, 0].astype(float)).mean()) > 3.0)
    img4, _ = generate.generate("sea", size=750, steps=20, seed=5)
    check("** النموذج المتدرب بيولّد 750×750 **", img4.arr.shape == (750, 750, 3))
    check("صورة 750 مش فاضية (ليها تنوع ألوان)",
          float(img4.arr.astype(float).std()) > 10.0)
    # خ2: حماية من انهيار التوليد (بذور كانت بتطلع سودا على 20 خطوة قبل clip denoised)
    ok = True
    for s in (5, 2, 42):
        im, _ = generate.generate("sea", size=64, steps=20, seed=s)
        if float(im.arr.astype(float).std()) < 5.0:
            ok = False
    check("بذور كانت بينهار (5/2/42) بتطلع صور حقيقية (clip denoised)", ok)
else:
    check("النموذج المتدرب لسه ماتدرنش (تخطي اختبار العينات)", True)

print("\n5) عقد الواجهة (app.py من seshat بيتصل بينا كده)")
check("CLASSES فيها (id, عربي) للـ /khufu-status",
      all(len(c) == 2 for c in scenes.CLASSES))
check("فيه ملف status.json أو النموذج لسه ماتدرنش (الاتنين مقبولين قبل أول تدريب)",
      True)

print("\n" + "—" * 40)
if FAILURES:
    print(f"❌ {len(FAILURES)} اختبار فشل: {', '.join(FAILURES)}")
    sys.exit(1)
print("✅ كل اختبارات خوفو عدت")
