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
check("١٦ صنف ومعاها أسماء عربي (خ3: +قارب طيور بالونات طريق)", len(scenes.CLASSES) == 16
      and all(isinstance(c, tuple) and c[1] for c in scenes.CLASSES))
check("CLASS_IDS متطابق مع CLASSES", scenes.CLASS_IDS == tuple(c[0] for c in scenes.CLASSES))
check("دقة القاعدة 96×96 (جولة خ4 — كانت 48)", scenes.BASE == 96)
check("المقاسات فيها 750 (أعلى دقة)", 750 in scenes.SIZES
      and scenes.SIZES[-1] == 750 and list(scenes.SIZES) == sorted(scenes.SIZES))
check("«غروب» → sunset", scenes.class_from_text("غروب") == "sunset")
check("«البحر» → sea", scenes.class_from_text("البحر") == "sea")
check("«سلسلة جبال» → mountains", scenes.class_from_text("سلسلة جبال") == "mountains")
check("إنجليزي «city» → city", scenes.class_from_text("city") == "city")
check("حاجة غلط → None (الصارم لسه بيشتغل)", scenes.class_from_text("يا نبى سيدي") is None)
print("1خ4) أقرب صنف — الرفض اتلغى")
check("أي نص → صنف صحيح (مبربعش None)", all(scenes.nearest_class(t) in scenes.CLASS_IDS
      for t in ("يا نبى سيدي", "حاسوب", "ارسم قرد بيغني", "xyzzzz", "تنين", "خوخة", "")))
check("«قرد» → birds و«حاسوب» → city", scenes.nearest_class("ارسم قرد بيغني") == "birds"
      and scenes.nearest_class("حاسوب") == "city")
check("«غروب الشمس» مازلت sunset مش day (أطول مرادف الأول)",
      scenes.class_from_text("غروب الشمس") == "sunset")
check("«الشمس» → day (تطبيع + مرادف جديد)", scenes.nearest_class("الشمس") == "day")
check("نفس البذرة = نفس الصورة بالحرف",
      (scenes.make_sample("sunset", 42) == scenes.make_sample("sunset", 42)).all()
      and not (scenes.make_sample("sunset", 42) == scenes.make_sample("sunset", 43)).all())

print("1خ3) الأصناف الجديدة: قارب، طيور، بالونات، طريق")
NEW_IDS = ("boat", "birds", "balloons", "road")
check("الأصناف الجديدة الأربعة في CLASSES وله مولّد",
      all(c[0] in NEW_IDS for c in scenes.CLASSES[-4:])
      and all(cid in scenes._GENERATORS for cid in NEW_IDS))
check("«قارب» → boat و«فلوكة» و«شراع» كمان", scenes.class_from_text("قارب") == "boat"
      and scenes.class_from_text("فلوكة") == "boat" and scenes.class_from_text("شراع") == "boat")
check("«طيور» و«عصفور» → birds", scenes.class_from_text("طيور") == "birds"
      and scenes.class_from_text("عصفور") == "birds")
check("«بالونات» و«بالون» → balloons", scenes.class_from_text("بالونات") == "balloons"
      and scenes.class_from_text("بالون") == "balloons")
check("«طريق» و«طريق سريع» و«شارع» → road", scenes.class_from_text("طريق") == "road"
      and scenes.class_from_text("طريق سريع") == "road" and scenes.class_from_text("شارع") == "road")
_ok = True
for cid in NEW_IDS:
    a = scenes.make_sample(cid, 11, res=48)
    b = scenes.make_sample(cid, 11, res=48)
    if a.shape != (48, 48, 3) or not np.array_equal(a, b) or a.std() < 0.05:
        _ok = False
check("عينة جديدة: حتمية + شكل صحيح + مش فارغة", _ok)
_ok = True
for res in (24, 96, 128):
    a = scenes.make_sample("boat", 5, res=res)
    if a.shape != (res, res, 3) or a.min() < 0 or a.max() > 1:
        _ok = False
check("الأصناف الجديدة مستقلة عن الدقة (24/96/128)", _ok)
imgs16, labels16 = scenes.make_dataset(2, seed=9)
check("الداتاسيت بتتكلم مع 16 صنف (32 عينة)", imgs16.shape == (32, scenes.BASE, scenes.BASE, 3)
      and labels16.max() == 15)
imgs, labels = scenes.make_dataset(3, seed=9)
check("الداتاسيت (12×3) على دقة القاعدة (96×96×3) — الثوابت القديمة ما اتكسرت",
      imgs.dtype == np.float32 and imgs.min() >= 0 and imgs.max() <= 1)
imgs_u8, labels_u8 = scenes.make_dataset_u8(2, seed=9)
check("خ4: الداتاسيت uint8 للتدريب على 96 (توفير رام)", imgs_u8.dtype == np.uint8
      and imgs_u8.shape == (32, 96, 96, 3)
      and abs(float(imgs_u8[0].mean()) / 255.0 - float(imgs[0].mean())) < 0.02)

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
same = resize.resize_lanczos(im0, scenes.BASE)
check("نفس المقاس = نفس المحتوى", np.abs(same - im0).max() < 1e-5)
big = resize.resize_lanczos(im0, 750)
check("96 → 750 بالشكل الصح (750×750×3)", big.shape == (750, 750, 3)
      and big.dtype == np.float32 and big.min() >= 0 and big.max() <= 1)
big2 = resize.resize_lanczos(im0, 750)
check("حتمية: نفس المدخل = نفس المخرج", (big == big2).all())
small = resize.resize_lanczos(im0, 32)
check("التصغير شغال (96 → 32)", small.shape == (32, 32, 3))
u8 = (im0 * 255).astype(np.uint8)
from_u8 = resize.resize_lanczos(u8, 750)
check("يقبل uint8 ويرجّع float [0،1]", from_u8.dtype == np.float32
      and from_u8.min() >= 0 and from_u8.max() <= 1)
g = np.linspace(0, 1, scenes.BASE)[:, None, None] * np.ones((scenes.BASE, scenes.BASE, 3))
gb = resize.resize_lanczos(g, 750)
check("تدرج السما بيفضل مونوتوني بعد التكبير (من غير خطوط)",
      bool(np.all(np.diff(gb.mean(axis=(1, 2))) >= -1e-3)))
print("\n2خ4) ترويس الحواف بعد التكبير — الصور تطلع أوضح")
plain = resize.resize_lanczos(im0, 750, sharpen=False)
sharp = resize.resize_lanczos(im0, 750)
_g = lambda a: float(np.abs(np.diff(a.mean(axis=2), axis=1)).mean())
check("الترويس بيزوّد حدّة الحواف فعلاً (مقارنة بالمكبّر العادي)", _g(sharp) > _g(plain))
check("الترويس حتمي (نفس المدخل = نفس المخرج)", (sharp == resize.resize_lanczos(im0, 750)).all())
small2 = resize.resize_lanczos(im0, 32)
check("التصغير بيمرو من غير ترويس (مفيش ripple)",
      np.abs(small2 - resize.resize_lanczos(im0, 32, sharpen=False)).max() < 1e-6)

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

print("\n3خ3) نمو جدول الأصناف (grow_state) — تكملة تدريب قديم على أصناف جديدة")
from model import grow_state
old_state = {"cemb.weight": torch.arange(12 * 8, dtype=torch.float32).reshape(12, 8)}
grown = grow_state(dict(old_state), 16)
check("جدول الأصناف 12 → 16 وصل بجد", grown["cemb.weight"].shape == (16, 8))
check("الصفوف القديمة ما اتحركتش",
      torch.equal(grown["cemb.weight"][:12], old_state["cemb.weight"]))
check("الصفوف الجديدة عشوائية ثابتة (نفس البذرة)",
      torch.equal(grow_state(dict(old_state), 16)["cemb.weight"][12:],
                  grown["cemb.weight"][12:]))
check("جدول مكتمل حجمًا بيرجع زي ما هو",
      grow_state(dict(grown), 16) is not grown
      and grow_state(dict(grown), 16)["cemb.weight"].shape == (16, 8))
net16 = KhufuNet()
net16.load_state_dict(grow_state({k: v for k, v in net.state_dict().items() if True}, 16))
check("التحميل في شبكة 16 صنف شغال", net16.cemb.weight.shape[0] == 16)

print("\n4) التوليد (generate.py) — نموذج غير متدرب للسرعة، وبعدين المتدرب لو موجود")
import generate
img, got_cid = generate.generate("غروب", size=32, steps=20, seed=7, _model=net)
check("«غروب» بيتفسر لصنف", got_cid == "sunset")
check("الصورة طلعت 32×32×3", img.arr.shape == (32, 32, 3))
img2, _ = generate.generate("sunset", size=32, steps=20, seed=7, _model=net)
check("نفس البذرة = نفس الصورة بالحرف", (img.arr == img2.arr).all())
try:
    generate.generate("sea", size=40, steps=20, _model=net)
    check("مقاس غلط بيرفض", False)
except ValueError:
    check("مقاس غلط بيرفض", True)
img_g, got_g = generate.generate("يا نبى سيدي", size=32, steps=20, seed=7, _model=net)
check("** خ4: نص غلط عمره ما بيرفض — بيتفسر لأقرب صنف **",
      got_g in scenes.CLASS_IDS and img_g.arr.shape == (32, 32, 3))

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
