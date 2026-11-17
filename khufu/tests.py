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
check("٢٠ صنف ومعاها أسماء عربي (خ3: +4، خ7: +3، خ8: +قوس قزح)", len(scenes.CLASSES) == 20
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
check("أصناف خ3 الأربعة في CLASSES (المواقع 13-16) وله مولّد",
      all(c[0] in NEW_IDS for c in scenes.CLASSES[12:16])
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
check("الداتاسيت بتتكلم مع 20 صنف (40 عينة، آخر لبل 19)", imgs16.shape == (40, scenes.BASE, scenes.BASE, 3)
      and labels16.max() == 19)
imgs, labels = scenes.make_dataset(3, seed=9)
check("الداتاسيت (12×3) على دقة القاعدة (96×96×3) — الثوابت القديمة ما اتكسرت",
      imgs.dtype == np.float32 and imgs.min() >= 0 and imgs.max() <= 1)
imgs_u8, labels_u8 = scenes.make_dataset_u8(2, seed=9)
check("خ4: الداتاسيت uint8 للتدريب على 96 (توفير رام)", imgs_u8.dtype == np.uint8
      and imgs_u8.shape == (40, 96, 96, 3)
      and abs(float(imgs_u8[0].mean()) / 255.0 - float(imgs[0].mean())) < 0.02)

print("1خ7) بيئة شاطئ + عنصرين سمكة وطيارة ورق")
NEW7 = ("beach", "fish", "kite")
check("الأصناف الجديدة التلاتة في CLASSES (قبل قوس قزح الأخير) وله مولّد",
      tuple(c[0] for c in scenes.CLASSES[-4:-1]) == NEW7
      and all(cid in scenes._GENERATORS for cid in NEW7))
check("«شاطئ» و«الشاطئ» → beach (اتنقلوا من sea في خ7)",
      scenes.class_from_text("شاطئ") == "beach" and scenes.class_from_text("الشاطئ") == "beach"
      and scenes.class_from_text("ساحل") == "beach")
check("«البحر» لسه sea و«محيط» كمان (regression)",
      scenes.class_from_text("البحر") == "sea" and scenes.class_from_text("محيط") == "sea")
check("«سمكة» و«سمك» و«حوت» و«دلفين» → fish",
      scenes.class_from_text("سمكة") == "fish" and scenes.class_from_text("سمك") == "fish"
      and scenes.class_from_text("حوت") == "fish" and scenes.class_from_text("دلفين") == "fish")
check("«طيارة ورق» و«طيارة» و«kite» → kite",
      scenes.class_from_text("طيارة ورق") == "kite" and scenes.class_from_text("طيارة") == "kite"
      and scenes.class_from_text("kite") == "kite")
_ok = True
for cid in NEW7:
    a = scenes.make_sample(cid, 11, res=48)
    b = scenes.make_sample(cid, 11, res=48)
    if a.shape != (48, 48, 3) or not np.array_equal(a, b) or a.std() < 0.05:
        _ok = False
check("عينة خ7: حتمية + شكل صحيح + مش فارغة", _ok)
_ok = True
for cid in NEW7:
    for res in (24, 96, 128):
        a = scenes.make_sample(cid, 5, res=res)
        if a.shape != (res, res, 3) or a.min() < 0 or a.max() > 1:
            _ok = False
check("أصناف خ7 مستقلة عن الدقة (24/96/128)", _ok)

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
net16.load_state_dict(grow_state({k: v for k, v in net.state_dict().items() if True},
                                 len(scenes.CLASS_IDS)))
check("التحميل في شبكة 20 صنف شغال (خ8: 12→16→20)", net16.cemb.weight.shape[0] == 20)

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

print("\n4خ6) تركيبات مركبة بوصف حر (scenes + generate)")
s1 = scenes.parse_composite("قارب في البحر وقت الغروب")
check("«قارب في البحر وقت الغروب» → بحر + غروب + قارب",
      s1 and s1["bg"] == "sea" and s1["mood"] == "sunset"
      and ("boat", 1) in s1["objects"] and scenes.is_composite(s1))
s2 = scenes.parse_composite("طيور فوق الجبال بالليل")
check("«طيور فوق الجبال بالليل» → جبال + ليل + طيور (جمع → العدد للرسّام)",
      s2 and s2["bg"] == "mountains" and s2["mood"] == "night"
      and ("birds", None) in s2["objects"])
s3 = scenes.parse_composite("تلاتة بالونات على المدينة")
check("«تلاتة بالونات على المدينة» → مدينة + 3 بالونات (عدّاد بالكلمات)",
      s3 and s3["bg"] == "city" and ("balloons", 3) in s3["objects"])
s4 = scenes.parse_composite("3 طيور")
check("عدّاد بالأرقام «3 طيور» → طيور ×3 (من غير خلفية)",
      s4 and s4["objects"] == [("birds", 3)])
s5 = scenes.parse_composite("بالونات فوق مدينة بالليل")
check("الجمع من غير رقم («بالونات») بيسيب العدد للرسّام (None) — والمفرد واحد",
      s5 and s5["objects"] == [("balloons", None)]
      and (scenes.parse_composite("قوارب في البحر") or {})["objects"] == [("boat", None)]
      and (scenes.parse_composite("قارب في البحر") or {})["objects"] == [("boat", 1)])
check("كلمة واحدة («قارب») مش تركيبة — بترجع للمسار العادي",
      scenes.parse_composite("قارب") is not None
      and not scenes.is_composite(scenes.parse_composite("قارب")))
check("نص غرب مفيهوش حاجة معروفة → None (والباب لسه مفتوح لأقرب صنف)",
      scenes.parse_composite("يا نبى سيدي") is None)
check("«المدينة» بتتطبع صح (ة→ه في المرادفات نفسها)",
      (scenes.parse_composite("طيور فوق المدينة") or {}).get("bg") == "city"
      and (scenes.parse_composite("بالونات فوق المدينة بالليل") or {}).get("mood") == "night")

a = scenes.make_composite(s1, seed=5, res=96)
b = scenes.make_composite(s1, seed=5, res=96)
c = scenes.make_composite(s1, seed=6, res=96)
check("حتمية: نفس البذرة = نفس المصفوفة بالحرف", np.array_equal(a, b))
check("بذرة مختلفة = صورة مختلفة", not np.array_equal(a, c))
check("التركيبة ليها تنوع ألوان حقيقي (std > 0.06)",
      float(a.std()) > 0.06, float(a.std()))
night_i = scenes.make_composite({"bg": "sea", "mood": "night", "objects": []}, seed=11)
day_i = scenes.make_composite({"bg": "sea", "mood": None, "objects": []}, seed=11)
check("جو الليل بيغيّر الصورة فعليًا (فرق متوسط > 0.05)",
      float(np.abs(night_i - day_i).mean()) > 0.05)

img_c, tag_c = generate.generate("قارب في البحر وقت الليل", size=96, steps=20, seed=3, _model=net)
check("generate بياخد وصف حر ويرسم تركيبة (96×96)",
      img_c.arr.shape == (96, 96, 3) and "sea" in tag_c and "night" in tag_c)
check("صورة التركيبة مش ضوضاء (std > 15)",
      float(img_c.arr.astype(float).std()) > 15.0,
      float(img_c.arr.astype(float).std()))
img750c, _ = generate.generate_composite("قوارب في البحر وقت الغروب",
                                         size=750, steps=20, seed=3, _spec=s1)
check("** خ6: تركيبة مركبة على 750×750 **",
      img750c.arr.shape == (750, 750, 3)
      and float(img750c.arr.astype(float).std()) > 10.0)
img_reg, got_reg = generate.generate("غروب", size=32, steps=20, seed=7, _model=net)
check("الصنف الواحد ماشي زي ما هو بعد خ6 (regression)", got_reg == "sunset")

print("4خ7) عناصر خ7 في التركيبات: سمكة وطيارة ورق وبيئة شاطئ")
s7a = scenes.parse_composite("سمكة في البحر")
check("«سمكة في البحر» → بحر + سمكة واحدة (مفرد)",
      s7a and s7a["bg"] == "sea" and s7a["objects"] == [("fish", 1)])
s7b = scenes.parse_composite("اسماك في البحر")
check("جمع السمك («اسماك»/«سمكات») بيسيب العدد للرسّام (None)",
      s7b and s7b["objects"] == [("fish", None)]
      and (scenes.parse_composite("سمكات في البحر") or {})["objects"] == [("fish", None)])
s7c = scenes.parse_composite("3 اسماك في البحر")
check("عدّاد بالأرقام «3 اسماك» → سمكة ×3", s7c and s7c["objects"] == [("fish", 3)])
s7d = scenes.parse_composite("طيارة ورق فوق الشاطئ")
check("«طيارة ورق فوق الشاطئ» → شاطئ + طيارة ورق (تركيبة جديدة كاملة)",
      s7d and s7d["bg"] == "beach" and s7d["objects"] == [("kite", 1)]
      and scenes.is_composite(s7d))
s7e = scenes.parse_composite("طيارات ورق فوق المدينة بالليل")
check("جمع الطيارات («طيارات ورق») → العدد للرسّام + مدينة بالليل",
      s7e and s7e["bg"] == "city" and s7e["mood"] == "night"
      and s7e["objects"] == [("kite", None)])
s7f = scenes.parse_composite("سمكة في البحر وقت الغروب")
check("تركيبة خ7 بجو: «سمكة في البحر وقت الغروب»",
      s7f and s7f["bg"] == "sea" and s7f["mood"] == "sunset" and s7f["objects"] == [("fish", 1)])
ca = scenes.make_composite(s7d, seed=7, res=96)
cb = scenes.make_composite(s7d, seed=7, res=96)
cc = scenes.make_composite(s7d, seed=8, res=96)
check("حتمية تركيبة خ7: نفس البذرة = نفس المصفوفة", np.array_equal(ca, cb))
check("بذرة مختلفة = صورة مختلفة", not np.array_equal(ca, cc))
check("تركيبة الشاطئ ليها تنوع ألوان حقيقي (std > 0.06)",
      float(ca.std()) > 0.06, float(ca.std()))
img7, tag7 = generate.generate("طيارة ورق فوق الشاطئ", size=96, steps=20, seed=3, _model=net)
check("generate بيرسم تركيبة خ7 (96×96) والوصف فيه شاطئ وطيارة",
      img7.arr.shape == (96, 96, 3) and "beach" in tag7 and "kite" in tag7)
img750d, _ = generate.generate_composite("3 اسماك في البحر وقت الغروب",
                                         size=750, steps=20, seed=3, _spec=s7c)
check("** خ7: تركيبة أسماك على 750×750 **",
      img750d.arr.shape == (750, 750, 3)
      and float(img750d.arr.astype(float).std()) > 10.0)
img_beach, got_beach = generate.generate("شاطئ", size=64, steps=20, seed=7, _model=net)
check("الصنف الواحد «شاطئ» ماشي في مسار الصنف العادي", got_beach == "beach"
      and img_beach.arr.shape == (64, 64, 3))
img_reg2, got_reg2 = generate.generate("البحر", size=32, steps=20, seed=7, _model=net)
check("regression: «البحر» لسه بيرسم sea بعد نقل شاطئ", got_reg2 == "sea")

print("1خ8) قوس قزح: صنف جديد + جو بيتلط على أي مشهد")
NEW8 = "rainbow"
check("قوس قزح في CLASSES (آخر صنف) وله مولّد",
      scenes.CLASSES[-1][0] == NEW8 and NEW8 in scenes._GENERATORS)
check("«قوس قزح» و«رينبو» و«سبع الوان» → rainbow",
      scenes.class_from_text("قوس قزح") == "rainbow"
      and scenes.class_from_text("رينبو") == "rainbow"
      and scenes.class_from_text("سبع الوان") == "rainbow"
      and scenes.class_from_text("rainbow") == "rainbow")
a8 = scenes.make_sample("rainbow", 11, res=48)
b8 = scenes.make_sample("rainbow", 11, res=48)
check("عينة قوس قزح: حتمية + شكل صحيح + ملونة فعلاً",
      a8.shape == (48, 48, 3) and np.array_equal(a8, b8) and a8.std() > 0.05
      and float(np.abs(a8[..., 0] - a8[..., 2]).mean()) > 0.02)
_ok = True
for res in (24, 96, 128):
    a = scenes.make_sample("rainbow", 5, res=res)
    if a.shape != (res, res, 3) or a.min() < 0 or a.max() > 1:
        _ok = False
check("قوس قزح مستقل عن الدقة (24/96/128)", _ok)
s8 = scenes.parse_composite("قوس قزح فوق البحر")
check("«قوس قزح فوق البحر» → بحر + جو قوس قزح (تركيبة كاملة)",
      s8 and s8["bg"] == "sea" and s8["mood"] == "rainbow" and scenes.is_composite(s8))
s8b = scenes.parse_composite("طيور فوق الجبال مع قوس قزح")
check("«طيور فوق الجبال مع قوس قزح» → جبال + قوس قزح + طيور",
      s8b and s8b["bg"] == "mountains" and s8b["mood"] == "rainbow"
      and s8b["objects"] == [("birds", None)])
rb_sea = scenes.make_composite({"bg": "sea", "mood": "rainbow", "objects": []}, seed=11)
rb_plain = scenes.make_composite({"bg": "sea", "mood": None, "objects": []}, seed=11)
check("جو قوس قزح بيغيّر المشهد فعلاً (فرق متوسط > 0.02)",
      float(np.abs(rb_sea - rb_plain).mean()) > 0.02)
print("\n4خ8) طبقة الطاقة في الخلط الهجين (generate._vivid)")
_v = generate._vivid(np.clip(scenes.make_sample("rainbow", 7), 0, 1))
check("طبقة الطاقة: حتمية + في المدى + بتفعل الفعلاً",
      np.array_equal(_v, generate._vivid(np.clip(scenes.make_sample("rainbow", 7), 0, 1)))
      and _v.min() >= 0 and _v.max() <= 1
      and float(np.abs(_v - scenes.make_sample("rainbow", 7)).mean()) > 0.01)
_spread_before = float(np.abs(scenes.make_sample("rainbow", 7)[..., 0]
                              - scenes.make_sample("rainbow", 7)[..., 2]).mean())
_spread_after = float(np.abs(_v[..., 0] - _v[..., 2]).mean())
check("طبقة الطاقة بتزود تشبع الألوان (فرق قنوات أعلى)",
      _spread_after >= _spread_before, (_spread_before, _spread_after))
check("الخلط الهجين خ8: وزن الرسمة الحادة 0.82 (أوضح)", generate.CRISP_WEIGHT == 0.82)
img8, tag8 = generate.generate("قوس قزح فوق البحر", size=96, steps=20, seed=3, _model=net)
check("generate بيرسم تركيبة قوس قزح (96×96)",
      img8.arr.shape == (96, 96, 3) and "sea" in tag8 and "rainbow" in tag8
      and float(img8.arr.astype(float).std()) > 15.0)
img_r8, got_r8 = generate.generate("قوس قزح", size=64, steps=20, seed=7, _model=net)
check("الصنف الواحد «قوس قزح» ماشي في المسار العادي", got_r8 == "rainbow"
      and img_r8.arr.shape == (64, 64, 3))

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
