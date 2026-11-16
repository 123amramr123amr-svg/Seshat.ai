# -*- coding: utf-8 -*-
"""
khufu/scenes.py — أصناف المشاهد ومولّد البيانات الإجرائي
=========================================================
خوفو لسه في البداية: مفيش صور حقيقية جاهزة للتدريب، فالبيانات بتتولد إجرائيًا
بكود مكتوب بالإيد (تدرجات سماء، شمس/قمر، نجوم، دوالي رمال، سلاسل جبال، عمارات…)
— آلاف العينات لكل صنف بألوان ومواقع عشوائية، والنموذج بيتعلم توزيع الصنف
ويولّد تركيبات جديدة عمره ما شافها.

**جولة خ2 — الاستقلال عن الدقة:** كل المولدات بترسم بإحداثيات في فضاء 24×24
وبتتضرب في معامل التكبير k = الدقة الفعلية / 24. نفس البذرة على نفس الدقة =
نفس الصورة بالحرف، وعلى دقة أعلى = نفس التكوين بتفاصيل أكتر (أشعة سماء أنعم،
عمارات بنوافذ أكتر، جبال بقمم أدق). ده اللي بيخلي رفع القاعدة 24→48 وانتاج
750×720 ممكن من غير ما نعيد كتابة الرسمة الواحدة يدويًا.

الأساس 48×48 RGB (اترفع من 24 في خ2) — والمقاسات الأكبر (حتى 750) بتتولد
بنفس القاعدة وبتتكبّر بمكبّر Lanczos3 (resize.py).
"""

import numpy as np

BASE = 48                      # دقة القاعدة اللي النموذج بيتدرب عليها (خ2: كانت 24)
SIZES = (32, 64, 96, 128, 256, 512, 750)   # المقاسات المسموحة — 750 أعلى دقة (خ2)
STEPS = (20, 50, 100)          # خطوات الجودة (سلاسل الانتشار)

# الصنف: (المعرّف، الاسم بالعربي) — ده اللي الواجهة بتعرضه
CLASSES = [
    ("day", "نهار صحو"),
    ("sunset", "غروب"),
    ("night", "ليل بنجوم"),
    ("dawn", "فجر"),
    ("sea", "بحر"),
    ("desert", "صحرا"),
    ("mountains", "جبال"),
    ("forest", "غابة"),
    ("city", "مدينة بالليل"),
    ("rain", "مطر"),
    ("snow", "ثلج"),
    ("flowers", "ورود"),
    ("boat", "قارب"),
    ("birds", "طيور"),
    ("balloons", "بالونات"),
    ("road", "طريق"),
]
CLASS_IDS = tuple(c[0] for c in CLASSES)

# مرادفات عربي/إنجليزي — class_from_text بيبتدي من هنا
_SYNONYMS = {
    "day": ("نهار", "صحو", "نهارداي", "صباح", "سماء زرقا", "day", "sunny", "morning"),
    "sunset": ("غروب", "الغروب", "غروب الشمس", "غروب شمس", "المغيب", "sunset", "dusk"),
    "night": ("ليل", "الليل", "بنجوم", "نجوم", "قمر", "سما بالليل", "night", "stars", "moon"),
    "dawn": ("فجر", "الفجر", "سحر", "السحر", "بكور", "dawn", "sunrise", "twilight"),
    "sea": ("بحر", "البحر", "محط", "مية", "مياه", "شاطئ", "الشاطئ", "ساحل", "sea", "ocean", "beach"),
    "desert": ("صحرا", "الصحرا", "صحراء", "رمال", "الرمال", "desert", "dunes", "sand"),
    "mountains": ("جبال", "جبل", "الجبل", "سلسلة جبال", "قمم", "mountains", "mountain", "hills"),
    "forest": ("غابة", "الغابة", "غابا", "أشجار", "اشجار", "شجر", "forest", "trees", "jungle"),
    "city": ("مدينة", "المدينة", "عمارات", "مبانى", "مباني", "building", "city", "skyline"),
    "rain": ("مطر", "الشتا", "شتا", "امطار", "أمطار", "rain", "rainy", "storm"),
    "snow": ("ثلج", "التلج", "تلج", "snow", "snowy", "winter"),
    "flowers": ("ورود", "ورد", "الورد", "زهور", "فل", "flowers", "roses", "garden"),
    "boat": ("قارب", "القارب", "قوارب", "مركب", "فلوكة", "شراع", "سفينة", "boat", "sailboat", "ship"),
    "birds": ("طيور", "الطيور", "طير", "طائر", "عصافير", "عصفور", "birds", "bird"),
    "balloons": ("بالونات", "بالون", "بالونة", "البالونات", "balloon", "balloons"),
    "road": ("طريق", "الطريق", "شارع", "الشارع", "طريق سريع", "اسفلت", "إسفلت", "road", "highway", "street"),
}

_CLASS_BY_ID = dict(CLASSES)


def class_from_text(text):
    """نص حر عربي/إنجليزي → معرّف صنف، أو None لو مش عارفه."""
    t = (text or "").strip().lower()
    if not t:
        return None
    if t in _CLASS_BY_ID:
        return t
    for cid, words in _SYNONYMS.items():
        for w in words:
            if w == t or (len(w) >= 3 and w in t):
                return cid
    return None


def class_name(cid):
    return _CLASS_BY_ID.get(cid, cid)


# ============================================================
# أدوات رسم إجرائية (كل حاجة بأرقام — مفيش مكتبات صور)
# كل أداة بتاخد الصورة وبتقرا دقتها منها، وk = معامل التكبير من فضاء 24
# ============================================================
def _v_gradient(rng, top, bottom, n):
    """تدرج رأسي من لونين مع جيتر خفيف — أساس كل السما."""
    ramp = np.linspace(0.0, 1.0, n)[:, None]
    img = np.zeros((n, n, 3))
    for c in range(3):
        jitter = 1.0 + rng.normal(0.0, 0.015)
        img[..., c] = (top[c] * (1 - ramp) + bottom[c] * ramp) * jitter
    return img


def _horizon(img, row, ground_top, ground_bottom, wobble=1.2, rng=None, k=1.0):
    """الأرض من صف معين لتحت، بخط أفقي متموج خفيف."""
    n = img.shape[0]
    w = max(0.4, wobble * k)
    for x in range(n):
        r = int(row + rng.normal(0.0, w))
        r = max(2, min(n - 1, r))
        ramp = np.linspace(0.0, 1.0, max(1, n - r)).ravel()
        for c in range(3):
            img[r:, x, c] = ground_top[c] * (1 - ramp) + ground_bottom[c] * ramp


def _disc(img, cx, cy, rad, color, glow=0.35, rng=None, k=1.0):
    """قرص (شمس/قمر/وردة) بهالة ناعمة — الأنصاف والأبعاد بتتقاس مع الدقة."""
    n = img.shape[0]
    core_w = 1.2 * k
    halo_w = 3.5 * k
    yy, xx = np.mgrid[0:n, 0:n]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    core = np.clip((rad + 0.6 * k - d) / core_w, 0, 1)
    halo = np.clip((rad + halo_w - d) / halo_w, 0, 1) * glow
    for c in range(3):
        img[..., c] = img[..., c] * (1 - core) + color[c] * core
        img[..., c] = np.clip(img[..., c] * (1 - halo) + color[c] * halo * 0.5, 0, 1)


def _stars(img, rng, count, bright=0.95):
    n = img.shape[0]
    for _ in range(count):
        x, y = rng.integers(0, n), rng.integers(0, max(1, n // 2))
        img[y, x] = np.clip(img[y, x] * 0.2 + bright, 0, 1)


def _rain_lines(img, rng, count, k=1.0):
    n = img.shape[0]
    for _ in range(count):
        x, y = rng.integers(0, n), rng.integers(0, max(1, n - int(4 * k)))
        L = int(rng.integers(3, 5) * k)
        shade = 0.75 + rng.normal(0, 0.05)
        for kk in range(L):
            if y + kk < n:
                img[y + kk, x] = img[y + kk, x] * 0.55 \
                    + np.array([0.8, 0.85, 0.95]) * shade * 0.45


def _snow_dots(img, rng, count, k=1.0):
    n = img.shape[0]
    pair = k >= 1.5                      # على الدقة العالية الثلج بيبقى نقطتين مش بيكسل واحد
    for _ in range(count):
        x, y = rng.integers(0, n), rng.integers(0, n)
        img[y, x] = img[y, x] * 0.3 + 0.7
        if pair and x + 1 < n and y + 1 < n and rng.random() < 0.5:
            img[y, x + 1] = img[y, x + 1] * 0.3 + 0.7
            img[y + 1, x] = img[y + 1, x] * 0.3 + 0.7


def _skyline(img, rng, row, color_dark, color_win, k=1.0):
    """عمارات من خط الأفق لفوق بسلول شباك منورة — العرض/الطول/الشباك بتتقاس."""
    n = img.shape[0]
    x = 0
    while x < n:
        w = max(2, int(rng.integers(3, 6) * k))
        h = int(rng.integers(int(5 * k), max(int(6 * k) + 1, row - int(2 * k))))
        top = row - h
        for xx in range(x, min(n, x + w)):
            img[top:, xx] = img[top:, xx] * 0.15 + np.array(color_dark) * 0.85
        for _ in range(max(1, (w * h) // 6)):
            wx = int(rng.integers(x, min(n, x + w)))
            wy = int(rng.integers(top + 1, max(top + 2, row - 1)))
            img[wy, wx] = np.array(color_win)
        x += w + max(1, int(rng.integers(1, 3) * k))


def _ridge(img, rng, base_row, height, color, sharp=2.2, k=1.0):
    """سلسلة جبال/دوالي: silhouette بتقاطعات خطوط مائلة — الكثافة بتتقاس مع الدقة،
    والقمم بتكبر مع العرض تلقائيًا (المسافة بين نقاط الكسر بتتضاعف)."""
    n = img.shape[0]
    step = max(2, int(rng.integers(3, 7) * k))
    slopes = []
    x0 = 0
    while x0 < n:
        slopes.append((x0, rng.normal(0.0, sharp)))
        x0 += step
    top = np.full(n, float(base_row))
    for i in range(n):
        for (sx, sl) in slopes:
            top[i] = min(top[i], base_row - height * 0.5
                         - abs((i - sx) * sl) * 0.5 + height * 0.5)
        top[i] = max(2, min(base_row, top[i]))
    for x in range(n):
        r = int(top[x])
        img[r:, x] = img[r:, x] * 0.2 + np.array(color) * 0.8


def _trees(img, rng, ground_row, count, trunk=(0.28, 0.18, 0.08),
           leaf_choices=((0.12, 0.42, 0.14), (0.18, 0.5, 0.16), (0.35, 0.55, 0.12)),
           k=1.0):
    n = img.shape[0]
    tw = max(1, int(round(k)))           # جذع بعرض نسبي ثابت
    for _ in range(count):
        x = int(rng.integers(1, max(2, n - 2 - tw)))
        h = int(rng.integers(int(4 * k), max(int(5 * k) + 1, ground_row - int(3 * k))))
        top = max(1, ground_row - h)
        img[top:ground_row, x:x + tw] = \
            img[top:ground_row, x:x + tw] * 0.3 + np.array(trunk) * 0.7
        leaf = np.array(leaf_choices[int(rng.integers(0, len(leaf_choices)))])
        _disc(img, x + tw / 2.0 - 0.5, top - 1, 1.8 * k, leaf, glow=0.15, rng=rng, k=k)


def _fill_poly(img, pts, color, strength=0.9):
    """تعبئة مضلع محدب (هيكل قارب/شراع/طريق) بخلط لوني — الإحداثيات بكسل مباشر،
    وبالاختبار "كل ضربات الاتجاه بنفس الإشارة" فبتشتغل مع أي اتجاه للنقاط."""
    n = img.shape[0]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = max(0, int(min(xs))), min(n - 1, int(max(xs)) + 1)
    y0, y1 = max(0, int(min(ys))), min(n - 1, int(max(ys)) + 1)
    if x1 < x0 or y1 < y0:
        return
    col = np.array(color, dtype=np.float32)

    def side(a, b, p):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            p = (x + 0.5, y + 0.5)
            ss = [side(pts[i], pts[(i + 1) % len(pts)], p) for i in range(len(pts))]
            if all(s >= 0 for s in ss) or all(s <= 0 for s in ss):
                img[y, x] = img[y, x] * (1 - strength) + col * strength


def _stroke(img, x0, y0, x1, y1, color, w=1.0, strength=0.85):
    """خط من نقاط متقاربة (ساري/خيط بالون/طائر V/حواف طريق) — w عرض البكسلات."""
    n = img.shape[0]
    steps = max(2, int(max(abs(x1 - x0), abs(y1 - y0)) * 2))
    col = np.array(color, dtype=np.float32)
    r = max(0, int(round(w)))
    for i in range(steps + 1):
        t = i / steps
        x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
        cx, cy = int(round(x)), int(round(y))
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                xx, yy = cx + dx, cy + dy
                if 0 <= xx < n and 0 <= yy < n:
                    img[yy, xx] = img[yy, xx] * (1 - strength) + col * strength


# ============================================================
# مولّدات الأصناف — كل واحدة بترسم في فضاء 24 وبتتكبّر بk
# ============================================================
def _day(rng, n):
    k = n / 24.0
    sky_top = np.array([0.30, 0.55, 0.85]) + rng.normal(0, 0.04, 3)
    sky_bot = np.array([0.62, 0.78, 0.95]) + rng.normal(0, 0.04, 3)
    img = _v_gradient(rng, np.clip(sky_top, 0, 1), np.clip(sky_bot, 0, 1), n)
    _disc(img, rng.uniform(4, 20) * k, rng.uniform(3, 8) * k,
          rng.uniform(1.5, 2.5) * k, np.array([1.0, 0.97, 0.80]), glow=0.3, rng=rng, k=k)
    _horizon(img, int(rng.integers(17, 20) * k), np.array([0.45, 0.62, 0.35]),
             np.array([0.30, 0.45, 0.24]), rng=rng, k=k)
    return np.clip(img, 0, 1)


def _sunset(rng, n):
    k = n / 24.0
    top = np.array([0.35, 0.22, 0.42]) + rng.normal(0, 0.04, 3)
    bot = np.array([0.98, 0.62, 0.25]) + rng.normal(0, 0.05, 3)
    img = _v_gradient(rng, np.clip(top, 0, 1), np.clip(bot, 0, 1), n)
    _disc(img, rng.uniform(6, 18) * k, rng.uniform(14, 17) * k,
          rng.uniform(1.6, 2.4) * k, np.array([1.0, 0.85, 0.45]), glow=0.5, rng=rng, k=k)
    _horizon(img, int(rng.integers(18, 21) * k), np.array([0.30, 0.18, 0.24]),
             np.array([0.18, 0.10, 0.16]), rng=rng, k=k)
    return np.clip(img, 0, 1)


def _night(rng, n):
    k = n / 24.0
    top = np.array([0.04, 0.06, 0.16]) + rng.normal(0, 0.02, 3)
    bot = np.array([0.10, 0.14, 0.30]) + rng.normal(0, 0.03, 3)
    img = _v_gradient(rng, np.clip(top, 0, 1), np.clip(bot, 0, 1), n)
    _disc(img, rng.uniform(5, 19) * k, rng.uniform(3, 8) * k,
          rng.uniform(1.2, 1.8) * k, np.array([0.95, 0.95, 0.85]), glow=0.25, rng=rng, k=k)
    _stars(img, rng, int(rng.integers(14, 30) * k))
    _horizon(img, int(rng.integers(18, 21) * k), np.array([0.08, 0.12, 0.14]),
             np.array([0.05, 0.08, 0.09]), rng=rng, k=k)
    return np.clip(img, 0, 1)


def _dawn(rng, n):
    k = n / 24.0
    top = np.array([0.45, 0.40, 0.62]) + rng.normal(0, 0.04, 3)
    bot = np.array([0.95, 0.75, 0.70]) + rng.normal(0, 0.04, 3)
    img = _v_gradient(rng, np.clip(top, 0, 1), np.clip(bot, 0, 1), n)
    _disc(img, rng.uniform(6, 18) * k, rng.uniform(15, 18) * k,
          rng.uniform(1.3, 2.0) * k, np.array([1.0, 0.92, 0.80]), glow=0.4, rng=rng, k=k)
    _horizon(img, int(rng.integers(18, 21) * k), np.array([0.35, 0.28, 0.35]),
             np.array([0.22, 0.18, 0.25]), rng=rng, k=k)
    return np.clip(img, 0, 1)


def _sea(rng, n):
    k = n / 24.0
    sky_kind = int(rng.integers(0, 3))
    palettes = {
        0: (np.array([0.35, 0.58, 0.88]), np.array([0.70, 0.82, 0.95])),
        1: (np.array([0.55, 0.35, 0.45]), np.array([0.95, 0.60, 0.30])),
        2: (np.array([0.06, 0.09, 0.20]), np.array([0.14, 0.18, 0.32])),
    }
    top, bot = palettes[sky_kind]
    img = _v_gradient(rng, top + rng.normal(0, 0.03, 3), bot + rng.normal(0, 0.03, 3), n)
    row = int(rng.integers(11, 14) * k)
    water = np.array([0.10, 0.35, 0.55]) + rng.normal(0, 0.04, 3)
    water2 = np.array([0.06, 0.25, 0.42]) + rng.normal(0, 0.03, 3)
    _horizon(img, row, np.clip(water, 0, 1), np.clip(water2, 0, 1), wobble=0.6, rng=rng, k=k)
    for _ in range(int(rng.integers(3, 7) * k)):
        y = int(rng.integers(row + int(2 * k), n - 1))
        x = int(rng.integers(int(2 * k), max(int(2 * k) + 1, n - int(6 * k))))
        w = max(1, int(rng.integers(2, 5) * k))
        img[y, x:x + w] = img[y, x:x + w] * 0.6 + 0.4
    return np.clip(img, 0, 1)


def _desert(rng, n):
    k = n / 24.0
    sky_top = np.array([0.45, 0.60, 0.85]) + rng.normal(0, 0.05, 3)
    sky_bot = np.array([0.85, 0.78, 0.65]) + rng.normal(0, 0.05, 3)
    img = _v_gradient(rng, np.clip(sky_top, 0, 1), np.clip(sky_bot, 0, 1), n)
    row = int(rng.integers(12, 16) * k)
    sand1 = np.array([0.90, 0.75, 0.48]) + rng.normal(0, 0.04, 3)
    sand2 = np.array([0.72, 0.55, 0.30]) + rng.normal(0, 0.04, 3)
    _ridge(img, rng, row, rng.uniform(1.5, 3.5) * k, np.clip(sand1, 0, 1), sharp=1.2, k=k)
    _ridge(img, rng, int(row + 5 * k), rng.uniform(1.0, 2.5) * k,
           np.clip(sand2, 0, 1), sharp=1.0, k=k)
    return np.clip(img, 0, 1)


def _mountains(rng, n):
    k = n / 24.0
    sky_kind = int(rng.integers(0, 2))
    if sky_kind == 0:
        top, bot = np.array([0.35, 0.55, 0.85]), np.array([0.75, 0.85, 0.95])
    else:
        top, bot = np.array([0.40, 0.30, 0.50]), np.array([0.95, 0.65, 0.35])
    img = _v_gradient(rng, top + rng.normal(0, 0.04, 3), bot + rng.normal(0, 0.04, 3), n)
    row = int(rng.integers(13, 16) * k)
    shade = rng.uniform(0.55, 0.8)
    _ridge(img, rng, row, rng.uniform(4, 7) * k,
           np.array([shade * 0.75, shade * 0.72, shade * 0.80]), sharp=2.6, k=k)
    _ridge(img, rng, int(row + 4 * k), rng.uniform(2, 4) * k,
           np.array([0.25 * shade, 0.28 * shade, 0.22 * shade]), sharp=1.8, k=k)
    return np.clip(img, 0, 1)


def _forest(rng, n):
    k = n / 24.0
    sky_top = np.array([0.55, 0.72, 0.90]) + rng.normal(0, 0.05, 3)
    sky_bot = np.array([0.80, 0.88, 0.92]) + rng.normal(0, 0.04, 3)
    img = _v_gradient(rng, np.clip(sky_top, 0, 1), np.clip(sky_bot, 0, 1), n)
    row = int(rng.integers(14, 17) * k)
    _horizon(img, row, np.array([0.30, 0.50, 0.22]), np.array([0.18, 0.35, 0.14]),
             rng=rng, k=k)
    _trees(img, rng, row, int(rng.integers(3, 6) * k), k=k)
    return np.clip(img, 0, 1)


def _city(rng, n):
    k = n / 24.0
    top = np.array([0.08, 0.10, 0.22]) + rng.normal(0, 0.03, 3)
    bot = np.array([0.20, 0.16, 0.32]) + rng.normal(0, 0.04, 3)
    img = _v_gradient(rng, np.clip(top, 0, 1), np.clip(bot, 0, 1), n)
    _stars(img, rng, int(rng.integers(6, 14) * k))
    _skyline(img, rng, int(rng.integers(15, 19) * k),
             color_dark=(0.12, 0.10, 0.14), color_win=(0.95, 0.85, 0.45), k=k)
    return np.clip(img, 0, 1)


def _rain(rng, n):
    k = n / 24.0
    top = np.array([0.35, 0.40, 0.45]) + rng.normal(0, 0.04, 3)
    bot = np.array([0.55, 0.60, 0.65]) + rng.normal(0, 0.04, 3)
    img = _v_gradient(rng, np.clip(top, 0, 1), np.clip(bot, 0, 1), n)
    _horizon(img, int(rng.integers(17, 20) * k), np.array([0.22, 0.25, 0.28]),
             np.array([0.15, 0.17, 0.20]), rng=rng, k=k)
    _rain_lines(img, rng, int(rng.integers(10, 22) * k), k=k)
    return np.clip(img, 0, 1)


def _snow(rng, n):
    k = n / 24.0
    top = np.array([0.55, 0.65, 0.80]) + rng.normal(0, 0.04, 3)
    bot = np.array([0.80, 0.85, 0.92]) + rng.normal(0, 0.04, 3)
    img = _v_gradient(rng, np.clip(top, 0, 1), np.clip(bot, 0, 1), n)
    _horizon(img, int(rng.integers(15, 18) * k), np.array([0.92, 0.94, 0.98]),
             np.array([0.80, 0.84, 0.92]), rng=rng, k=k)
    _ridge(img, rng, int(rng.integers(10, 13) * k), rng.uniform(2, 4) * k,
           np.array([0.85, 0.88, 0.95]), sharp=2.0, k=k)
    _snow_dots(img, rng, int(rng.integers(12, 26) * k), k=k)
    return np.clip(img, 0, 1)


def _flowers(rng, n):
    k = n / 24.0
    sky_top = np.array([0.50, 0.68, 0.88]) + rng.normal(0, 0.05, 3)
    sky_bot = np.array([0.82, 0.85, 0.90]) + rng.normal(0, 0.04, 3)
    img = _v_gradient(rng, np.clip(sky_top, 0, 1), np.clip(sky_bot, 0, 1), n)
    row = int(rng.integers(14, 17) * k)
    _horizon(img, row, np.array([0.25, 0.52, 0.20]), np.array([0.16, 0.38, 0.14]),
             rng=rng, k=k)
    choices = ((0.95, 0.35, 0.45), (0.98, 0.80, 0.30), (0.90, 0.45, 0.75), (0.98, 0.98, 0.95))
    for _ in range(int(rng.integers(4, 9) * k)):
        x = int(rng.integers(1, n - 1))
        y = int(rng.integers(row, n - 1))
        col = np.array(choices[int(rng.integers(0, len(choices)))])
        _disc(img, x, y, rng.uniform(0.8, 1.4) * k, col, glow=0.2, rng=rng, k=k)
    return np.clip(img, 0, 1)


# ============================================================
# جولة خ3 — أصناف جديدة: قارب، طيور، بالونات، طريق
# نفس القاعدة: إحداثيات نسبية في فضاء 24 × k وبذور حتمية
# ============================================================
def _boat(rng, n):
    """بحر وقارب شراعي: هيكل شبه منحرف + ساري + شراع مثلث فوق خط المية."""
    k = n / 24.0
    sky_kind = int(rng.integers(0, 3))
    palettes = {
        0: (np.array([0.35, 0.58, 0.88]), np.array([0.70, 0.82, 0.95])),
        1: (np.array([0.55, 0.35, 0.45]), np.array([0.95, 0.60, 0.30])),
        2: (np.array([0.06, 0.09, 0.20]), np.array([0.14, 0.18, 0.32])),
    }
    top, bot = palettes[sky_kind]
    img = _v_gradient(rng, np.clip(top + rng.normal(0, 0.03, 3), 0, 1),
                      np.clip(bot + rng.normal(0, 0.03, 3), 0, 1), n)
    row = int(rng.integers(12, 14) * k)
    water = np.array([0.10, 0.35, 0.55]) + rng.normal(0, 0.04, 3)
    water2 = np.array([0.06, 0.25, 0.42]) + rng.normal(0, 0.03, 3)
    _horizon(img, row, np.clip(water, 0, 1), np.clip(water2, 0, 1), wobble=0.6, rng=rng, k=k)
    # موضع القارب وحجمة — كله نسبي
    bx = rng.uniform(6, 18) * k
    wy = rng.uniform(row + 1.0, min(n - 3.0, row + 6.0))
    hull_w = rng.uniform(2.6, 3.6) * k
    hull_h = rng.uniform(0.7, 1.1) * k
    sail_h = rng.uniform(3.2, 5.0) * k
    hull = np.clip(np.array([0.42, 0.24, 0.12]) + rng.normal(0, 0.05, 3), 0, 1)
    if rng.random() < 0.6:
        sail = np.clip(np.array([0.96, 0.95, 0.90]), 0, 1)
    else:
        sail = np.clip(np.array([0.85, 0.30, 0.25]) + rng.normal(0, 0.04, 3), 0, 1)
    # الهيكل: شبه منحرف عريض فوق وأضيق تحت
    _fill_poly(img, [(bx - hull_w / 2, wy), (bx + hull_w / 2, wy),
                     (bx + hull_w * 0.32, wy + hull_h), (bx - hull_w * 0.32, wy + hull_h)],
               hull, strength=0.9)
    # الساري والشراع
    mast_x = bx + rng.uniform(-0.4, 0.4) * k
    _stroke(img, mast_x, wy, mast_x, wy - sail_h, np.array([0.30, 0.20, 0.10]),
            w=0.5 * k, strength=0.9)
    side = 1.0 if rng.random() < 0.5 else -1.0
    _fill_poly(img, [(mast_x, wy - sail_h), (mast_x, wy - 0.5 * k),
                     (mast_x + side * rng.uniform(1.5, 2.4) * k, wy - 0.7 * k)],
               sail, strength=0.92)
    # انعكاس خفيف على المية تحت القارب
    ref_w = max(2, int(hull_w * 0.7))
    ref_y = int(wy + hull_h + 1.0 * k)
    if 0 <= ref_y < n:
        x0 = max(0, int(bx - ref_w / 2))
        x1 = min(n, int(bx + ref_w / 2))
        img[ref_y, x0:x1] = img[ref_y, x0:x1] * 0.75 + 0.12
    # موجات خفيفة
    for _ in range(int(rng.integers(2, 5) * k)):
        y = int(rng.integers(row + int(1.5 * k), n - 1))
        x = int(rng.integers(int(1 * k), max(2, n - int(5 * k))))
        w = max(1, int(rng.integers(2, 5) * k))
        img[y, x:x + w] = img[y, x:x + w] * 0.65 + 0.35
    return np.clip(img, 0, 1)


def _birds(rng, n):
    """سما نهار أو غروب وطيور صغيرة على شكل V بأحجام ومواقع عشوائية."""
    k = n / 24.0
    if rng.random() < 0.65:
        top = np.array([0.30, 0.55, 0.85]) + rng.normal(0, 0.04, 3)
        bot = np.array([0.62, 0.78, 0.95]) + rng.normal(0, 0.04, 3)
    else:
        top = np.array([0.55, 0.35, 0.45]) + rng.normal(0, 0.04, 3)
        bot = np.array([0.95, 0.60, 0.30]) + rng.normal(0, 0.05, 3)
    img = _v_gradient(rng, np.clip(top, 0, 1), np.clip(bot, 0, 1), n)
    _disc(img, rng.uniform(4, 20) * k, rng.uniform(3, 7) * k,
          rng.uniform(1.3, 2.2) * k, np.array([1.0, 0.97, 0.80]), glow=0.3, rng=rng, k=k)
    _horizon(img, int(rng.integers(18, 21) * k), np.array([0.45, 0.62, 0.35]),
             np.array([0.30, 0.45, 0.24]), rng=rng, k=k)
    dark = np.clip(np.array([0.15, 0.16, 0.20]) + rng.normal(0, 0.04, 3), 0, 1)
    for _ in range(int(rng.integers(3, 7) * k)):
        x = rng.uniform(2, 22) * k
        y = rng.uniform(2, 13) * k
        s = rng.uniform(0.8, 1.6) * k
        wing_w = max(0, int(0.45 * k))
        _stroke(img, x - s, y - s * 0.35, x, y + s * 0.15, dark, w=wing_w)
        _stroke(img, x, y + s * 0.15, x + s, y - s * 0.35, dark, w=wing_w)
    return np.clip(img, 0, 1)


def _balloons(rng, n):
    """سما صافية وبالونات ملونة بأقراص هالة وخيوط نازلة."""
    k = n / 24.0
    top = np.array([0.30, 0.55, 0.85]) + rng.normal(0, 0.04, 3)
    bot = np.array([0.68, 0.82, 0.96]) + rng.normal(0, 0.04, 3)
    img = _v_gradient(rng, np.clip(top, 0, 1), np.clip(bot, 0, 1), n)
    _horizon(img, int(rng.integers(19, 22) * k), np.array([0.45, 0.62, 0.35]),
             np.array([0.30, 0.45, 0.24]), rng=rng, k=k)
    palette = ((0.95, 0.30, 0.35), (0.30, 0.55, 0.95), (0.98, 0.75, 0.25),
               (0.35, 0.75, 0.35), (0.85, 0.40, 0.85), (0.95, 0.55, 0.20))
    for _ in range(int(rng.integers(3, 6) * k)):
        x = rng.uniform(3, 21) * k
        y = rng.uniform(3, 10) * k
        r = rng.uniform(1.0, 1.7) * k
        col = np.clip(np.array(palette[int(rng.integers(0, len(palette)))])
                      + rng.normal(0, 0.03, 3), 0, 1)
        _disc(img, x, y, r, col, glow=0.25, rng=rng, k=k)
        _stroke(img, x, y + r, x + rng.uniform(-0.8, 0.8) * k,
                min(n - 1.0, y + r + rng.uniform(2.5, 5.0) * k),
                np.array([0.85, 0.85, 0.85]), w=0, strength=0.5)
    return np.clip(img, 0, 1)


def _road(rng, n):
    """طريق إسفلت بمنظور: نقطة تلاشي قرب المنتصف + حواف بيضاء + خط منقط أصفر."""
    k = n / 24.0
    if rng.random() < 0.6:
        top = np.array([0.35, 0.55, 0.85]) + rng.normal(0, 0.04, 3)
        bot = np.array([0.70, 0.80, 0.92]) + rng.normal(0, 0.04, 3)
    else:
        top = np.array([0.40, 0.30, 0.50]) + rng.normal(0, 0.04, 3)
        bot = np.array([0.95, 0.65, 0.35]) + rng.normal(0, 0.05, 3)
    img = _v_gradient(rng, np.clip(top, 0, 1), np.clip(bot, 0, 1), n)
    row = int(rng.integers(11, 14) * k)
    _horizon(img, row, np.array([0.42, 0.60, 0.32]), np.array([0.28, 0.44, 0.20]),
             rng=rng, k=k)
    vx = rng.uniform(9, 15) * k
    vy = row + rng.uniform(0.5, 2.0) * k
    bot_w = rng.uniform(7.5, 11.0) * k
    road = np.clip(np.array([0.30, 0.30, 0.33]) + rng.normal(0, 0.03, 3), 0, 1)
    _fill_poly(img, [(vx - 1.2 * k, vy), (vx + 1.2 * k, vy),
                     (vx + bot_w / 2, n), (vx - bot_w / 2, n)], road, strength=0.85)
    edge = np.array([0.92, 0.92, 0.90])
    _stroke(img, vx - 1.2 * k, vy, vx - bot_w / 2, n - 1, edge,
            w=0.3 * k, strength=0.7)
    _stroke(img, vx + 1.2 * k, vy, vx + bot_w / 2, n - 1, edge,
            w=0.3 * k, strength=0.7)
    segs = int(rng.integers(3, 6))
    dash = np.clip(np.array([0.95, 0.90, 0.45]) + rng.normal(0, 0.03, 3), 0, 1)
    for i in range(segs):
        t0 = (i + rng.uniform(0.1, 0.35)) / segs
        t1 = min(1.0, t0 + rng.uniform(0.12, 0.2))
        y0 = vy + (n - vy) * t0
        y1 = vy + (n - vy) * t1
        hw0 = 1.2 + (bot_w / 2 - 1.2) * t0
        hw1 = 1.2 + (bot_w / 2 - 1.2) * t1
        w0 = hw0 * 0.12
        w1 = hw1 * 0.12
        _fill_poly(img, [(vx - w0, y0), (vx + w0, y0), (vx + w1, y1), (vx - w1, y1)],
                   dash, strength=0.85)
    return np.clip(img, 0, 1)


_GENERATORS = {
    "day": _day, "sunset": _sunset, "night": _night, "dawn": _dawn,
    "sea": _sea, "desert": _desert, "mountains": _mountains, "forest": _forest,
    "city": _city, "rain": _rain, "snow": _snow, "flowers": _flowers,
    "boat": _boat, "birds": _birds, "balloons": _balloons, "road": _road,
}


def make_sample(cid, seed, res=None):
    """عينة واحدة من صنف — نفس البذرة على نفس الدقة = نفس الصورة بالحرف
    (مضمون للاختبارات). res اختيارية لو محتاج ترسم على دقة تانية."""
    rng = np.random.default_rng(seed)
    return _GENERATORS[cid](rng, int(res or BASE))


def make_dataset(n_per_class=600, seed=1234):
    """الداتاسيت الكامل: (عدد الأصناف × n_per_class) عينة + اللابل — بالترتيب ثابت."""
    imgs, labels = [], []
    for i, cid in enumerate(CLASS_IDS):
        s = np.random.default_rng(seed + i)
        seeds = s.integers(0, 2**31 - 1, size=n_per_class)
        for sd in seeds:
            imgs.append(make_sample(cid, int(sd)))
            labels.append(i)
    return np.stack(imgs).astype(np.float32), np.array(labels, dtype=np.int64)


def to_model_range(img01):
    """[0,1] → [-1,1] (مدى التدريب) والعكس."""
    return img01 * 2.0 - 1.0
