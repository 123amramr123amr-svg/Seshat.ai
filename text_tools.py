#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""خ15 — الأدوات النصية المتقدمة لسِشات (text_tools)

أدوات محلية 100% تعمل بلا إنترنت وبلا أي مكتبة جديدة:
  1) تحويل الوحدات (طول/كتلة/حرارة/سرعة/حجم/بيانات/وقت)
  2) كتابة الأرقام بالحروف العربية (حتى 999,999,999)
  3) إحصاء أرقام (متوسط/وسيط/انحراف) بـnumpy
  4) تحليل نص (كلمات/جمل/وقت قراءة/أشهر الكلمات)
  5) تلخيص استخراجي عربي/إنجليزي (تردد الكلمات)
  6) فرق التواريخ ورقم اليوم في السنة
  7) توليد كلمات مرور قوية وقياس قوتها
  8) تحويل الأرقام الهندية ↔ الإنجليزية

المخرجات كلها نص عربي مباشر يرجع في الشات. مفيش صور خالص.
"""

import datetime
import math
import re
import secrets

try:
    import numpy as np
except Exception:  # numpy اختياري للإحصاء فقط
    np = None

# ---------------------------------------------------------------- أرقام بالحروف

_ONES = ["", "واحد", "اثنان", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة"]
_TEENS = ["عشرة", "أحد عشر", "اثنا عشر", "ثلاثة عشر", "أربعة عشر", "خمسة عشر",
          "ستة عشر", "سبعة عشر", "ثمانية عشر", "تسعة عشر"]
_TENS = ["", "", "عشرون", "ثلاثون", "أربعون", "خمسون", "ستون", "سبعون", "ثمانون", "تسعون"]
_HUNDREDS = ["", "مائة", "مائتان", "ثلاثمائة", "أربعمائة", "خمسمائة", "ستمائة", "سبعمائة", "ثمانمائة", "تسعمائة"]


def _three_digits_to_words(n):
    parts = []
    h = n // 100
    rest = n % 100
    if h:
        parts.append(_HUNDREDS[h])
    if rest >= 10 and rest < 20:
        parts.append(_TEENS[rest - 10])
    else:
        t, o = rest // 10, rest % 10
        if o:
            parts.append(_ONES[o])
        if t:
            parts.append(_TENS[t])
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    # «ثلاثة وأربعون» — الحالات المركبة من جزئين بس
    if len(parts) == 2 and parts[-1] in _TENS:
        return parts[0] + " و" + parts[1]
    return " و".join(parts)


def number_to_arabic_words(n):
    """رقم صحيح من 0 إلى 999,999,999 بالحروف العربية."""
    try:
        n = int(n)
    except Exception:
        return None
    if n < 0 or n > 999_999_999:
        return None
    if n == 0:
        return "صفر"
    parts = []
    millions = n // 1_000_000
    thousands = (n % 1_000_000) // 1000
    rest = n % 1000
    if millions:
        if millions == 1:
            parts.append("مليون")
        elif millions == 2:
            parts.append("مليونان")
        else:
            parts.append(_three_digits_to_words(millions) + " مليون")
    if thousands:
        if thousands == 1:
            parts.append("ألف")
        elif thousands == 2:
            parts.append("ألفان")
        elif 3 <= thousands <= 10:
            parts.append(_three_digits_to_words(thousands) + " آلاف")
        else:
            parts.append(_three_digits_to_words(thousands) + " ألف")
    if rest:
        parts.append(_three_digits_to_words(rest))
    return " و".join(p for p in parts if p)


# ---------------------------------------------------------------- الوحدات

_LENGTH = {"كيلومتر": 1000.0, "كم": 1000.0, "km": 1000.0, "متر": 1.0, "م": 1.0, "m": 1.0,
           "سم": 0.01, "سنتيمتر": 0.01, "cm": 0.01, "مم": 0.001, "mm": 0.001,
           "ميل": 1609.344, "mile": 1609.344, "miles": 1609.344,
           "قدم": 0.3048, "قدم Feet": 0.3048, "قدم feet": 0.3048, "feet": 0.3048, "foot": 0.3048, "ft": 0.3048,
           "بوصة": 0.0254, "inch": 0.0254, "inches": 0.0254, "yard": 0.9144, "ياردة": 0.9144}
_MASS = {"كيلوجرام": 1.0, "كيلو": 1.0, "كجم": 1.0, "kg": 1.0, "جرام": 0.001, "جم": 0.001, "g": 0.001, "gram": 0.001,
         "طن": 1000.0, "ton": 1000.0, "tonne": 1000.0, "رطل": 0.4536, "lb": 0.4536, "pound": 0.4536,
         "أوقية": 0.02835, "ounce": 0.02835, "oz": 0.02835}
_VOLUME = {"لتر": 1.0, "ل": 1.0, "liter": 1.0, "litre": 1.0, "l": 1.0, "مل": 0.001, "ملليلتر": 0.001, "ml": 0.001,
           "جالون": 3.78541, "gallon": 3.78541, "كوب": 0.24, "cup": 0.24}
_DATA = {"كيلوبايت": 1024.0, "kb": 1024.0, "ميجابايت": 1024.0 ** 2, "mb": 1024.0 ** 2,
         "جيجابايت": 1024.0 ** 3, "gb": 1024.0 ** 3, "تيرابايت": 1024.0 ** 4, "tb": 1024.0 ** 4,
         "بايت": 1.0, "byte": 1.0}
_SPEED = {"m/s": 1.0, "متر/ثانية": 1.0, "كم/ساعة": 1000.0 / 3600.0, "km/h": 1000.0 / 3600.0,
          "kph": 1000.0 / 3600.0, "mph": 1609.344 / 3600.0, "ميل/ساعة": 1609.344 / 3600.0}
_TIME = {"ثانية": 1.0, "ث": 1.0, "second": 1.0, "s": 1.0, "دقيقة": 60.0, "minute": 60.0,
         "ساعة": 3600.0, "hour": 3600.0, "h": 3600.0, "يوم": 86400.0, "day": 86400.0}
_TEMPS = ["سيلسيوس", "مئوية", "celsius", "c", "فهرنهايت", "fahrenheit", "f", "كلفن", "kelvin", "k"]
_FAMILIES = [(_LENGTH, "الطول", "متر"), (_MASS, "الكتلة", "كيلوجرام"), (_VOLUME, "الحجم", "لتر"),
             (_DATA, "البيانات", "بايت"), (_SPEED, "السرعة", "متر/ثانية"), (_TIME, "الوقت", "ثانية")]

_NUM_RE = r"(\d+(?:[.,]\d+)?)"
_SEP = r"\s*(?:الى|إلى|على|في|بـ|كـ|to|in|as)\s*"


def _find_units(text):
    """يرجع [(اسم، عائلة، معامل)] لكل الوحدات الموجودة في النص مرتبة بالموضع،
    مع استبعاد التطابقات اللي جوه كلمة أطول (متر جوه كيلومتر مثلًا)."""
    low = text.lower()
    guard = r"(?<![a-zA-Z\u0621-\u0652\u0670\u0640])" + "%s" + r"(?![a-zA-Z\u0621-\u0652\u0670\u0640])"
    found = []
    for table, family, base in _FAMILIES:
        for name in table:
            m = re.search(guard % re.escape(name), low)
            if m:
                found.append((name, family, table[name], m.start(), m.end()))
    for t in _TEMPS:
        m = re.search(guard % re.escape(t), low)
        if m:
            found.append((t, "الحرارة", None, m.start(), m.end()))
    out = []
    # الأطول أولًا، وبعدين بالموضع — ومنع التكرار لنفس الوحدة أو تطابق جوه تطابق أطول
    for name, family, factor, s, e in sorted(found, key=lambda x: (-(x[4] - x[3]), x[3])):
        if any((N, F) == (name, family) for N, F, K, S, E in out):
            continue
        if any(s >= S and e <= E for N, F, K, S, E in out):
            continue
        out.append((name, family, factor, s, e))
    out.sort(key=lambda x: x[3])
    return [(n, f, k) for n, f, k, s, e in out]


def convert_units(text):
    """يحاول تحويل رقم + وحدة إلى وحدة تانية في نفس العائلة."""
    low = text.lower()
    m = re.search(_NUM_RE + r"\s*([^\s]+)", low)
    if not m:
        return None
    val = float(m.group(1).replace(",", ""))
    units = _find_units(low)
    if len(units) < 2:
        return None
    # ترتيب الوحدات حسب ظهورها في النص
    units.sort(key=lambda u: low.find(u[0]))
    (n1, f1, k1), (n2, f2, k2) = units[0], units[1]
    if f1 != f2 or f1 == "الحرارة":
        if f1 == "الحرارة" and f2 == "الحرارة":
            def to_c(x, u):
                u = u.lower()
                if u in ("فهرنهايت", "fahrenheit", "f"):
                    return (x - 32.0) * 5.0 / 9.0
                if u in ("كلفن", "kelvin", "k"):
                    return x - 273.15
                return x

            def from_c(x, u):
                u = u.lower()
                if u in ("فهرنهايت", "fahrenheit", "f"):
                    return x * 9.0 / 5.0 + 32.0
                if u in ("كلفن", "kelvin", "k"):
                    return x + 273.15
                return x
            c = to_c(val, n1)
            res = from_c(c, n2)
            return "%g %s = %g %s" % (val, n1, res, n2)
        return None
    res = val * k1 / k2
    return "%g %s = %g %s" % (val, n1, res, n2)


# ---------------------------------------------------------------- إحصاء وتحليل

_AR_STOP = set("في من على الى إلى عن مع هذا هذه ذلك تلك التي الذي الذين ما لا لم لن هل كيف اين أين متى كم كان كانت يكون تكون هو هي هم انا أنا انت أنت لقد قد ثم أو أو او و كل بعض بين بعد قبل عند لكن لا حتى إذا اذا ايضا أيضا كما مثل غير فيه فيه منها منه له لها هناك هنالك نعم يا با بـ لل ولم كذلك نفس شيء يمكن ينبغ ينبغي يجب حول بينما عندما لماذا إذن حسنا طيب جدا".split())
_EN_STOP = set("the a an and or of to in on at for with by is are was were be been being it its this that these those as from but if then than so such can could will would should may might not no nor do does did doing have has had i you he she we they my your our their there here what which who whom when where why how all any both each few more most other some only own same too very just also about into over after before between during through under again further once".split())


def _tokens(text):
    return [w for w in re.findall(r"[\w\u0621-\u064A\u0660-\u0669]+", text.lower()) if len(w) > 1]


def text_stats(text):
    words = _tokens(text)
    sentences = [s for s in re.split(r"[.!؟?\n]+", text) if s.strip()]
    uniq = set(words)
    stop = _AR_STOP | _EN_STOP
    freq = {}
    for w in words:
        if w in stop:
            continue
        freq[w] = freq.get(w, 0) + 1
    top = sorted(freq.items(), key=lambda x: -x[1])[:5]
    minutes = len(words) / 200.0 if words else 0.0
    lines = [
        "تحليل النص:",
        "- عدد الأحرف: %d" % len(text),
        "- عدد الكلمات: %d" % len(words),
        "- عدد الجمل: %d" % len(sentences),
        "- كلمات فريدة: %d" % len(uniq),
        "- وقت القراءة التقريبي: %s" % ("أقل من دقيقة" if minutes < 1 else "%.1f دقيقة" % minutes),
    ]
    if top:
        lines.append("- أشهر الكلمات: " + "، ".join("%s (%d)" % (w, c) for w, c in top))
    return "\n".join(lines)


def summarize(text, k=3):
    sents = [s.strip() for s in re.split(r"[.!؟?\n]+", text) if len(s.strip()) > 15]
    if len(sents) <= k:
        return None
    stop = _AR_STOP | _EN_STOP
    freq = {}
    for w in _tokens(text):
        if w not in stop:
            freq[w] = freq.get(w, 0) + 1
    scored = []
    for i, s in enumerate(sents):
        ws = _tokens(s)
        if not ws:
            continue
        score = sum(freq.get(w, 0) for w in ws) / (len(ws) + 1.0)
        scored.append((score, i, s))
    top = sorted(sorted(scored, key=lambda x: -x[0])[:k], key=lambda x: x[1])
    return "الملخص (%d جمل من %d):\n" % (len(top), len(sents)) + "\n".join("- " + s for _, _, s in top)


def num_stats(nums):
    if np is None or len(nums) < 2:
        return None
    a = np.array(nums, dtype=float)
    return ("إحصاء الأرقام (%d قيمة):\n"
            "- المتوسط: %.4g\n- الوسيط: %.4g\n- الانحراف المعياري: %.4g\n"
            "- الأصغر: %.4g\n- الأكبر: %.4g\n- المجموع: %.4g") % (
        len(a), a.mean(), float(np.median(a)), a.std(), a.min(), a.max(), a.sum())


# ---------------------------------------------------------------- تواريخ وكلمات مرور

_DATE_RE = re.compile(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})")


def date_diff(d1, d2):
    a = datetime.date(int(d1[0]), int(d1[1]), int(d1[2]))
    b = datetime.date(int(d2[0]), int(d2[1]), int(d2[2]))
    days = (b - a).days
    return "الفرق بين %s و %s = %d يوم (%.1f أسبوع تقريبًا)" % (
        a.isoformat(), b.isoformat(), abs(days), abs(days) / 7.0)


def day_of_year(d):
    dt = datetime.date(int(d[0]), int(d[1]), int(d[2]))
    y = dt.timetuple().tm_yday
    return "%s هو اليوم رقم %d من سنة %d، وتبقى %d يوم على آخرها" % (
        dt.isoformat(), y, dt.year, (366 if dt.year % 4 == 0 and (dt.year % 100 != 0 or dt.year % 400 == 0) else 365) - y)


_PW_LOWER = "abcdefghijkmnopqrstuvwxyz"
_PW_UPPER = "ABCDEFGHJKLMNPQRSTUVWXYZ"
_PW_DIGIT = "23456789"
_PW_SYMBOL = "!@#$%&*+-=?"


def gen_password(length=14, symbols=True):
    length = max(8, min(64, int(length)))
    pools = [_PW_LOWER, _PW_UPPER, _PW_DIGIT]
    if symbols:
        pools.append(_PW_SYMBOL)
    out = [secrets.choice(p) for p in pools]
    allp = "".join(pools)
    out += [secrets.choice(allp) for _ in range(length - len(out))]
    secrets.SystemRandom().shuffle(out)
    return "".join(out)


def password_strength(pw):
    score = 0
    score += min(40, len(pw) * 3)
    classes = 0
    for pattern, pts in ((r"[a-z]", 15), (r"[A-Z]", 15), (r"\d", 15), (r"[^A-Za-z0-9]", 15)):
        if re.search(pattern, pw):
            classes += 1
            score += pts
    if len(pw) >= 12 and classes >= 3:
        verdict = "قوية جدًا"
    elif classes >= 3 and len(pw) >= 10:
        verdict = "قوية"
    elif classes >= 2 and len(pw) >= 8:
        verdict = "متوسطة"
    else:
        verdict = "ضعيفة — زوّد الطول والأنواع"
    return "قوة كلمة المرور: %d/100 — %s (الطول %d، أنواع المحارف %d)" % (
        min(100, score), verdict, len(pw), classes)


_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def digits_convert(text):
    has_ar = any(ch in "٠١٢٣٤٥٦٧٨٩" for ch in text)
    if has_ar:
        converted = text.translate(_AR_DIGITS)
        return "الأرقام بعد التحويل: %s" % converted
    converted = text.translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))
    return "الأرقام بعد التحويل: %s" % converted


# ---------------------------------------------------------------- الموزع

def try_text_tool(message):
    """يرجع (الرد، التفكير) لو الرسالة مطابقة أداة، أو None."""
    msg = (message or "").strip()
    if not msg:
        return None
    low = msg.lower()

    # 7) تواريخ
    dates = _DATE_RE.findall(msg)
    if len(dates) == 2 and re.search(r"الفرق|مدة|بين|between|من .* الى ", low):
        return date_diff(dates[0], dates[1]), "حسبت الفرق بين التاريخين بالتقويم"
    if len(dates) == 1 and re.search(r"رقم كام|اليوم رقم|day of year", low):
        return day_of_year(dates[0]), "حسبت رقم اليوم في السنة"

    # 1) كلمة مرور
    if re.search(r"كلمة مرور|كلمة السر|password", low) and not re.search(r"قوة|قوية|strength", low):
        n = re.search(r"(\d{1,2})", msg)
        pw = gen_password(int(n.group(1)) if n else 14)
        return "كلمة مرور قوية جديدة (%d محرف):\n%s\nمتولدة محليًا ولم تُحفظ في أي مكان." % (
            len(pw), pw), "ولدت كلمة مرور عشوائية آمنة محليًا"
    if re.search(r"قوة|قوية|strength", low) and "كلمة" in low:
        cand = re.search(r"[A-Za-z0-9!@#$%&*+\-=?_.]{8,}", msg)
        if cand:
            return password_strength(cand.group(0)), "قست قوة كلمة المرور بالطول والأنواع"

    # 2) وحدات
    if re.search(_NUM_RE, low) and re.search(r"حوّل|حول|convert|يساوي كم|كم يساوي", low):
        r = convert_units(low)
        if r:
            return r, "حولت الوحدات بمعاملات دقيقة"

    # 3) رقم بالحروف
    if re.search(r"بالحروف|باللغة العربية|كلمة", low) and re.fullmatch(
            r"[^\d]*\d{1,9}[^\d]*", msg.replace("،", "")) and re.search(r"بالحروف", low):
        m = re.search(r"(\d{1,9})", msg)
        w = number_to_arabic_words(m.group(1))
        if w:
            return "الرقم %s بالحروف العربية: %s" % (m.group(1), w), "حوّلت الرقم لحروف عربية"

    # 4) إحصاء أرقام
    if re.search(r"متوسط|وسيط|median|انحراف|المدى|إحصاء|احصاء|mean|std", low):
        nums = [float(x.replace(",", ".")) for x in re.findall(r"(\d+(?:[.,]\d+)?)", msg)]
        if len(nums) >= 3:
            r = num_stats(nums)
            if r:
                return r, "حسبت المتوسط والوسيط والانحراف بـnumpy"

    # 5) تلخيص
    if re.search(r"لخص|لخّص|الملخص|summarize|تلخيص", low) and len(msg) > 250:
        body = re.sub(r"^.*(لخص|لخّص|الملخص|summarize|تلخيص)\s*(النص|التالي|ده|ده:|:)?\s*", "", msg, count=1)
        r = summarize(body if len(body) > 200 else msg)
        if r:
            return r, "لخصت النص بترجيح جمل التردد العالي (تلخيص استخراجي)"

    # 6) تحليل نص
    if re.search(r"حلل النص|تحليل النص|عدد الكلمات|كم كلمة|كام كلمة|وقت القراءة|text stats|analyze text", low):
        body = re.sub(r"^.*(حلل النص|تحليل النص|عدد الكلمات|كم كلمة|كام كلمة|وقت القراءة)\s*(ده|التالي|:)?\s*", "", msg, count=1)
        return text_stats(body if body.strip() else msg), "حللت النص إحصائيًا"

    # 8) أرقام هندية/إنجليزية
    if re.search(r"حوّل الأرقام|الأرقام الهندية|arabic digits|convert digits", low):
        return digits_convert(re.sub(r"^.*(?:حوّل الأرقام|الأرقام الهندية|convert digits)\s*(في|ده|:)?\s*", "", msg, count=1)), "حوّلت شكل الأرقام"

    return None
