"""
seshat.chet — المخ الجديد (brain.py)
====================================

طبقة ذكية بتشتغل فوق النموذج المحلي، بتخلي seshat يبقى مساعد حقيقي:

  1. أدب مع أي لفظ خارج — أي شتم بأي صيغة (حتى بالأرقام زي "ا7مق" أو
     بالإنجليزي) بياخد رد مؤدب بدل ما النموذج يتلخبط فيه.
  2. بحث في الإنترنت — ويكيبيديا و DuckDuckGo من غير أي مفتاح: سؤال برّه
     اللي اتعلمه بيتجاوب من النت بدل "لسه ماتعلمتش".
  3. تعلم ذاتي — أي معلومة جديدة بيجاوب عليها (من النت أو من المخ الخارجي)
     بتتخزن في data/learned.json، والمرة اللي جاية بيرد عليها من الذاكرة
     على طول. وكمان يقدر تعلمه بيدويًا بصيغتين:
         علمني: س: اللعبة المفضلة لعمرو ج: الشطرنج
         علمني: ايه اللعبة المفضلة لعمرو؟ الشطرنج   (صيغة مرنة: سؤال؟ جواب)
  4. الصور — "ارسم قطة بتبص للقمر" → صورة واقعية من مولّد صور خارجي (imagegen.py)، وتعديلها
     بالوصف. "ارسم بالمربعات …" → النموذج المحلي بيرسم بنفسه (engine.draw).
  5. مخ خارجي اختياري — لو ضفت مفتاح API متوافق مع OpenAI (زرار الإعدادات
     ⚙ في الواجهة أو متغيرات البيئة)، الردود بتشتغل بشبكة عصبية ضخمة بترد
     على أي سؤال في الدنيا — و seshat لسه بيحافظ على هويته (seshat من
     تطوير عمرو أحمد).

النموذج المحلي (engine.py) لسه هو البداية: الهوية، الحساب، الأكواد، وكل
اللي اتعلمه — والمخ ده بيشتغل بس لما النموذج مايعرفش، فمفيش حاجة اتكسرت.
"""

import difflib
import json
import os
import random
import re
import time
from urllib.parse import parse_qs, quote, unquote, urlparse

import requests

from engine import extract_question

HERE = os.path.dirname(os.path.abspath(__file__))
LEARNED_PATH = os.path.join(HERE, "data", "learned.json")
CONFIG_PATH = os.path.join(HERE, "data", "config.local.json")

_UA = {"User-Agent":
       "seshat.chet/1.0 (https://github.com/123amramr123amr-svg/Seshat.ai)"}
_NET_TIMEOUT = 12
LEARNED_MAX = 4000
LEARNED_Q_MAX = 1200      # أقصى طول للسؤال المتعلم (بيتبع سياق الإدخال الجديد)

SOURCE_LABELS = {
    "memory": "ذاكرة",
    "memory+model": "ذاكرة + نموذج",
    "model": "نموذج محلي",
    "none": "",
}


# ============================================================
# الإعدادات: مفتاح API اختياري + البحث في النت
# ============================================================
class BrainConfig:
    """بيقرا من متغيرات البيئة الأول، ولو فيه ملف إعدادات محلي بيحترمه."""

    def __init__(self):
        self.api_key = os.environ.get("SESHAT_API_KEY", "")
        self.api_base = os.environ.get("SESHAT_API_BASE", "https://api.openai.com/v1")
        self.api_model = os.environ.get("SESHAT_API_MODEL", "gpt-4o-mini")
        self.search_enabled = os.environ.get("SESHAT_SEARCH", "1") != "0"
        # الصور الواقعية بتيجي من مولّد صور خارجي (شوف imagegen.py). SESHAT_EXTERNAL_IMAGES=0
        # بيقفله، وساعتها "ارسم" بيرجع لرسم seshat بالمربعات بس.
        self.external_images = os.environ.get("SESHAT_EXTERNAL_IMAGES", "1") != "0"
        # مفتاح مجاني من خدمة الصور: بيشيل اللوجو وحد الـ 45 ثانية (اختياري)
        self.image_key = os.environ.get("SESHAT_IMAGE_KEY", "").strip()
        self.reload()

    def reload(self):
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                saved = json.load(f)
        except Exception:
            return
        for k in ("api_key", "api_base", "api_model"):
            v = saved.get(k)
            if isinstance(v, str) and v.strip():
                setattr(self, k, v.strip())
        if isinstance(saved.get("search_enabled"), bool):
            self.search_enabled = saved["search_enabled"]

    def save(self, **kw):
        saved = {}
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                saved = json.load(f)
        except Exception:
            pass
        for k in ("api_key", "api_base", "api_model", "search_enabled"):
            if k in kw and kw[k] is not None:
                saved[k] = kw[k]
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        tmp = CONFIG_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(saved, f, ensure_ascii=False, indent=1)
        os.replace(tmp, CONFIG_PATH)
        self.reload()

    def public(self):
        """حالة المخ للواجهة — المفتاح نفسه مش بيتعرض أبدًا."""
        return {"api": bool(self.api_key),
                "api_model": self.api_model if self.api_key else "",
                "api_base": self.api_base,
                "search": self.search_enabled}


# ============================================================
# أدب مع أي لفظ خارج
# ============================================================
# الناس بتكتب الشتم بأشكال كتير: "ا7مق" أو "a7bal" — فبنوحد الحروف الأول
# وبنحول الأرقام اللي بتمثل حروف (3 = ع، 7 = ح…) لحروف عادية.
_LEET = str.maketrans({"3": "ع", "5": "خ", "7": "ح", "2": "ا", "6": "ط",
                       "8": "ق", "9": "ص", "٤": "ع", "٥": "خ", "٧": "ح",
                       "٢": "ا", "٦": "ط", "٩": "ص"})

# كلمات قصيرة (٣ حروف أو أقل): لازم تكون كلمة لوحدها عشان ما نصطادش
# كلمات عادية زي "مسكن" و"الزبون".
_INSULT_EXACT = {
    "احا", "خرا", "خره", "زب", "كس", "طيز", "عير", "زق", "تف", "تفو",
    "زفت", "نوس", "وصخ", "عرص", "خول", "ردء",
}

# كلمات طويلة: بتجيك كجزء من أي كلمة (زي "احمقان" و"شمتوه" و"متناكه").
_INSULT_STEMS = (
    "احمق", "اهبل", "ابله", "غبي", "غباء", "تافه", "حقير", "سافل", "واطي",
    "قذر", "خنزير", "منحرف", "فاجر", "خاين", "متخلف", "مغفل", "نشال",
    "شرموط", "شرمود", "قحبه", "منيوك", "متناك", "تنيك", "ينيك", "نايك",
    "زوابط", "بيضان", "دلاخ", "جموط", "صايع", "ردعي", "قواد", "مخنث",
    "زنان", "عارض", "ميس", "زول", "تفو", "استهبال", "تخنيق",
)

# شتم لاتيني (حروف إنجليزي + عربي بالحروف اللاتيني)
_LATIN_INSULTS = {
    "ahbal", "ahabal", "ghabi", "ghabiya", "kalb", "kalba", "hmar",
    "himar", "himaar", "khara", "kharra", "kus", "koss", "kusok", "zib",
    "zabb", "ayr", "sharmut", "sharmuta", "gahba", "qahba", "menyook",
    "manyouk", "metnak", "tayz", "taz", "behima", "ars", "arss", "nos",
    "zafet", "wasekh", "ks", "kes", "emak", "oxmak", "zeby", "a7r",
    "stupid", "idiot", "dumb", "dumbass", "moron",
    "fool", "foolish", "loser", "trash", "garbage", "jerk", "bastard",
    "bitch", "shit", "shitty", "fuck", "fucking", "fucked", "asshole",
    "wanker", "retard", "retarded", "kys",
}

# تركيبات (شتم مركب — ابن كلب، كس اختك، خرب بيتك…)
_INSULT_PATTERNS = (
    r"ابن\s*ال?(كلب|كلبه|خرا|خره|متناكه|شرموطه|قحبه|مجانيه|حيوانه|وسخه|لعيونه|مرا|مره|فرخه|دعاذه|دعازه)",
    r"(يلعن|اللعنه|لعنه|نيل)\s*.{0,12}(ابوك|امك|ابوكو|امكو|العائله|الشجره|الدين|دماغك)",
    r"الله\s*ي?لعنك",
    r"تبا\s*لك",
    r"خرب\s*(في|علي|على|بيت|بيتك|دماغك|دينك)",
    r"تولع\s*في",
    r"كس\s*(ام|اخت)",
    r"(زب|عير|طيز|خرا|خره)\s*في(ك|كم|يك)",
    r"افتح\s*(كسك|طيزك)",
    r"\bf(u|v|ph)?ck",
    r"\bsh[i1!]t",
    r"\bb[i1!]tch",
    r"\bass\s*hole",
    r"\byou\s+(suck|stupid|idiot|dumb|loser|trash|garbage|clown)",
    r"\bscrew\s+you",
    r"\bf\s*off\b",
    r"\bi\s+hate\s+you",
    r"\bklb\b",
)

# مقارنة الحيوان لشخص بتاعت اتجاه: "انت كلب" شتم، بس "ارسم لي كلب" مش شتم
# فبنشترط وجود "انت / يا / you" قبلها أو جنبها.
_ANIMAL_RE = re.compile(
    r"(انت|انتو|انتم|يا|you)\s+(?:يا\s+)?"
    r"(كلب|كلبه|حمار|حيوان|خنزير|قرد|جحش|جاحش|بهيم|بهيمه|وصخ|نوس)")


def _curse_norm(text):
    t = text.lower()
    t = re.sub(r"[\u064B-\u0652\u0670\u0640]", "", t)          # تشكيل وتطويل
    t = re.sub("[أإآٱ]", "ا", t)
    t = t.replace("ى", "ي").replace("ة", "ه")
    t = t.replace("ؤ", "و").replace("ئ", "ي")
    t = re.sub(r"([^\W\d_])\1{2,}", r"\1", t)                   # حرف متكرر
    t = re.sub(r"[^\w\s\u0600-\u06FF]", " ", t)                 # من غير رموز
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def is_insult(message):
    t = _curse_norm(message)
    variants = [t, t.translate(_LEET)]
    for v in variants:
        if not v:
            continue
        for w in v.split():
            if w in _INSULT_EXACT or w in _LATIN_INSULTS:
                return True
            for stem in _INSULT_STEMS:
                if stem in w:
                    return True
        joined = " " + v + " "
        for pat in _INSULT_PATTERNS:
            if re.search(pat, joined):
                return True
        if _ANIMAL_RE.search(v):
            return True
    return False


POLITE_AR = [
    "ربنا يسامحك 🙂 معلش عدت خاطري — تحب نتكلم في حاجة تانية؟",
    "ربنا يسامحك، أنا مش بزعل أبدًا ❤️ اسألني أي حاجة.",
    "عادي جدًا — ربنا يسامحك ويعافيك. أنا هنا عشان أساعدك.",
    "ربنا يسامحك 😄 مفيش مشكلة خالص، قولي محتاج إيه؟",
    "ولا يهمك — ربنا يسامحك. يلا نكمل في اللي يفيدك.",
]
POLITE_EN = [
    "May God forgive you 🙂 I don't mind at all — how can I help?",
    "It's okay, May God forgive you. Ask me anything you want.",
    "No hard feelings — May God forgive you. What do you need?",
]
APOLOGY_AR = [
    "ولا يهمك أبدًا ❤️ ربنا يسامح ويكرم.",
    "ولا يهمك خالص، أنا مبسوط إنك اسألت 🙂",
]
APOLOGY_EN = [
    "No worries at all! I'm glad you asked 🙂",
]

_APOLOGY_RE = re.compile(
    r"^(اسف|آسف|اسفين|معلش|معليش|مش\s*قصدي|سامحني|sorry|my\s*bad)\b", re.I)


def _is_arabic_text(text):
    return any("\u0600" <= ch <= "\u06FF" for ch in text)


# ============================================================
# تعلم ذاتي — ذاكرة data/learned.json
# ============================================================
STOP_WORDS = {
    "مين", "ايه", "إيه", "ازاي", "إزاي", "اين", "أين", "امتى", "إمتى",
    "ليه", "لماذا", "في", "من", "عن", "على", "هو", "هي", "هل", "الى",
    "the", "what", "who", "how", "where", "when", "why", "is", "are",
    "was", "were", "a", "an", "of", "in", "on", "to", "do", "does",
    "did", "and", "or", "for", "with", "about",
}


def _sig_words(q):
    """الكلمات المهمة (اللي بتفرق سؤال عن سؤال) — بنستخدمها في مراجعة التشابه"""
    return [w for w in q.split() if len(w) >= 4 and w not in STOP_WORDS]


class LearnedMemory:
    """كل حاجة seshat بيتعلمها بتتخزن هنا وترجع تتجاب لما تشابهها يجي عالي."""

    def __init__(self, path=LEARNED_PATH):
        self.path = path
        self.items = []
        self.exact = {}
        self._load()

    def _load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            self.items = data if isinstance(data, list) else []
            self.items = [it for it in self.items
                          if isinstance(it, dict) and it.get("q") and it.get("a")]
            self.items = self.items[-LEARNED_MAX:]
        except Exception:
            self.items = []
        self._index()

    def _index(self):
        self.exact = {it["q"]: it for it in self.items}

    def add(self, question, answer, src="user"):
        from tokenizer import normalize_question
        q = normalize_question(question)[:LEARNED_Q_MAX]
        a = (answer or "").strip()
        if not q or not a:
            return False
        if len(a) > 4000:
            a = a[:4000]
        self.items = [it for it in self.items if it.get("q") != q]
        self.items.append({"q": q, "a": a, "src": src,
                           "ts": time.strftime("%Y-%m-%d %H:%M")})
        self._save()
        self._index()
        return True

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.items[-LEARNED_MAX:], f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)

    def lookup(self, message):
        """
        أقرب سؤال اتعلم: بالحرف، أو تشابه كبير (٦٢٪+) مع شرط إن الكلمات
        المهمة في سؤال المستخدم تكون موجودة في السؤال المتعلم — عشان
        "مين اكتشف امريكا" ما تختلطش مع "مين اكتشف بنزين".
        ولو كل كلمات السؤال المتعلم موجودة في سؤال المستخدم، ده نفس السؤال
        بكلام زيادة ("لعبة عمرو" جوه "لعبة عمرو المفضلة ايه") — بيتقبل حتى
        لو التشابه النسبي قل لأن الرسالة أطول.
        """
        from tokenizer import normalize_question
        q = normalize_question(message)[:LEARNED_Q_MAX]
        if not q or not self.items:
            return None
        hit = self.exact.get(q)
        if hit:
            return hit
        sig = _sig_words(q)
        best, best_ratio = None, 0.0
        for it in self.items:
            r = difflib.SequenceMatcher(None, q, it["q"]).ratio()
            if r > best_ratio:
                best, best_ratio = it, r
        if best is None:
            return None
        bs = _sig_words(best["q"])
        # قاعدة الاحتواء: كل كلمات السؤال المتعلم (اتنين أو أكتر) موجودة
        # في سؤال المستخدم → نفس السؤال
        if bs and sig and len(bs) >= 2 and all(
                any(difflib.SequenceMatcher(None, w, b).ratio() >= 0.8
                    for b in sig) for w in bs):
            return best
        if best_ratio < 0.62:
            return None
        if not bs or not sig:
            return best
        matched = sum(1 for w in sig
                      if any(difflib.SequenceMatcher(None, w, b).ratio() >= 0.8
                             for b in bs))
        # من غير التساوي: نص الكلمات بالظبط مش كفاية ("مين اكتشف بنزين" ما
        # يتاخدش على إنه "مين اكتشف امريكا")
        if matched / len(sig) > 0.5:
            return best
        return None

    def forget(self, message):
        """بيمسح سؤال اتعلم — بالحرف أو أقرب واحد شبهه كفاية (٧٥٪+ عشان
        ما يمسحش سؤال تاني بالغلط). بيرجع السؤال الممسوح أو None."""
        from tokenizer import normalize_question
        q = normalize_question(message)[:LEARNED_Q_MAX]
        if not q:
            return None
        if q in self.exact:
            target = q
        else:
            best, best_ratio = None, 0.0
            for it in self.items:
                r = difflib.SequenceMatcher(None, q, it["q"]).ratio()
                if r > best_ratio:
                    best, best_ratio = it, r
            target = best["q"] if best and best_ratio >= 0.75 else None
        if not target:
            return None
        self.items = [it for it in self.items if it["q"] != target]
        self._save()
        self._index()
        return target


_FORGET_RE = re.compile(r"^\s*(انسى|انسي|forget)\s*[:،]?\s*(.+)$", re.I)


def parse_forget(message):
    m = _FORGET_RE.match(message)
    if not m:
        return None
    q = m.group(2).strip(" -–—؟?")
    return q or None


_TEACH_RE = re.compile(r"^\s*(علمني|احفظ\s*لي|احفظ|تذكر|teach)\s*[:،]?\s*(.*)$",
                       re.I | re.S)   # re.S: التعليم المرن ممكن يجي على سطرين
_QA_RE = re.compile(
    r"^\s*(?:س:|السؤال:|سؤال:)?\s*(.+?)\s*(?:ج:|ج\s*:|الجواب:|جواب:|جواب\s*:|a:|answer:)\s*(.+)$",
    re.I | re.S)
# الناس مش دايمًا بتحط س:/ج: — «علمني: ايه عاصمة تشاد؟ هي ندجامنا» صيغة طبيعية.
_FLEX_ANS_PRONOUN_RE = re.compile(r"^(?:هي|هو|هما)\s+")
_QWORD_RE = re.compile(
    r"(?:^|\s)(ايه|اى|إيه|مين|من|فين|امتى|إمتى|ليه|لية|ازاي|إزاي|كام|كم|هل|اين|وين"
    r"|who|what|where|when|why|how|which)\b", re.I)


def _flex_teach(payload):
    """صيغة تعليم مرنة من غير س:/ج: — «سؤال؟ جواب».
    السؤال بيتحدد بعلامة الاستفهام (أول ؟/?) والباقي بيبقى الجواب،
    ومفيش علامة استفهام بنتقبل newline أو شرطة بس لو الأول فيه كلمة سؤال.
    شروط الأمان: الجواب مش فاضي ومش طويل (٤٠٠ حرف أقصى) عشان ما نلخبطش
    رسالة سؤال عادي بتعليم."""
    m = re.search(r"[؟?]", payload)
    if m:
        q = payload[:m.end()].strip(" \n-–—")
        a = payload[m.end():].strip(" \n-–—،:؟?")
    else:
        parts = re.split(r"\n+|\s+[-–—]\s+", payload, maxsplit=1)
        if len(parts) != 2:
            return None
        q, a = parts[0].strip(" \n-–—"), parts[1].strip(" \n-–—،:")
        if not _QWORD_RE.search(q):
            return None
    a = _FLEX_ANS_PRONOUN_RE.sub("", a)
    if len(q) < 4 or not a or len(a) > 400:
        return None
    return q, a


def parse_teach(message):
    m = _TEACH_RE.match(message)
    if not m:
        return None
    qa = _QA_RE.match(m.group(2).strip())
    if not qa:
        return _flex_teach(m.group(2).strip())    # صيغة مرنة: «سؤال؟ جواب»
    q = qa.group(1).strip(" -–—")
    a = qa.group(2).strip(" -–—")
    if q and a:
        return q, a
    return None


# ============================================================
# حساب مباشر — مضمون ١٠٠٪ حتى للأرقام الكبيرة والكسور
# ============================================================
_MATH_ONE = re.compile(
    r"^\D*?(-?\d+(?:[.,]\d+)?)\s*([+\-*x×÷/])\s*(-?\d+(?:[.,]\d+)?)\D*$")

# الحساب بالكلمات العربي: «احسب 17 في 24» / «15 على 3» / «8 زائد 2» / «10 ناقص 4»
# الشرط: المسألة رقم + كلمة العملية + رقم من غير كلام كتير حواليها،
# عشان «في» العادي في الجمل (زي «في 2024 حصلت حاجة») ما تتحسبش غلط.
_MATH_WORDS = re.compile(
    r"^\s*\D*?(-?\d+(?:[.,]\d+)?)\s*"
    r"(مضروب\s*في|تقسيم\s*على|مقسوم\s*على|زائد|ناقص|ضرب|في|على)\s*"
    r"(-?\d+(?:[.,]\d+)?)\s*\D*$")
_MATH_WORD_SYM = {"مضروب في": "×", "تقسيم على": "÷", "مقسوم على": "÷",
                  "ضرب": "×", "في": "×", "زائد": "+", "ناقص": "-", "على": "÷"}


def _math_word_sym(sym):
    """بتوحّد المسافات جوه كلمة العملية (مضروب  في → مضروب في)."""
    return " ".join(sym.split())


def try_math(message):
    m = _MATH_ONE.search(message)
    if m:
        a, sym, b = m.group(1), m.group(2), m.group(3)
    elif len(message.split()) <= 6:
        mw = _MATH_WORDS.search(message)
        if not mw:
            return None
        sym = _MATH_WORD_SYM.get(_math_word_sym(mw.group(2)))
        if not sym:
            return None
        a, b = mw.group(1), mw.group(3)
    else:
        return None
    a = a.replace(",", ".")
    b = b.replace(",", ".")
    try:
        x, y = float(a), float(b)
    except ValueError:
        return None
    sym = "×" if sym in "*x×" else ("÷" if sym in "/÷" else sym)
    if sym == "+":
        r = x + y
    elif sym == "-":
        r = x - y
    elif sym == "×":
        r = x * y
    else:
        if y == 0:
            return None
        r = x / y
    if abs(r - round(r)) < 1e-9:
        out = str(int(round(r)))
    else:
        out = ("%.6f" % r).rstrip("0").rstrip(".")
    return "%s %s %s = %s" % (a, sym, b, out)


# ============================================================
# طلب الصور: استخراج الوصف، ومعرفة هل المطلوب صورة واقعية ولا رسم seshat بالمربعات
# ============================================================
# "صورة لقطة" = صورة + لـ + قطة: اللام بتتشال، إلا لو الكلمة نفسها بتبدأ بلام (لمبة، ليمون…)
_L_OF = (r"(?:\s+ل(?!مب|يمون|عب|بن|وح|يل|حم|ؤلؤ|اعب|ون|سان|هب|ندن|بنان|يبيا|اب|عوب|ص|غز|حظ|قاء)"
         r"(?=[\u0621-\u064A]{3}))?")
_NOUN = r"(?:صور[ةه]|لوح[ةه]|رسم[ةه])"
# الأفعال الصريحة: المطلوب صورة جديدة حتى لو فيه صورة مرفوعة
_DRAW_VERBS = (
    r"(?:ارسم(?:ي|ها|ه|لي|ولي|\s+لي)?|[تن]رسم(?:لي|ي|\s+لي)?)(?![\w\u0600-\u06FF])|"
    r"(?:[اأإ]نش[ئيى]|انشاء|إنشاء|ول[ّ]?د|صمم|اصنع)(?:لي|\s+لي)?\s+" + _NOUN
    + r"(?![\w\u0600-\u06FF])" + _L_OF + r"|"
    r"(?:draw(?:\s+me)?|imagine|"
    r"(?:generate|create|make)(?:\s+me)?\s+(?:an?\s+)?(?:image|picture|photo|art|drawing)(?:\s+of)?)"
    r"(?![\w\u0600-\u06FF])")
# صيغ ممكن تكون طلب صورة جديدة أو كلام عن صورة مرفوعة ("اعمل صورة أبيض وأسود")
_DRAW_SOFT = (
    r"(?:اعمل|اعملي|هات|هاتلي|عايز|عاوز|عايزه|عاوزه|محتاج|اريد|أريد|ابغى|أبغى)(?:لي|\s+لي)?\s+"
    + _NOUN + r"(?![\w\u0600-\u06FF])" + _L_OF + r"|"
    r"(?:" + _NOUN + r"|ديزاين)\s+(?:لـ\s*|ل(?!مب|يمون|عب|بن|وح|يل|حم|ؤلؤ|اعب|ون|سان|هب|ندن|بنان|يبيا|اب|عوب|ص|غز|حظ|قاء)"
    r"(?=[\u0621-\u064A]{3}))|"
    r"(?:image|picture|photo)\s+of(?![\w\u0600-\u06FF])")
_DRAW_RE = re.compile(r"(?<![\w\u0600-\u06FF])(?:(?P<strong>" + _DRAW_VERBS + r")|" + _DRAW_SOFT
                      + r")", re.I)
_LL = re.compile(r"(" + _NOUN + r")\s+لل(?=[\u0621-\u064A]{2})")      # "صورة للقطة" → "صورة لالقطة"

_FILLERS = re.compile(
    r"^\s*(لي|لانا|ليا|من\s*فضلك|لو\s*سمحت|ممكن|please|me|a|an|the|of|صورة|صوره|رسمة|رسمه|"
    r"لوحة|لوحه|لـ|عن)\s+", re.I)

# طلب صريح لرسم seshat نفسه بالمربعات
_PIXEL_RE = re.compile(
    r"(بالمربعات|مربعات|بكسل|بيكسل|بنفسك|ب[اإ]يدك|رسمتك\s+انت|pixel|in\s+pixels|by\s+yourself|"
    r"yourself)", re.I)

# تعديل بالوصف على آخر صورة اتولدت: "خليها بالليل"، "ضيف قبعة"، "make it night"
_EDIT_DESC_RE = re.compile(
    r"^\s*(?:ممكن\s+|لو\s+سمحت\s+|please\s+|can\s+you\s+|could\s+you\s+)?"
    r"(خل[يى]\S*|غي[ّ]?ر\S*|بد[ّ]?ل\S*|عد[ّ]?ل\S*|ضيف\S*|[اأ]ضف\S*|زو[ّ]?د\S*|حط\S*|شيل\S*|"
    r"احذف\S*|امسح\S*|لو[ّ]?ن\S*|اجعل\S*|make|change|turn|add|put|remove|replace|give)"
    r"(?![\w\u0600-\u06FF])", re.I)
# صورة تانية لنفس الوصف
_AGAIN_RE = re.compile(
    r"^\s*(?:(?:عايز|عاوز|هات|اعمل|ولد|ارسم)\S*\s+)?(?:(?:صور[ةه]|واحد[ةه])\s+(?:تاني[ةه]|كمان|غيرها)|"
    r"كمان\s+(?:صور[ةه]|واحد[ةه])|غيرها|جرب\s+تاني|تاني|another(?:\s+one)?|one\s+more|again|"
    r"try\s+again)\s*[.!؟?]*\s*$", re.I)


def try_draw(message):
    """
    طلب صورة؟ بيرجع {"prompt": الوصف، "strong": فعل رسم صريح، "pixels": رسم seshat بالمربعات}
    أو None.
    """
    message = _LL.sub(r"\1 لال", message)
    m = _DRAW_RE.search(message)
    if not m:
        return None
    prompt = message[m.end():]                 # الوصف = اللي بعد فعل الطلب
    pixels = bool(_PIXEL_RE.search(prompt))
    prompt = _PIXEL_RE.sub(" ", prompt)
    prompt = re.sub(r"\s+", " ", prompt).strip()
    for _ in range(4):
        new = _FILLERS.sub("", prompt)
        if new == prompt:
            break
        prompt = new.strip()
    prompt = re.sub(r"\s+(please|من\s*فضلك|لو\s*سمحت)\s*[.!؟?]*$", "", prompt, flags=re.I)
    prompt = prompt.strip("؟?،,. -–—:")
    if len(prompt) < 2:
        return None
    return {"prompt": prompt, "strong": bool(m.group("strong")), "pixels": pixels}


def parse_image_edit(message):
    """
    كلام عن آخر صورة اتولدت؟ بيرجع ("edit", نص التعديل) أو ("again", "") أو None.
    """
    msg = (message or "").strip()
    if not msg or len(msg) > 300:
        return None
    if _AGAIN_RE.match(msg):
        return "again", ""
    if _EDIT_DESC_RE.match(msg):
        return "edit", msg.strip(" .!؟?")
    return None


# ============================================================
# بحث في الإنترنت — ويكيبيديا أولًا (مصدر موثوق) و DuckDuckGo بعدها
# ============================================================
_WIKI_QUERY_DROP = {
    "مين", "ايه", "إيه", "ازاي", "إزاي", "اين", "أين", "امتى", "إمتى",
    "ليه", "لماذا", "في", "من", "عن", "على", "هو", "هي", "هل", "الى",
    "يا", "ان", "أن", "اللي", "اللى", "يكون", "بتكون", "the", "what",
    "who", "how", "where", "when", "why", "is", "are", "was", "were",
    "a", "an", "of", "in", "on", "to", "do", "does", "did", "and",
    "or", "for", "with", "about",
    # جولة 15: أفعال الاستخراج وكلمات المقارنة — كانت بتتلخبط في تقييم
    # العلاقة (بحث "مين اخترع الراديو" كان بيرفض صفحة الراديو عشان
    # كلمة "اخترع" مش مكتوبة في أول ٣ سطور منها → score 0.5 < 0.6 → رفض
    # جاف من غير بحث). الكلمات دي مش كلمات موضوع — الموضوع هو اللي بيتقيّم:
    "اخترع", "اكتشف", "كام", "عدد", "الفرق", "بين", "معنى", "تعريف",
    "invented", "discovered", "difference", "between", "meaning", "how many",
}


def _wiki_search_query(query):
    """كلمات الاستفهام بتضرب في البحث (بحثت عن "مين اكتشف امريكا" جابت
    طائرة الكونكورد!) فبنبعت الكلمات المهمة بس."""
    words = [w for w in query.split() if len(w) >= 3 and w not in _WIKI_QUERY_DROP]
    return " ".join(words) if words else query


def _wiki_word_match(w, hay_words):
    """كلمة السؤال موجودة في كلمات الصفحة؟ (تتحمل اختلاف الهمزات والصيغ)"""
    if w in hay_words:
        return True
    for hw in hay_words:
        if difflib.SequenceMatcher(None, w, hw).ratio() >= 0.8:
            return True
    return False


def _wiki_relevant(query, title, extract):
    """بيقيّم علاقة النتيجة بالسؤال من 0 لـ 1 — عشان بحث "اكتشف امريكا"
    ما يجيبش حواديت سيرته فيها كلمة "اكتشف" بالصدفة."""
    from tokenizer import normalize_question
    qwords = [w for w in normalize_question(_wiki_search_query(query)).split()
              if len(w) >= 3]
    if not qwords:
        return 1.0
    hay = set(normalize_question(title + " " + extract).split())
    matched = sum(1 for w in qwords if _wiki_word_match(w, hay))
    return matched / len(qwords)


def _wiki(query):
    lang = "ar" if _is_arabic_text(query) else "en"
    domain = f"{lang}.wikipedia.org"
    src_name = "ويكيبيديا" if lang == "ar" else "Wikipedia"
    try:
        r = requests.get(f"https://{domain}/w/api.php",
                         params={"action": "query", "list": "search",
                                 "srsearch": _wiki_search_query(query),
                                 "format": "json", "srlimit": 3},
                         headers=_UA, timeout=_NET_TIMEOUT)
        hits = (r.json().get("query") or {}).get("search") or []
        if not hits:
            return None
        # بنجرب أهم ٣ نتائج ونختار أفضل واحدة عن الموضوع فعلًا (٦٠٪+)
        best, best_score = None, 0.0
        for hit in hits[:3]:
            title = hit["title"]
            r2 = requests.get(
                f"https://{domain}/api/rest_v1/page/summary/" +
                quote(title, safe=""),
                headers=_UA, timeout=_NET_TIMEOUT)
            d = r2.json()
            if d.get("type") == "disambiguation":      # صفحة توضيح مالهاش رد واحد
                continue
            extract = (d.get("extract") or "").strip()
            if len(extract) < 60:
                continue
            score = _wiki_relevant(query, title, extract)
            if score > best_score:
                parts = re.split(r"(?<=[.!؟])\s+", extract)
                text = " ".join(parts[:3]).strip()
                if len(text) > 700:
                    text = text[:697].rsplit(" ", 1)[0] + "…"
                url = ((d.get("content_urls") or {}).get("desktop") or
                       {}).get("page", "")
                best = {"text": text, "url": url, "title": title}
                best_score = score
            if best_score >= 0.99:
                break
        if not best or best_score < 0.6:
            return None
        src = (f"المصدر: {src_name} — {best['title']}" if lang == "ar"
               else f"Source: Wikipedia — {best['title']}")
        think = (f"دورت في {src_name} على «{best['title']}»" if lang == "ar"
                 else f"searched {src_name} for “{best['title']}”")
        return {"text": best["text"] + "\n\n" + src,
                "think": think, "url": best["url"]}
    except Exception:
        return None


def _ddg(query):
    # مهلة أقصر من ويكي: لو DDG مش متاح (شبكة/حجب) مياخدش ١٢ ثانية كاملة
    # وما يعلقش الرد — ويكيبيديا هي المصدر الأساسي للعربي.
    ddg_timeout = min(8, _NET_TIMEOUT)
    try:
        r = requests.post("https://lite.duckduckgo.com/lite/", data={"q": query},
                          headers={**_UA, "Referer": "https://lite.duckduckgo.com/"},
                          timeout=ddg_timeout)
        links = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>\s*(.*?)\s*</a>', r.text)
        snippets = re.findall(
            r"class=['\"]result-snippet['\"][^>]*>(.*?)</td>", r.text, re.S)
        results = []
        for href, title in links:
            if not href.startswith("http"):
                if "uddg=" in href:
                    qs = parse_qs(urlparse("https:" + href).query)
                    if "uddg" in qs:
                        href = unquote(qs["uddg"][0])
                    else:
                        continue
                else:
                    continue
            if "duckduckgo.com" in urlparse(href).netloc:
                continue
            title = re.sub(r"<[^>]+>", "", title).strip()
            if not title:
                continue
            results.append((title, href))
            if len(results) >= 3:
                break
        if not results:
            return None
        ar = _is_arabic_text(query)
        lines = ["دي أقرب معلومات لقيتها على النت:" if ar
                 else "Here is the closest information I found on the web:"]
        for i, (title, href) in enumerate(results):
            snip = ""
            if i < len(snippets):
                snip = re.sub(r"<[^>]+>", "", snippets[i]).strip()
                snip = re.sub(r"\s+", " ", snip)[:220]
            dom = urlparse(href).netloc.replace("www.", "")
            lines.append("• " + title + (" — " + snip if snip else "") +
                         ("\n  المصدر: " if ar else "\n  Source: ") + dom)
        return {"text": "\n".join(lines),
                "think": ("دورت في DuckDuckGo على سؤالك" if ar
                          else "Searched DuckDuckGo for your question"),
                "url": results[0][1]}
    except Exception:
        return None


def web_answer(query):
    # العربي: ويكيبيديا الأول (بحثها العربي كويس) — الإنجليزي: DuckDuckGo الأول
    # (بحث ويكيبيديا الإنجليزي بيتلخبط في الأسئلة زي "capital of france")
    if _is_arabic_text(query):
        w = _wiki(query)
        if w:
            return w
        return _ddg(query)
    d = _ddg(query)
    if d:
        return d
    return _wiki(query)


# ============================================================
# المخ الخارجي — أي خدمة متوافقة مع OpenAI (مفتاح اختياري)
# ============================================================
_LLM_SYSTEM = (
    "أنت seshat، مساعد ذكي مصري طوّره عمرو أحمد، واسمك seshat."
    " بترحب بأسلوب ودود وبسيط ومختصر."
    " لو الرسالة بالعربي جاوب بالمصري البسيط، ولو بالإنجليزي جاوب بالإنجليزي."
    " جاوب على أي سؤال معلومات بدقة، ومتألفش معلومات وانت مش متأكد."
    " لما تكتب كود اكتبه في بلوك ``` لغة ```."
    " أنت seshat — لا تذكر أي نظام أو نموذج أو خدمة تانية أبدًا.")


def llm_answer(message, cfg):
    if not cfg.api_key:
        return None
    base = (cfg.api_base or "https://api.openai.com/v1").rstrip("/")
    try:
        r = requests.post(
            base + "/chat/completions",
            headers={"Authorization": "Bearer " + cfg.api_key,
                     "Content-Type": "application/json"},
            json={"model": cfg.api_model,
                  "messages": [{"role": "system", "content": _LLM_SYSTEM},
                               {"role": "user", "content": message[:4000]}],
                  "temperature": 0.6,
                  "max_tokens": 1024},
            timeout=60)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        return text or None
    except Exception:
        return None


def _other_subject(question, suggestion):
    """نفس شكل السؤال بس عن حاجة تانية؟ (شوف understanding.other_subject)"""
    try:
        from engine import understanding
        from tokenizer import normalize_question as norm
        return understanding.other_subject(norm(question), norm(suggestion))
    except Exception:
        return False


def llm_vision(message, image_url, cfg):
    """
    فهم صورة بالمخ الخارجي (لو فيه مفتاح والموديل بيشوف صور، زي gpt-4o-mini).
    image_url: الصورة data URL. بيرجع النص أو None لو مفيش مفتاح / الموديل مبيشوفش / حصل خطأ.
    """
    if not cfg.api_key:
        return None
    base = (cfg.api_base or "https://api.openai.com/v1").rstrip("/")
    ask = message.strip() or "اوصف الصورة دي بالتفصيل: فيها إيه؟"
    try:
        r = requests.post(
            base + "/chat/completions",
            headers={"Authorization": "Bearer " + cfg.api_key,
                     "Content-Type": "application/json"},
            json={"model": cfg.api_model,
                  "messages": [{"role": "system", "content": _LLM_SYSTEM},
                               {"role": "user", "content": [
                                   {"type": "text", "text": ask[:2000]},
                                   {"type": "image_url", "image_url": {"url": image_url}}]}],
                  "temperature": 0.4,
                  "max_tokens": 700},
            timeout=60)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        return text or None
    except Exception:
        return None


# ============================================================
# المخ الكامل — بيجاوب بالترتيب: أدب → حساب → رسم → تعلم →
# النموذج المحلي → المخ الخارجي → البحث في النت → كلام صريح
# ============================================================
class Brain:

    def __init__(self, engine=None):
        self.engine = engine            # ممكن يكون None في وضع المحلي الخالص
        self.cfg = BrainConfig()
        self.learned = LearnedMemory()
        from imagegen import ImageMaker
        self.images = ImageMaker(key=self.cfg.image_key, llm=self._llm_translate)

    def _llm_translate(self, ask):
        """ترجمة وصف الصورة بالمخ الخارجي لو متربط (غير كده imagegen بيستخدم طرق تانية)"""
        return llm_answer(ask, self.cfg) if self.cfg.api_key else None

    def status(self):
        return {"search": self.cfg.search_enabled, "api": bool(self.cfg.api_key),
                "api_model": self.cfg.api_model if self.cfg.api_key else "",
                "learned": len(self.learned.items),
                "politeness": True, "images": True}

    def _out(self, response, source, source_label="", thinking="",
             confidence=1.0, fallback=False, matched="", suggestion="",
             image=None, image_id=None):
        return {"response": response, "thinking": thinking, "image_id": image_id,
                "confidence": round(float(confidence), 3), "seconds": 0.0,
                "fallback": fallback, "source": source, "matched": matched,
                "suggestion": suggestion, "image": image,
                "source_label": source_label}

    def _own_drawing(self, msg, temperature, max_new_tokens):
        """رسمة النموذج المحلي نفسه. بيرجع رد المحرك (ممكن يكون fallback) أو None لو مفيش محرك."""
        if self.engine is None or not hasattr(self.engine, "draw"):
            return None
        try:
            return self.engine.draw(msg, temperature=temperature, max_new_tokens=max_new_tokens)
        except Exception:
            return None

    def reply(self, message, temperature=0.0, max_new_tokens=2000, **kw):
        t0 = time.time()
        out = self._reply(message, temperature=temperature,
                          max_new_tokens=max_new_tokens, last_image=kw.get("last_image"))
        if not out.get("seconds"):
            out["seconds"] = round(time.time() - t0, 2)
        return out

    def _reply(self, message, temperature=0.0, max_new_tokens=2000, **kw):
        t0 = time.time()
        msg = (message or "").strip()
        if not msg:
            return self._out("اكتب حاجة الأول 🙂", "none", "")

        # 1) أي لفظ خارج → رد مؤدب دايمًا (المطلوب الأول والأهم)
        if is_insult(msg):
            pool = POLITE_AR if _is_arabic_text(msg) else POLITE_EN
            return self._out(random.choice(pool), "politeness", "أدب")

        # 2) اعتذار بعد شتم؟ بنتعامل بكرم
        if _APOLOGY_RE.match(msg) and len(msg) < 60:
            pool = APOLOGY_AR if _is_arabic_text(msg) else APOLOGY_EN
            return self._out(random.choice(pool), "politeness", "أدب")

        # 3) حساب مباشر — مضمون ١٠٠٪ حتى أرقام كبيرة وكسور
        math = try_math(msg)
        if math:
            return self._out(math, "math", "حساب", thinking="حسبتها بنفسي بدقة")

        # 4) الصور.
        #    • "ارسم / اعمل صورة …" → صورة واقعية من مولّد الصور (imagegen.py). الرد بيرجع
        #      فورًا والصورة بتتولد لما المتصفح يطلبها من /generated/<id>.jpg.
        #    • "ارسم بالمربعات …" أو المولّد مقفول → رسم seshat نفسه (النموذج المحلي).
        #    • فيه صورة اتولدت قبل كده (last_image) والكلام تعديل عليها ("خليها بالليل") أو
        #      "صورة تانية" → صورة جديدة مبنية على وصف القديمة.
        arabic = _is_arabic_text(msg)
        draw = try_draw(msg)
        last = self.images.meta(kw.get("last_image")) if kw.get("last_image") else None
        if draw and (draw["pixels"] or not self.cfg.external_images):
            own = self._own_drawing(msg, temperature, max_new_tokens)
            if own:
                return self._out(own["response"], "draw", "رسم seshat",
                                 thinking=own.get("thinking", ""),
                                 fallback=bool(own.get("fallback")),
                                 confidence=own.get("confidence", 1.0))
            return self._out("الرسم محتاج النموذج المحلي، وهو مش شغال دلوقتي." if arabic else
                             "Drawing needs the local model, which is not loaded right now.",
                             "draw", "رسم seshat", fallback=True)
        if draw:
            image_id = self.images.new(draw["prompt"])
            return self._out(("بجهزلك صورة «%s»." if arabic else "Making an image of “%s”.")
                             % draw["prompt"], "image", "صور",
                             thinking=("طلب صورة: " if arabic else "image request: ") + draw["prompt"],
                             image="generated/%s.jpg" % image_id, image_id=image_id, confidence=0.9)
        if last and self.cfg.external_images:
            change = parse_image_edit(msg)
            if change and change[0] == "again":
                image_id = self.images.new("", parent=last["id"])
                return self._out(("صورة تانية لـ «%s»." if arabic else "Another image of “%s”.")
                                 % last["prompt"], "image", "صور",
                                 thinking="صورة تانية لنفس الوصف" if arabic
                                 else "another image for the same description",
                                 image="generated/%s.jpg" % image_id, image_id=image_id,
                                 confidence=0.9)
            if change:
                image_id = self.images.new("", parent=last["id"], edit=change[1])
                text = ("بعمل صورة جديدة بالوصف: «%s».\n(ده توليد من الأول بنفس البذرة مش تعديل "
                        "على نفس البكسلات، فممكن تفاصيل تانية تتغير.)" if arabic else
                        "Making a new image from: “%s”.\n(This regenerates the picture with the "
                        "same seed; it does not edit the same pixels, so other details may change.)")
                return self._out(text % self.images.meta(image_id)["prompt"], "image", "صور",
                                 thinking=("تعديل بالوصف على آخر صورة: " if arabic
                                           else "edit by description on the last image: ") + change[1],
                                 image="generated/%s.jpg" % image_id, image_id=image_id,
                                 confidence=0.8)

        # 5) تعليم يدوي: علمني: س: … ج: …
        teach = parse_teach(msg)
        if teach:
            q, a = teach
            self.learned.add(q, a, src="user")
            return self._out(
                "اتعلمت ✔ سؤال: «%s» — والمرة اللي جاية هرد عليك من الذاكرة." % q,
                "learn", "تعلم جديد", thinking="خزنت سؤال جديد في ذاكرتي")

        # 5.5) نسيان: انسى: السؤال — بيحذف حاجة اتعلمت غلط
        forget_q = parse_forget(msg)
        if forget_q:
            removed = self.learned.forget(forget_q)
            if removed:
                return self._out("انسيتها ✔ — مسحت «%s» من الذاكرة." % removed,
                                 "learn", "تعلم جديد",
                                 thinking="مسحت سؤال من الذاكرة")
            return self._out(
                "مش لاقي «%s» في اللي اتعلمته — تأكد من السؤال وجرب تاني." % forget_q,
                "learn", "تعلم جديد", fallback=True)

        # 6) حاجة اتعلمتها بنفسي قبل كده — تعليم المستخدم وردود المخ الخارجي
        #    أولوية على كل حاجة، لكن أجوبة الويب المتخزنة ما بتغطيش على النموذج:
        #    بتتفعل بس لو النموذج والبحث الحي ما جابوش حاجة (شوف خطوة 9.5)
        learned = self.learned.lookup(extract_question(msg))
        if not learned:
            learned = self.learned.lookup(msg)
        if learned and learned.get("src") in ("user", "llm"):
            return self._out(learned["a"], "learned", "ذاكرة التعلم",
                             thinking="من اللي اتعلمته: «%s»" % learned["q"])
        web_cached = learned if learned and learned.get("src") == "web" else None

        # 7) النموذج المحلي: الهوية والأكواد وكل اللي اتعلمه — لو واثق ممنقلش
        local = None
        if self.engine is not None:
            try:
                local = self.engine.reply(msg, temperature=temperature,
                                          max_new_tokens=max_new_tokens)
            except Exception:
                local = None
        if local and not local.get("fallback"):
            local.setdefault("image", None)
            local.setdefault("source_label",
                             SOURCE_LABELS.get(local.get("source"), ""))
            local["seconds"] = round(time.time() - t0, 2)
            return local

        # النموذج المحلي معندوش المعلومة، بس فهم نوع السؤال وموضوعه (engine.py + data/understanding.py)
        understood = local if local and local.get("source") == "understand" else None
        arabic = _is_arabic_text(msg)

        # 8) المخ الخارجي (لو فيه مفتاح): شبكة عصبية ضخمة بجاوب أي سؤال
        llm = llm_answer(msg, self.cfg)
        if llm:
            self.learned.add(msg, llm, src="llm")
            return self._out(llm, "llm", "شبكة عصبية خارجية",
                             thinking="المخ الخارجي جاوب، وخزنت الرد في الذاكرة")

        # 8.5) أسئلة ردّها معروف من نوعها (رأي، توقع، طلب تنفيذ، سعر النهارده…): البحث
        #      فيها بيجيب كلام ملوش علاقة، فـ seshat بيرد بالجملة اللي كوّنها من فهمه.
        if understood and not understood.get("searchable"):
            return self._understood(understood, t0)

        # 9) بحث في النت — ويكيبيديا و DuckDuckGo. لو seshat حدد موضوع السؤال بنبحث بيه
        #    ("كلمني عن نابليون" → "نابليون") عشان النتيجة تبقى أدق.
        if self.cfg.search_enabled:
            topic = (understood or {}).get("topic") or ""
            use_topic = topic and understood.get("kind") in ("about", "person")
            web = (web_answer(topic) if use_topic else None) or web_answer(msg)
            if web:
                self.learned.add(msg, web["text"], src="web")
                think = web["think"]
                if understood:
                    think = understood.get("thinking", "") + " — " + think
                return self._out(web["text"], "web", "بحث ويب",
                                 thinking=think, confidence=0.9)

        # 9.5) جواب ويب متخزن من جلسة قبل — بيتفعل هنا بس، بعد ما النموذج
        #      والبحث الحي ما جابوش حاجة، عشان ما يغطيش على معرفة النموذج المتدربة
        if web_cached:
            return self._out(web_cached["a"], "learned", "ذاكرة بحث سابقة",
                             thinking="من بحث سابق: «%s»" % web_cached["q"],
                             confidence=0.75)

        # 10) البحث مجابش حاجة. لو السؤال قريب من سؤال يعرفه → يرد بيه ويقول إنه افترض كده.
        sug = (local or {}).get("suggestion") or ""
        if sug and _other_subject(msg, sug):
            sug = ""          # نفس شكل السؤال بس عن حاجة تانية (سكان الصين ≠ سكان مصر)
        if sug and self.engine is not None:
            try:
                near = self.engine.reply(sug, temperature=temperature,
                                         max_new_tokens=max_new_tokens)
            except Exception:
                near = None
            if near and not near.get("fallback"):
                lead = ("لو قصدك «%s»:\n" if arabic else "If you mean “%s”:\n") % sug
                return self._out(lead + near["response"], "memory", "أقرب سؤال",
                                 thinking=(local or {}).get("thinking", ""),
                                 confidence=(local or {}).get("confidence", 0.0),
                                 matched=sug)
        # 11) وإلا → الرد اللي seshat كوّنه من فهمه للسؤال (مش جملة ثابتة)
        if understood:
            return self._understood(understood, t0)
        text = ("مش قادر أحدد قصدك من «%s». ممكن تقولها بكلمات تانية؟" if arabic else
                "I cannot tell what you mean by “%s”. Could you say it in other words?") % msg[:80]
        return self._out(text, "none", "", fallback=True, confidence=0.0)

    def _understood(self, local, t0):
        """رد متكوّن من فهم السؤال (نوعه وموضوعه) — بيتعرض كرد عادي مش كـ\"مش عارف\""""
        out = dict(local)
        out.update(fallback=False, source="understand", source_label="فهم السؤال",
                   image=None, seconds=round(time.time() - t0, 2))
        out["confidence"] = 1.0 if out.get("understood_by") in ("model", "model-text", "rules") else 0.5
        return out

if __name__ == "__main__":
    # اختبار سريع من التيرمنال:  python brain.py "انت اهبل"
    import sys

    class _NoEngine:
        def reply(self, *a, **kw):
            return {"response": "(local)", "thinking": "", "confidence": 1.0,
                    "seconds": 0, "fallback": False, "source": "model",
                    "matched": "", "suggestion": ""}

    b = Brain(_NoEngine())
    for m in sys.argv[1:] or ["انت اهبل", "7 × 8", "ارسم لي قطة بتنظر للقمر"]:
        print(m, "→", b.reply(m)["response"][:120].replace("\n", " / "))
