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
     على طول. وكمان يقدر تعلمه بيدويًا:
         علمني: س: اللعبة المفضلة لعمرو ج: الشطرنج
  4. إنشاء صور — "ارسم لي قطة" بموقع مجاني من غير مفتاح (pollinations.ai).
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

HERE = os.path.dirname(os.path.abspath(__file__))
LEARNED_PATH = os.path.join(HERE, "data", "learned.json")
CONFIG_PATH = os.path.join(HERE, "data", "config.local.json")

_UA = {"User-Agent":
       "seshat.chet/1.0 (https://github.com/123amramr123amr-svg/Seshat.ai)"}
_NET_TIMEOUT = 12
LEARNED_MAX = 4000

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
        q = normalize_question(question)[:300]
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
        """
        from tokenizer import normalize_question
        q = normalize_question(message)
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
        if not best or best_ratio < 0.62:
            return None
        bs = _sig_words(best["q"])
        if not bs or not sig:
            return best
        matched = sum(1 for w in sig
                      if any(difflib.SequenceMatcher(None, w, b).ratio() >= 0.8
                             for b in bs))
        if matched / len(sig) >= 0.5:
            return best
        return None

    def forget(self, message):
        """بيمسح سؤال اتعلم — بالحرف أو أقرب واحد شبهه كفاية (٧٥٪+ عشان
        ما يمسحش سؤال تاني بالغلط). بيرجع السؤال الممسوح أو None."""
        from tokenizer import normalize_question
        q = normalize_question(message)
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
                       re.I)
_QA_RE = re.compile(
    r"^\s*(?:س:|السؤال:|سؤال:)?\s*(.+?)\s*(?:ج:|ج\s*:|الجواب:|جواب:|جواب\s*:|a:|answer:)\s*(.+)$",
    re.I | re.S)


def parse_teach(message):
    m = _TEACH_RE.match(message)
    if not m:
        return None
    qa = _QA_RE.match(m.group(2).strip())
    if not qa:
        return None
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


def try_math(message):
    m = _MATH_ONE.search(message)
    if not m:
        return None
    a = m.group(1).replace(",", ".")
    b = m.group(3).replace(",", ".")
    sym = m.group(2)
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
# رسم صور — مجانًا ومن غير مفتاح (pollinations.ai)
# ============================================================
_DRAW_RE = re.compile(
    r"\b(ارسم|ارسمي|ارسمها?|ارسملي|ارسم\s*لي|اعمل\s*صوره|انشئ\s*صوره|"
    r"اصنع\s*صوره|صوره\s*(لـ|ل)\s*|لوحه\s*(لـ|ل)\s*|رسمه\s*(لـ|ل)\s*|"
    r"ديزاين\s*(لـ|ل)\s*|draw|draw\s+me|generate\s+(an?\s+)?(image|picture|photo|art)|"
    r"create\s+(an?\s+)?(image|picture|photo)|image\s+of|picture\s+of|"
    r"photo\s+of)", re.I)

_FILLERS = re.compile(
    r"^\s*(لي|لانا|من\s*فضلك|please|me|a|an|the|صورة|صوره|رسمة|رسمه|"
    r"لوحة|لوحه|لـ)\s+", re.I)


_DRAW_STYLES = ("digital art", "oil painting", "watercolor", "photorealistic",
                "cartoon style", "fantasy art", "anime style", "3d render")


def try_draw(message):
    m = _DRAW_RE.search(message)
    if not m:
        return None
    prompt = _DRAW_RE.sub(" ", message)
    prompt = re.sub(r"\s+", " ", prompt).strip()
    for _ in range(3):
        new = _FILLERS.sub("", prompt)
        if new == prompt:
            break
        prompt = new.strip()
    prompt = prompt.strip("؟?،,. -–—")
    if len(prompt) < 2:
        prompt = "seshat golden egyptian art"
    # أسلوب عشوائي عشان نفس الرسمة كل مرة تطلع مختلفة
    style = random.choice(_DRAW_STYLES)
    # مفيش query params — الخدمة المجانية بتضربها بـ 402، والـ prompt لوحده هو الرابط
    url = "https://image.pollinations.ai/prompt/" + quote(prompt[:220] + ", " + style)
    return {"prompt": prompt, "url": url}


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
    try:
        r = requests.post("https://lite.duckduckgo.com/lite/", data={"q": query},
                          headers={**_UA, "Referer": "https://lite.duckduckgo.com/"},
                          timeout=_NET_TIMEOUT)
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
        lines = ["دي أقرب معلومات لقيتها على النت:"]
        for i, (title, href) in enumerate(results):
            snip = ""
            if i < len(snippets):
                snip = re.sub(r"<[^>]+>", "", snippets[i]).strip()
                snip = re.sub(r"\s+", " ", snip)[:220]
            dom = urlparse(href).netloc.replace("www.", "")
            lines.append("• " + title + (" — " + snip if snip else "") +
                         "\n  المصدر: " + dom)
        return {"text": "\n".join(lines),
                "think": "دورت في DuckDuckGo على سؤالك",
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


# ============================================================
# المخ الكامل — بيجاوب بالترتيب: أدب → حساب → رسم → تعلم →
# النموذج المحلي → المخ الخارجي → البحث في النت → كلام صريح
# ============================================================
class Brain:

    def __init__(self, engine=None):
        self.engine = engine            # ممكن يكون None في وضع المحلي الخالص
        self.cfg = BrainConfig()
        self.learned = LearnedMemory()

    def status(self):
        return {"search": self.cfg.search_enabled, "api": bool(self.cfg.api_key),
                "api_model": self.cfg.api_model if self.cfg.api_key else "",
                "learned": len(self.learned.items),
                "politeness": True, "images": True}

    def _out(self, response, source, source_label="", thinking="",
             confidence=1.0, fallback=False, matched="", suggestion="",
             image=None):
        return {"response": response, "thinking": thinking,
                "confidence": round(float(confidence), 3), "seconds": 0.0,
                "fallback": fallback, "source": source, "matched": matched,
                "suggestion": suggestion, "image": image,
                "source_label": source_label}

    def reply(self, message, temperature=0.0, max_new_tokens=600, **kw):
        t0 = time.time()
        out = self._reply(message, temperature=temperature,
                          max_new_tokens=max_new_tokens)
        if not out.get("seconds"):
            out["seconds"] = round(time.time() - t0, 2)
        return out

    def _reply(self, message, temperature=0.0, max_new_tokens=600, **kw):
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

        # 4) رسم صورة
        draw = try_draw(msg)
        if draw:
            return self._out(
                "اتفضل 🙂 رسمتلك «%s» — لو الصورة ماظهرتشن استنى دقيقة "
                "واجرب تاني (خدمة الرسم المجانية بتسمح بصورة كل 45 ثانية)." % draw["prompt"],
                "draw", "رسم صورة", thinking="رسمت صورة عن: " + draw["prompt"],
                image=draw["url"])

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

        # 6) حاجة اتعلمتها بنفسي قبل كده
        learned = self.learned.lookup(msg)
        if learned:
            return self._out(learned["a"], "learned", "ذاكرة التعلم",
                             thinking="من اللي اتعلمته: «%s»" % learned["q"])

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

        # 8) المخ الخارجي (لو فيه مفتاح): شبكة عصبية ضخمة بجاوب أي سؤال
        llm = llm_answer(msg, self.cfg)
        if llm:
            self.learned.add(msg, llm, src="llm")
            return self._out(llm, "llm", "شبكة عصبية خارجية",
                             thinking="المخ الخارجي جاوب، وخزنت الرد في الذاكرة")

        # 9) بحث في النت — ويكيبيديا و DuckDuckGo
        if self.cfg.search_enabled:
            web = web_answer(msg)
            if web:
                self.learned.add(msg, web["text"], src="web")
                return self._out(web["text"], "web", "بحث ويب",
                                 thinking=web["think"], confidence=0.9)

        # 10) مفيش حاجة ناسعت → كلام صريح + اقتراح لو النموذج عنده واحد
        sug = (local or {}).get("suggestion") or ""
        if sug:
            text = "مش متأكد إني فهمت سؤالك. تقصد: «%s»؟" % sug
        else:
            text = ("لسه ماتعلمتش الرد على السؤال ده ومش قادر أوصل للنت "
                    "دلوقتي. جرب تاني بعد شوية أو اسألني حاجة تانية.")
        return self._out(text, "none", "", fallback=True, suggestion=sug,
                         confidence=0.0)


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
