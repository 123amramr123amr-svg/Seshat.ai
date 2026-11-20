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
  4. الرسم النصي — "ارسم بالمربعات …" → النموذج المحلي بيرسم بنفسه بحروف ومربعات
     (engine.draw) — مخرج نصي في الشات نفسه. مفيش توليد صور خالص (نموذج الصور اتشال
     بطلب المستخدم — أي طلب صورة بيرجع رد مؤدب بيقول إن النموذج نصي).
  5. مولّد الأكواد الضخمة (خ11) — "اكتب كود لعبة اكس او / تطبيق مهام flask…" →
     برنامج كامل (100+ سطر) بتتكوّن لحظيًا من قوالب data/big_code.py — مضمون
     100% من غير نموذج ولا نت، بتعليقات عربية وطريقة تشغيل.
  6. تكوين جمل جديدة (خ11) — "كوّن جملة عن…" → تركيبات جديدة بنفسه من بنوك
     كلمات (data/sent_composer.py) مع شرح التركيب النحوي — مش منسوخة من الذاكرة.
  7. سياق المحادثة (خ11) — المتابعات القصيرة («كمان / اكمل / ليه…») بترجع
     لآخر موضوع اتكلم عنه بدل «مش فاهم قصدك».
  8. مخ خارجي اختياري — لو ضفت مفتاح API متوافق مع OpenAI (زرار الإعدادات
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
import sys
import time
from urllib.parse import parse_qs, quote, unquote, urlparse

import requests

from engine import extract_question

# خ28 — حاسة البصر (إدخال فقط، حكم خ10 ثابت): لو الرسالة جواها مسار صورة PNG
# موجود فعلًا على الجهاز، عين seshat (eyes.py) بتقراه وبتحوّله وصف نصي هيكلي
# بيدخل للنموذج مع السؤال — اتجاه واحد صورة→نص بدون أي إخراج صور.
_IMG_PATH_RE = re.compile(r"[^\s\"']+\.png", re.IGNORECASE)


def _resolve_image_path(message, image_path=None):
    """بيرجع أول مسار PNG موجود فعلًا من المعامل أو من جوه الرسالة، وإلا None."""
    if image_path and os.path.isfile(str(image_path)):
        return str(image_path)
    for cand in _IMG_PATH_RE.findall(message or ""):
        if os.path.isfile(cand):
            return cand
    return None


HERE = os.path.dirname(os.path.abspath(__file__))

# خ11: مولّد الأكواد الضخمة + مكوّن الجمل — وحدات في data/ بتتكلم بالمسار نفسه
_DATA_DIR = os.path.join(HERE, "data")
if _DATA_DIR not in sys.path:
    sys.path.insert(0, _DATA_DIR)
from big_code import big_code_request, build_big_code, big_code_guide
from sent_composer import try_compose
from nawah import growth
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
    "long": "سياق ممتد",
    "none": "",
}

# خ12 — السياق الممتد: الرسايل الأطول من الحد ده بتتعامل كمستند + سؤال
# وبتتقرا بالنافذة المنزلقة (لحد ٢٠٠٠٠٠٠ حرف) بدل ما بيتقص منها آخر جزء بس.
LONG_MESSAGE_CHARS = 1400


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
        # مفيش توليد صور خالص — النموذج نصي بس (نموذج الصور اتشال بطلب المستخدم).
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

# خ29 — كلمات بريئة شائعة تحتوي جذور شتم بالصدفة كجذور فرعية، فكانت
# بتتفعّل غلط زي ما اتحصّل في خ28: «عندمي مقابلة شغل…» → «مقابله» فيها
# «ابله» جوّه فتفعّل رد الأدب (false-positive). بنفحصها الأول ونعدّي
# عليها من فحص الجذور (القوائم المضبوطة زي الشتم الصريح ما بتتأثرش).
_INNOCENT_WORDS = {
    "مقابله", "المقابله", "مقابلات",          # فيها «ابله» فرعية
    "معارض", "معارضه", "معارضات", "المعارض",  # فيها «عارض» فرعية
}

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
            if w in _INNOCENT_WORDS:      # خ29: كلمة بريئة معروفة — نعدّي
                continue
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

# خ8: أسئلة الهوية — بترد من تعريف النموذج الموحد نفسه (IDENTITY_AR/EN)
# مش من الكوربس، فالهوية بتتحدث مع أي جولة دمج من غير إعادة تدريب.
_IDENTITY_Q = re.compile(
    r"(?<![\u0600-\u06FF])(مين|من)\s*(انت|انتي|حضرتك)"
    r"|(?<![\u0600-\u06FF])(انت|انتي|حضرتك)\s*(مين|ايه|إيه)"
    r"|مين\s*(عملك|مطورك|مصنعك|صانعك)|مطورك\s*مين|مين\s*طورك"
    r"|عرفني\s*بنفسك|عرف\s*نفسك|عرّفني\s*بنفسك|تعرف\s*على\s*نفسك"
    r"|اسمك\s*(ايه|إيه)|ما\s*اسمك|انت\s*مين\s*ب?ال?ظبط|انت\s*ب?ال?ظبط\s*مين"
    r"|who\s*are\s*you|what\s*are\s*you|what'?s\s*your\s*name|your\s*name"
    r"|who\s*(made|built|created|developed)\s*you|introduce\s*yourself", re.I)


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


# خ11: سياق المحادثة — متابعات قصيرة بترجع لآخر موضوع اتكلم عنه
_FOLLOWUP_RE = re.compile(
    r"^(?:كمان|وزياد[ةه]|زياد[ةه]|اكمل|كمّ?ل|ليه\??|وانت\??|وانت ايه\??|"
    r"طب\??|طيب\??|more|continue|and you\??|why\??|go on)\W*$",
    re.IGNORECASE)

_TOPIC_STOP = STOP_WORDS | {
    "كلمني", "احكيلي", "قوللي", "عايز", "عاوز", "ممكن", "لازم", "جرب",
    "هات", "خلاص", "كفاية", "بصراحة", "تقصد", "معني", "يعني", "tell",
    "want", "give", "please", "seshat",
    # خ11: كلمات المتابعة نفسها ما تبقاش موضوع — عشان السياق يفضل على آخر حاجة حقيقية
    "كمان", "وزياده", "زياده", "اكمل", "كمل", "طب", "طيب", "وانت",
    "مش", "انا", "انت", "اوي", "جدا", "كده",
    "more", "continue", "also", "still",
}


def _topic_of(message):
    """موضوع الرسالة: أول كلمات معنوية — بيتسجل عشان سياق المحادثة (خ11).
    الرسايل القصيرة زي «كمان» مش بتجيب موضوع، فآخر موضوع بيفضل قايم."""
    words = re.findall(r"[\w\u0600-\u06FF]{3,}", message or "")
    core = [w for w in words
            if w.lower() not in _TOPIC_STOP and not w.isdigit()]
    return " ".join(core[:3]) if core else ""


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


# ------------------------------------------------------------------
# خ8: مسائل متسلسلة بالكلمات («2 زائد 3 زائد 10» / «5 ناقص 2 مضروب في 4»)
# بتحول كلمات العمليات لرموز، وتتأكد إن اللي فاضل أرقام وعمليات بس
# (عشان «في» العادي في الجمل ما تتحسبش)، وبعدين تقيّم بأولوية عمليات
# حقيقية (ضرب/قسمة قبل جمع/طرح) — من غير eval خالص.
# ------------------------------------------------------------------
def _ar_word_re(word):
    """كلمة عملية عربي بكلمة كاملة بس (فيها ما تتعدلش جوه)."""
    return re.compile(r"(?<![\u0600-\u06FF])" + re.escape(word) + r"(?![\u0600-\u06FF])")


_WORD_OPS_ORDERED = [        # الأطول الأول عشان «مضروب في» ما تتبوظش بـ«في»
    ("مضروب في", "*", None), ("تقسيم على", "/", None), ("مقسوم على", "/", None),
    ("زائد", "+", None), ("ناقص", "-", None), ("ضرب", "*", None),
    ("على", "/", None), ("في", "*", None),
]
for _i, (_w, _s, _r) in enumerate(_WORD_OPS_ORDERED):
    _WORD_OPS_ORDERED[_i] = (_w, _s, _ar_word_re(_w))
_X_RE = re.compile(r"(?<![A-Za-z])x(?![A-Za-z])", re.I)          # 2 x 3 (حرف x لوحده)
_MATH_STRIP_FILLER = re.compile(r"^(احسبلي|احسب|calc|calculate)\s+", re.I)
_MATH_CHAIN_OK = re.compile(r"^[\d\s+\-*/.]+$")

# خ9: كلمات الأرقام العربي (فصحى + عامية) → أرقام. عشان «اثنين زائد تلاتة»
# و«اثنين في خمسة» تشتغل زي «2 زائد 3» بالظبط — تحويل بس، من غير أي تخمين.
_NUM_WORDS = {
    "صفر": "0",
    "واحد": "1", "واحدة": "1",
    "اتنين": "2", "اثنين": "2", "اثنان": "2",
    "تلاتة": "3", "ثلاثة": "3", "تلات": "3", "تلاته": "3",
    "اربعة": "4", "أربعة": "4", "اربع": "4", "أربع": "4", "اربعه": "4",
    "خمسة": "5", "خمسه": "5", "خمس": "5",
    "ستة": "6", "سته": "6", "ست": "6",
    "سبعة": "7", "سبعه": "7", "سبع": "7",
    "ثمانية": "8", "ثمانيه": "8", "تمانية": "8",
    "تسعة": "9", "تسعه": "9", "تسع": "9",
    "عشرة": "10", "عشره": "10", "عشر": "10",
    "عشرين": "20", "تلاتين": "30", "ثلاثين": "30", "أربعين": "40", "اربعين": "40",
    "خمسين": "50", "ستين": "60", "سبعين": "70", "ثمانين": "80", "تسعين": "90",
    "مية": "100", "ميه": "100", "مئة": "100", "مائة": "100",
    "ميتين": "200", "مئتين": "200",
    "ألف": "1000", "الف": "1000", "ألفين": "2000", "الفين": "2000",
    # «و» المربوطة بالرقم (واحد وعشرين → واحد + وعشرين)
    "وعشرين": "20", "وتلاتين": "30", "وثلاثين": "30", "وأربعين": "40", "واربعين": "40",
    "وخمسين": "50", "وستين": "60", "وسبعين": "70", "وثمانين": "80", "وتسعين": "90",
    "ومية": "100", "وميه": "100", "ومئة": "100", "ومائة": "100",
    "وألف": "1000", "والف": "1000",
}
_NUM_WORD_RE = sorted(((w, _ar_word_re(w)) for w in _NUM_WORDS),
                      key=lambda kv: -len(kv[0]))
# تركيب الوحدات والعشرات/المئات: «1 20» → 21 (بعد تحويل الكلمات)
_NUM_COMPOUND_RE = re.compile(r"([1-9])\s+(20|30|40|50|60|70|80|90|100|200|1000|2000)\b")


def _num_words_to_digits(text):
    """كلمات الأرقام → أرقام + دمج التركيب (واحد وعشرين → 21)."""
    for w, rx in _NUM_WORD_RE:
        text = rx.sub(" %s " % _NUM_WORDS[w], text)
    text = re.sub(r"\s+", " ", text).strip()
    while True:
        m = _NUM_COMPOUND_RE.search(text)
        if not m:
            return text
        text = (text[:m.start()] + str(int(m.group(1)) + int(m.group(2)))
                + text[m.end():])


_MATH_TRAIL_JUNK = re.compile(r"[\s؟?!،,؛:.]+$")            # علامات سؤال وترقيم في الآخر
_MATH_TRAIL_WORD = re.compile(r"\s*(?:كام|بكام|يساوي|يعني)+\s*$")  # كلمة سؤال قفارة


def _word_expr(message):
    """مسألة متسلسلة بالكلمات → تعبير رمزي نقي («2 + 3 + 10»)، أو None."""
    if len(message.split()) > 12:
        return None
    t = _MATH_STRIP_FILLER.sub("", message.strip())
    t = _MATH_TRAIL_JUNK.sub("", t).strip()            # خ29: «…ضرب اربعة؟» → «…ضرب اربعة»
    t = _MATH_TRAIL_WORD.sub("", t).strip()            # خ29: «…ضرب اربعة كام» → «…ضرب اربعة»
    t = _MATH_TRAIL_JUNK.sub("", t).strip()            # خ29: مرور تاني لو كلمة السؤال كانت قبل ترقيم
    for w, sym, rx in _WORD_OPS_ORDERED:
        t = rx.sub(" %s " % sym, t)
    t = _X_RE.sub(" * ", t)
    t = _num_words_to_digits(t)                       # خ9: اتنين زائد تلاتة → 2 + 3
    t = re.sub(r"\s+", " ", t).strip()
    if not t or not _MATH_CHAIN_OK.search(t):
        return None
    return t


def _eval_math_chain(expr):
    """تقييم تعبير رمزي بأولوية عمليات — بيرجع نص الناتج أو None."""
    toks = re.findall(r"\d+(?:\.\d+)?|[+\-*/]", expr)
    merged = []                                            # دمج الناقص الأحادي
    for t in toks:
        if t == "-" and (not merged or merged[-1] in "+-*/"):
            merged.append("u-")
        else:
            merged.append(t)
    nums, ops = [], []
    sign = 1.0
    for t in merged:
        if t == "u-":
            sign = -sign
            continue
        if t in "+-*/":
            if not nums:
                return None
            ops.append(t)
        else:
            nums.append(sign * float(t))
            sign = 1.0
    if len(nums) < 2 or len(ops) != len(nums) - 1:
        return None
    # أولوية الضرب والقسمة الأول
    vals, ops2 = [nums[0]], []
    for o, v in zip(ops, nums[1:]):
        if o == "*":
            vals[-1] *= v
        elif o == "/":
            if v == 0:
                return None
            vals[-1] /= v
        else:
            ops2.append(o)
            vals.append(v)
    r = vals[0]
    for o, v in zip(ops2, vals[1:]):
        r = r + v if o == "+" else r - v
    if abs(r - round(r)) < 1e-9:
        return str(int(round(r)))
    return ("%.6f" % r).rstrip("0").rstrip(".")


def try_math(message):
    m = _MATH_ONE.search(message)
    if m:
        a, sym, b = m.group(1), m.group(2), m.group(3)
    elif len(message.split()) <= 6:
        mw = _MATH_WORDS.search(message)
        if not mw:
            return _try_chain_math(message)
        sym = _MATH_WORD_SYM.get(_math_word_sym(mw.group(2)))
        if not sym:
            return _try_chain_math(message)
        a, b = mw.group(1), mw.group(3)
    else:
        return _try_chain_math(message)
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


def _try_chain_math(message):
    """خ8: المسائل المتسلسلة («2 زائد 3 زائد 10») — بعد ما الطرق القديمة ما لقوش."""
    expr = _word_expr(message)
    if not expr:
        return None
    r = _eval_math_chain(expr)
    if r is None:
        return None
    return "%s = %s" % (expr, r)


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


# ============================================================
# المخ الكامل — بيجاوب بالترتيب: أدب → حساب → رسم → تعلم →
# النموذج المحلي → المخ الخارجي → البحث في النت → كلام صريح
# ============================================================
class Brain:

    # خ10: هوية النموذج النصي — بعد حذف نموذج الصور بطلب المستخدم،
    # سؤال «انت مين» بيرجع هوية نصية صريحة: مفيش توليد صور خالص.
    IDENTITY_AR = ("أنا seshat — النموذج النصي المحلي: حوار وحساب وتعلم ذاتي وبحث، "
                   "ورسم نصي بالمربعات، وبكوّن أكواد برمجية ضخمة وجمل جديدة بنفسي. "
                   "مطوّريني عمرو أحمد، وكل حاجة فيا محلية، ومفيش توليد صور خالص.")
    IDENTITY_EN = ("I am seshat — the local text model: chat, math, self-learning and "
                   "search, plus text drawing with squares — and i build big code "
                   "and compose new sentences by myself. Developed by Amr Ahmed, "
                   "fully local, no image generation.")

    def __init__(self, engine=None):
        self.engine = engine            # ممكن يكون None في وضع المحلي الخالص
        self.cfg = BrainConfig()
        self.learned = LearnedMemory()
        self.last_topic = ""            # خ11: آخر موضوع في المحادثة (للسياق)

    def status(self):
        return {"search": self.cfg.search_enabled, "api": bool(self.cfg.api_key),
                "api_model": self.cfg.api_model if self.cfg.api_key else "",
                "learned": len(self.learned.items),
                "politeness": True, "images": False,
                "big_code": True, "compose": True,
                "images_engine": "مفيش — النموذج نصي بس"}

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
        img_path = kw.pop("image_path", None) or _resolve_image_path(message or "")
        out = self._reply(message, temperature=temperature,
                          max_new_tokens=max_new_tokens, image_path=img_path)
        # خ28 — لو الصورة اتقريت بنجاح: بندلّف الرد بتوضيح إن الوصف جاي من
        # حاسة البصر (إدخال نصي بس — حقل image بيفضل None دايمًا حكم خ10)
        if img_path and out and not out.get("fallback"):
            _lbl = out.get("source_label") or ""
            out["source_label"] = (_lbl + " + حاسة البصر").strip() if _lbl else "حاسة البصر"
            out["thinking"] = ((out.get("thinking") or "")
                               + " — قريت الصورة بعين seshat وديّت وصفها للنموذج نصًا").strip(" —")
        # خ11: تسجيل آخر موضوع من الرسالة عشان المتابعات القصيرة تحفظ السياق
        # («كمان» و«ليه» مش بتجيبوا موضوع، فآخر موضوع بيفضل قايم)
        topic = _topic_of(message)
        if topic:
            self.last_topic = topic
        if not out.get("seconds"):
            out["seconds"] = round(time.time() - t0, 2)
        return out

    def _reply(self, message, temperature=0.0, max_new_tokens=2000, image_path=None, **kw):
        t0 = time.time()
        msg = (message or "").strip()
        if not msg:
            return self._out("اكتب حاجة الأول 🙂", "none", "")

        # 0) خ28 — حاسة البصر: مسار PNG موجود في الرسالة أو من image_path؟
        #    الوصف النصي الهيكلي بيتضاف في أول الرسالة عشان الذاكرة والنموذج
        #    يشتغلوا على الصورة كمدخل نصي (مفيش أي إخراج صور).
        img = image_path or _resolve_image_path(msg)
        img_rest = None
        if img:
            try:
                import eyes
                _d = eyes.describe_path(img)
                img_rest = (msg.replace(img, "").strip(" :-–—:")
                            or "صف الصورة دي")
                msg = "[وصف الصورة من عين seshat: %s]\n%s" % (_d["text"], img_rest)
            except Exception:
                pass        # صورة غير مدعومة — نمشي بالنص عادي بدون وصف

        # 1) أي لفظ خارج → رد مؤدب دايمًا (المطلوب الأول والأهم)
        if is_insult(msg):
            pool = POLITE_AR if _is_arabic_text(msg) else POLITE_EN
            return self._out(random.choice(pool), "politeness", "أدب")

        # 2) اعتذار بعد شتم؟ بنتعامل بكرم
        if _APOLOGY_RE.match(msg) and len(msg) < 60:
            pool = APOLOGY_AR if _is_arabic_text(msg) else APOLOGY_EN
            return self._out(random.choice(pool), "politeness", "أدب")

        # 3) حساب مباشر — مضمون ١٠٠٪ حتى أرقام كبيرة وكسور (خ28: على سؤال
        #    المستخدم نفسه — وصف الصورة المضاف ما يخدشش الحساب)
        math = try_math(img_rest or msg)
        if math:
            return self._out(math, "math", "حساب", thinking="حسبتها بنفسي بدقة")

        # 3.1) خ15 — أدوات نصية متقدمة محلية: وحدات، إحصاء، تلخيص، أرقام بالحروف،
        #      تواريخ، كلمات مرور، تحليل نص — كلها بلا إنترنت وبلا مكتبات جديدة
        try:
            from text_tools import try_text_tool
            _tool = try_text_tool(img_rest or msg)   # خ28: مش على وصف الصورة
        except Exception:
            _tool = None
        if _tool:
            return self._out(_tool[0], "tools", "أداة متقدمة",
                             thinking=_tool[1], confidence=1.0)

        # 3.5) سؤال الهوية — من تعريف النموذج الموحد نفسه مش من الكوربس
        if _IDENTITY_Q.search(img_rest or msg) and len(img_rest or msg) <= 60:
            return self._out(self.IDENTITY_AR if _is_arabic_text(msg) else self.IDENTITY_EN,
                             "identity", "هوية",
                             thinking="سؤال عن هويتي — بجاوب من تعريفي الموحد")

        # 3.6) خ11: متابعة قصيرة («كمان / اكمل / ليه…») — بترجع لآخر موضوع اتكلم عنه
        if self.last_topic and _FOLLOWUP_RE.match(msg):
            if _is_arabic_text(msg):
                text = ("ماشي — لسه معاك في موضوع «%s». اسألني أي سؤال واضح عنه "
                        "وأنا جاهز، أو لو عايز تفاصيل أكتر اكتب السؤال كامل." % self.last_topic)
            else:
                text = ("Still on \"%s\" with you — ask me any clear question "
                        "about it." % self.last_topic)
            return self._out(text, "context", "سياق المحادثة",
                             thinking="متابعة قصيرة — رجعت لآخر موضوع في الكلام",
                             confidence=1.0)

        # 4) الرسم — النموذج نصي بس: مفيش توليد صور خالص (نموذج الصور اتشال خ10).
        #    • "ارسم بالمربعات …" → رسم seshat نفسه بحروف ومربعات (مخرج نصي في الشات).
        #    • أي طلب صورة عادي ("ارسم قطة" / "اعمل صورة") → رد مؤدب بيقول إن النموذج نصي
        #      وبيوجّه للرسم النصي بالمربعات — من غير أي ملف صور ولا مولّد.
        arabic = _is_arabic_text(msg)
        draw = try_draw(msg)
        if draw and draw["pixels"]:
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
            return self._out(
                ("أنا مش بعمل صور — أنا نموذج نصي: بحاور وأحسب وأتعلم وببحث وبيرسم بالمربعات بس. "
                 "لو عايز رسمة نصية قول: «ارسم %s بالمربعات»." if arabic else
                 "I don't make images — I'm a text model: chat, math, learning and search, "
                 "and text drawing with squares. Try: “draw %s with squares”.") % draw["prompt"],
                "text", "النموذج نصي",
                thinking="طلب صورة والنموذج نصي — توجه للرسم النصي بالمربعات",
                confidence=1.0)

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

        # 6.5) خ11: مولّد الأكواد الضخمة — قوالب بتتكوّن لحظيًا، مضمونة 100% من غير نموذج ولا نت
        code_req = big_code_request(msg)
        if code_req:
            if code_req["kind"] == "template":
                return self._out(build_big_code(code_req, arabic),
                                 "big_code", "مولّد الأكواد",
                                 thinking="كوّنت كود كامل من قوالب التركيب — جاهز للتشغيل")
            return self._out(big_code_guide(arabic), "big_code", "مولّد الأكواد",
                             thinking="طلب كود من غير موضوع معروف — عرضت القوايم الجاهزة")

        # 6.6) خ11: تكوين جمل جديدة بنفسه — من بنوك الكلمات، تركيبة جديدة كل مرة
        sent = try_compose(msg)
        if sent:
            return self._out(sent["response"], "compose", "تكوين الجمل",
                             thinking="كوّنت جمل جديدة من بنوك الكلمات — مش منسوخة من الذاكرة")

        # 6.65) خ18 بطلب المستخدم: تقرير البنية الذاتية من نواة النمو — أرقام
        #       حية من ملفات النموذج والبيانات بتتحدث لوحدها مع كل تكبير
        #       خ27: أسئلة «جدول/خطة النمو» بترجع جدول التوسعة والموقع الحي
        if growth.matches_growth_question(msg):
            _t27 = msg.lower()
            if ("جدول" in _t27 or "خطة" in _t27 or "توسع" in _t27):
                return self._out(growth.growth_plan_text(), "nawah", "نواة النمو",
                                 thinking="قريت جدول النمو والموقع الحي من مصفوفات النموذج مباشرة")
            return self._out(growth.growth_summary(), "nawah", "نواة النمو",
                             thinking="قريت بنيتي الحية من مصفوفات النموذج وملفات البيانات مباشرة")

        # 6.8) خ12: الرسايل الضخمة (مستند/ملف/سجل + سؤال في آخرها) — بتتقرا
        #      كاملة بالنافذة المنزلقة لحد ٢٠٠٠٠٠٠ حرف بدل ما تتقص لآخر ١٠٠٠.
        if (self.engine is not None and hasattr(self.engine, "reply_long")
                and len(msg) > LONG_MESSAGE_CHARS):
            try:
                long_out = self.engine.reply_long(msg, extract_question(msg),
                                                  temperature=temperature,
                                                  max_new_tokens=max_new_tokens)
            except Exception:
                long_out = None
            if long_out and not long_out.get("fallback"):
                long_out.setdefault("image", None)
                long_out.setdefault("source_label", SOURCE_LABELS.get("long", "سياق ممتد"))
                long_out["seconds"] = round(time.time() - t0, 2)
                return long_out

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
