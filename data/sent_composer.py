"""
تكوين الجمل — جولة خ11
=======================

seshat بيكوّن جمل جديدة بنفسه من بنوك كلمات (فاعل / فعل / مفعول به /
شبه جملة مكان و زمان) مع دعم موضوع اختياري («كوّن جملة عن البحر»).

التركيبة عشوائية كل مرة — مش منسوخة من الذاكرة ولا من ملف التدريب،
وده بالظبط اللي بتطلبه جولة خ11: تكوين كلام وإنشاء جمل جديدة.

الواجهة:
    try_compose(message) → dict أو None
        {"arabic", "topic", "sentences": [3 جمل], "response": نص الرد}
"""

import random
import re

_NORM_TABLE = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ة": "ه",
                             "ى": "ي", "ؤ": "و", "ئ": "ي"})

_COMPOSE_RE = re.compile(
    r"(?:^|[\s،,.؟!])(?:كوّ?ن|اكتب|ولّ?د|تأليف|اقترح|جرب|هات)\s*"
    r"(?:لي)?\s*(?:٣|3)?\s*جمل(?:[ةه])?(?=[\s،,.؟!:]|$)"
    r"|(?:^|[\s،,.؟!])جمل(?:[ةه])?\s*جديد(?:[ةه])?(?=[\s،,.؟!]|$)"
    r"|(?:compose|write|create|make|generate|give\s*me)\s+"
    r"(?:me\s+)?(?:a\s+|some\s+)?(?:new\s+)?sentence",
    re.IGNORECASE)

_TOPIC_RE = re.compile(
    r"(?:عن|حول|في)\s+([\u0621-\u064A][\u0621-\u064A\s]{1,38})"
    r"|about\s+([a-z\s]{1,38})", re.IGNORECASE)

_TRIM_RE = re.compile(r"[\s،,.؟!:؛]+$")


# ----------------------------------------------------------------------
# بنوك الكلمات — فصحى بسيطة + إنجليزي
# ----------------------------------------------------------------------
AR_SUBJECTS = ["الصياد", "الطفلة", "القبطان", "الفلاح", "الرسام", "الطبيب",
               "المهندس", "البحار", "الطالب", "العجوز", "القطة", "العصفور"]

# المؤنث لازم توافق في الفعل — نعرفهم عشان التذكير/التأنيث يتعمل صح
AR_FEM_SUBJECTS = {"الطفلة", "القطة"}

AR_VERBS = ["يبحث عن", "يجمع", "يرسم", "يكتشف", "يذكر", "يراقب", "يصور",
            "يحلم بـ", "يحكي عن", "يهتم بـ", "يفكر في", "يدوّن"]

AR_OBJECTS = ["اللؤلؤ", "الزهور", "النجوم", "القصص القديمة", "الطيور",
              "الألوان", "الأحلام", "الخرائط", "المذكرات", "الأنغام"]

AR_PLACES = ["عند الشاطئ", "في السوق", "فوق الجبل", "في الحديقة",
             "قرب النهر", "في المدينة القديمة", "تحت الشجرة", "على الشرفة"]

AR_TIMES = ["في الصباح الباكر", "وقت الغروب", "في الليل الهادئ",
            "بعد المطر", "في فصل الربيع", "يوم الجمعة"]

EN_SUBJECTS = ["the fisherman", "a young girl", "the captain", "an old artist",
               "a curious student", "the farmer", "a little bird", "the doctor"]

EN_VERBS = ["searches for", "collects", "paints", "discovers", "describes",
            "watches", "dreams about", "writes about", "thinks about"]

EN_OBJECTS = ["pearls", "wild flowers", "distant stars", "old stories",
              "colorful birds", "secret maps", "bright dreams", "old songs"]

EN_PLACES = ["by the shore", "in the old market", "on the hilltop",
             "in the garden", "near the river", "under the big tree"]

EN_TIMES = ["at dawn", "at sunset", "on quiet nights", "after the rain",
            "in spring", "every friday"]


# ----------------------------------------------------------------------
# أدوات
# ----------------------------------------------------------------------
def _is_arabic(text):
    return any("\u0600" <= ch <= "\u06FF" for ch in (text or ""))


def _clean_topic(topic):
    topic = _TRIM_RE.sub("", (topic or "").strip())
    topic = re.sub(r"\s+", " ", topic)
    if len(topic) < 2:
        return ""
    return topic[:40]


def _extract_topic(message):
    m = _TOPIC_RE.search(message)
    if not m:
        return ""
    return _clean_topic(m.group(1) or m.group(2) or "")


# ----------------------------------------------------------------------
# التكوين
# ----------------------------------------------------------------------
def _agree_ar(subject, verb):
    """موافقة الفعل للفاعل: المؤنث ياخد «ت» مكان «ي» أول الفعل (تحكي/تجمع)."""
    if subject in AR_FEM_SUBJECTS and verb.startswith("ي"):
        return "ت" + verb[1:]
    return verb


def _arabic_sentences(topic):
    """٣ جمل عربية جديدة — الموضوع بيدخل في جملة منهم لو موجود."""
    subj = random.sample(AR_SUBJECTS, 2)
    verbs = random.sample(AR_VERBS, 3)
    objs = random.sample(AR_OBJECTS, 3)
    place = random.choice(AR_PLACES)
    time_a, time_b = random.sample(AR_TIMES, 2)

    v0 = _agree_ar(subj[0], verbs[0])
    v1 = _agree_ar(subj[1], verbs[1])
    v2 = _agree_ar(subj[1], verbs[2])
    s1 = "%s %s %s %s." % (subj[0], v0, objs[0], place)
    focus = topic or objs[1]
    s2 = "%s، %s %s %s." % (time_a, subj[1], v1, focus)
    s3 = "%s %s %s أو %s %s." % (subj[1], v2, focus, objs[2], time_b)
    return [s1, s2, s3]


def _english_sentences(topic):
    """3 fresh English sentences — the topic enters one of them if given."""
    subj = random.sample(EN_SUBJECTS, 2)
    verbs = random.sample(EN_VERBS, 3)
    objs = random.sample(EN_OBJECTS, 3)
    place = random.choice(EN_PLACES)
    time_a, time_b = random.sample(EN_TIMES, 2)

    s1 = "%s %s %s %s." % (subj[0], verbs[0], objs[0], place)
    focus = topic or objs[1]
    s2 = "%s, %s %s %s." % (time_a, subj[1], verbs[1], focus)
    s3 = "%s %s %s or %s %s." % (subj[1], verbs[2], focus, objs[2], time_b)
    return [s1, s2, s3]


def _build_response(arabic, topic, sentences):
    where = "عن «%s»" % topic if topic else ""
    if arabic:
        head = "تمام — دي ٣ جمل جديدة كوّنتها بنفسي %s:\n" % where
        body = "\n".join("%d) %s" % (i + 1, s) for i, s in enumerate(sentences))
        tail = ("\n\nالتركيب: فاعل + فعل + مفعول به + شبه جملة (مكان أو زمان). "
                "كل تركيبة بتتولد لحظة الطلب — مش منسوخة من الذاكرة. "
                "عايز جمل عن موضوع معين؟ قول: كوّن جملة عن الشجر.")
        return head + body + tail
    head = "Sure — here are 3 fresh sentences i composed myself %s:\n" % (
        "about \"%s\"" % topic if topic else "")
    body = "\n".join("%d) %s" % (i + 1, s) for i, s in enumerate(sentences))
    tail = ("\n\nStructure: subject + verb + object + adverbial (place/time). "
            "Each composition is generated at request time — not memorized. "
            "Want sentences about a topic? say: compose a sentence about the sea.")
    return head + body + tail


def try_compose(message):
    """كشف طلب تكوين جملة — بيرجع dict كامل أو None."""
    if not message or not _COMPOSE_RE.search(message.translate(_NORM_TABLE)):
        return None
    arabic = _is_arabic(message)
    topic = _extract_topic(message)
    sentences = _arabic_sentences(topic) if arabic else _english_sentences(topic)
    return {
        "arabic": arabic,
        "topic": topic,
        "sentences": sentences,
        "response": _build_response(arabic, topic, sentences),
    }
