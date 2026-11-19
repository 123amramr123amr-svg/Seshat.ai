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
    # خ17 — الصيغ المفتوحة الطبيعية: «اتكلم عن/في X» و«اكتب لي فكرة عن X»
    # و«اكتب سطرين عن X» — كلها نية تكوين جمل جديدة عن موضوع، ودي اللي كانت
    # بتقع في «مش قادر أحدد قصدك» رغم إنها أسهل طلب مفتوح.
    r"|(?:^|[\s،,.؟!])(?:اتكلم|تكلّ?م|احكي)(?:\s+لي)?\s+(?:في|عن|حول)\s+"
    r"|(?:^|[\s،,.؟!])(?:اكتب)?\s*لي?\s*فكر(?:[ةه])\s+جديد(?:[ةه])?\s+(?:عن|حول|في)\s+"
    r"|(?:^|[\s،,.؟!])اكتب\s+(?:سطرين|جملتين|فقرة\s+قصير[ةه])\s+(?:عن|حول|في)\s+"
    r"|talk\s+(?:something|about)|write\s+(?:two\s+lines|a\s+few\s+words)\s+about"
    r"|(?:compose|write|create|make|generate|give\s*me)\s+"
    r"(?:me\s+)?(?:a\s+|some\s+)?(?:new\s+)?sentence",
    re.IGNORECASE)

# خ17 — نية «رأيك في X»: تكوين رأي مؤلف لحظياً من بنوك صفات، مش رفض
_OPINION_RE = re.compile(
    r"(?:^|[\s،,.؟!])(?:ايه\s+رأيك|ايه\s+تقريك|رايك\s+في|قول\s+رايك\s+في|تنصحني\s+في)"
    r"|what(?:'s|\s+is)\s+your\s+(?:opinion|view|take)\s+(?:on|about)",
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

# خ17 — ربط الموضوع بمكانه الطبيعي: جملة «عن البحر» تبقى جواها البحر فعلًا
# مش جملة عامة بالصدفة — الخريطة بتكبر مع كل جولة بيانات جديدة (طلب المستخدم)
TOPIC_PLACES_AR = {
    "بحر": "عند الشاطئ", "شاطئ": "عند الشاطئ", "موج": "داخل القارب",
    "قارب": "داخل القارب", "سمك": "عند الشاطئ",
    "نيل": "قرب النهر", "نهر": "قرب النهر", "جبل": "فوق الجبل",
    "زرع": "في الحقل", "فلاح": "في الحقل", "سوق": "في السوق",
    "شجر": "تحت الشجرة", "شجرة": "تحت الشجرة", "حديق": "في الحديقة",
    "كتاب": "في المكتبة", "قرا": "في المكتبة", "مكتب": "في المكتبة",
    "مدرس": "في الحديقة", "تعليم": "في الحديقة", "مدين": "في المدينة القديمة",
    "عمل": "في السوق", "شغل": "في السوق",
}

# خ17 — بنك الرأي المؤلف: صفات عامة + جملة ختامية، والرد بيبان إنه تكوين
OPINION_QUALITIES_AR = [
    "موضوع بيساعد على التفكير الهادي",
    "حاجة بتاخد منها وقتها الجرعة الصح من غير إفراط",
    "شغل بيصنع عادة كويسة لو اتكرر يوميًا",
    "شيء بيوسّع الخيال وبيطمّن العقل",
    "حاجة بتجمع الناس وتحكي عنها وقت جميل",
]
OPINION_CLOSERS_AR = [
    "الأهم إنك تبقى متحمّس له مش مجبر عليه، فالاستمرار بييجي من الحب مش من الضغط.",
    "لو حبيت تفاصيل أدق اسألني سؤال محدد عنه وأنا هجاوبك من اللي اتعلّمته.",
    "الجرعة الصح بتتحدد بتجربتك الشخصية — ابدأ صغير وزوّد بالتدريج.",
]
OPINION_QUALITIES_EN = [
    "something that supports calm thinking",
    "a habit worth keeping in moderation",
    "an activity that rewards daily repetition",
    "something that widens imagination and steadies the mind",
]
OPINION_CLOSERS_EN = [
    "Try it yourself and grade it honestly.",
    "The key is genuine interest, not pressure — consistency follows love.",
]


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


def _topic_place(topic):
    """خ17: مكان طبيعي للموضوع من الخريطة — البحر يطلع عند الشاطئ مش في مدينة."""
    t = (topic or "").translate(_NORM_TABLE)
    for key, place in TOPIC_PLACES_AR.items():
        if key in t:
            return place
    return ""


def _arabic_sentences(topic):
    """٣ جمل عربية جديدة — الموضوع بيغطي الجمل الثلاثة لو موجود، والمكان
    المرتبط بيه بيحط في الأولى عشان الجمل تبقى فعلًا عن الموضوع (تطوير خ17)."""
    subj = random.sample(AR_SUBJECTS, 2)
    verbs = random.sample(AR_VERBS, 3)
    objs = random.sample(AR_OBJECTS, 3)
    place = _topic_place(topic) or random.choice(AR_PLACES)
    time_a, time_b = random.sample(AR_TIMES, 2)

    v0 = _agree_ar(subj[0], verbs[0])
    v1 = _agree_ar(subj[1], verbs[1])
    v2 = _agree_ar(subj[1], verbs[2])
    focus = topic or objs[1]
    s1 = ("%s %s %s %s %s." % (subj[0], v0, focus, place, time_a) if topic
          else "%s %s %s %s." % (subj[0], v0, objs[0], place))
    s2 = "%s، %s %s %s." % (time_b, subj[1], v1, focus)
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


def try_opinion(message):
    """خ17 — كشف نية الرأي وتكوين رأي معلن: بيرجع dict بالصيغة الموحدة أو None."""
    if not message or not _OPINION_RE.search(message):
        return None
    arabic = _is_arabic(message)
    m = _OPINION_RE.search(message)
    after = message[m.end():]
    after = re.sub(r"^(?:ال|في|عن|حول)\s+", "", after.strip())
    topic = _clean_topic(after)
    if not topic:
        return None
    if arabic:
        q = random.choice(OPINION_QUALITIES_AR)
        c = random.choice(OPINION_CLOSERS_AR)
        resp = "رأيي في «%s» (تكويني الشخصي، مش معلومة ثابتة): «%s» — %s" % (topic, q, c)
    else:
        q = random.choice(OPINION_QUALITIES_EN)
        c = random.choice(OPINION_CLOSERS_EN)
        resp = "My take on \"%s\" (composed by me, not a fixed fact): \"%s\" — %s" % (topic, q, c)
    return {"arabic": arabic, "topic": topic, "sentences": [], "response": resp}


def try_compose(message):
    """كشف طلب تكوين جملة أو رأي — بيرجع dict كامل أو None.
    خ17: الرأي ليه مسار لوحدُه، والصيغ المفتوحة (اتكلم عن/فكرة عن) بتعبر عادي."""
    if not message:
        return None
    op = try_opinion(message)
    if op:
        return op
    if not _COMPOSE_RE.search(message.translate(_NORM_TABLE)):
        return None
    arabic = _is_arabic(message)
    topic = _extract_topic(message)
    # خ17: تنظيف الموضوع من كلمات الكم المتبقية («سطرين عن الكتابة» → «الكتابة»)
    if topic:
        if " عن " in topic:
            topic = topic.split(" عن ")[-1]
        if " في " in topic:
            topic = topic.split(" في ")[-1]
        for w in ("سطرين ", "جملتين ", "فقرة "):
            if topic.startswith(w):
                topic = topic[len(w):].strip()
        topic = _clean_topic(topic)
    sentences = _arabic_sentences(topic) if arabic else _english_sentences(topic)
    return {
        "arabic": arabic,
        "topic": topic,
        "sentences": sentences,
        "response": _build_response(arabic, topic, sentences),
    }
