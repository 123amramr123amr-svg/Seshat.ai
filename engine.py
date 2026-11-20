"""
seshat.chet — محرك الرد: بيحمّل النموذج المحفوظ وبيرد على سؤال واحد.
بيستخدمه السيرفر (app.py) والشات في التيرمنال (chat.py).

فيه محركين بيقروا نفس ملف النموذج:
  • NumpySeshat — التشغيل بـ NumPy بس (من غير torch). خفيف جدًا على الرام،
    وده اللي بيشتغل على الاستضافة.
  • TorchSeshat — التشغيل بـ torch (للأحجام الكبيرة أو كارت الشاشة).

الاختيار تلقائي: NumPy هو الافتراضي. عشان تجبره:
    SESHAT_ENGINE=torch python app.py
"""

import math
import os
import re
import sys
import threading
import time

# خ28 — تسريع التوليد: مصفوفات التوليد توكن-ب-توكن صغيرة (GEMV)، وBLAS متعدد
# الخيوط بيصرف وقت أكتر في المزامنة من الحساب نفسه — خيط واحد أسرع قياسًا.
# setdefault بيحترم قيمة معمولة قبل التشغيل عشان المقارنة العادلة تفضل ممكنة.
for _blas_t in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_blas_t, "1")

import numpy as np

from checkpoint import load_checkpoint
from tokenizer import CharTokenizer, normalize_question, is_arabic

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "seshat_chet_model.npz")
sys.path.insert(0, os.path.join(HERE, "data"))
import drawings                                    # noqa: E402  (شكل الرسمة والتأكد منها)
import understanding                               # noqa: E402  (فهم نوع السؤال وموضوعه)

# لو ثقة النموذج في الرد أقل من الحد ده، بنعتبر إنه مش عارف الإجابة وبنقول كده
# بصراحة بدل ما نطلّع كلام ملخبط.
# الثقة = متوسط احتمال الحروف اللي اختارها، مع تركيز على أول 12 حرف: دول اللي
# النموذج بيقرر فيهم هيرد بإيه، ولو متردد فيهم يبقى غالبًا السؤال برّه اللي اتعلمه.
# الرقم متظبط بالقياس: بيسيب ~97% من الردود الصح، وبيمسك حوالي تلتين الأسئلة
# اللي برّه التدريب (مش كلها — النموذج الصغير بيبقى واثق حتى وهو غلطان).
CONFIDENCE_THRESHOLD = 0.93
CONFIDENCE_HEAD = 12

# أقصى طول للجزء اللي بيوصل للنموذج من الرسالة الطويلة (آخر جزء منها) —
# بتتبع السياق الاتسع: 4096 // 4 مع حد 1000 حرف، والتدريب بيغطي نفس الحد
# بالأغلفة الطويلة في dataset.py.
INPUT_CAP = 1000


def _trim_memory():
    """بيرجّع للنظام الرام اللي اتستخدمت مؤقتًا في بناء الذاكرة (لينكس بس)"""
    try:
        import ctypes
        import gc
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _numbers(text):
    """الأرقام اللي في النص مترتبة — عشان نتأكد إن النموذج قرا المسألة صح"""
    return sorted(re.findall(r"\d+", text))


def confidence_from(probs, n_think=0):
    """
    probs: احتمال كل توكن النموذج اختاره. n_think: عدد توكنز سطر التفكير.
    لو فيه تفكير، الثقة بتتحسب عليه (هو اللي بيحدد النموذج فهم السؤال ولا لأ).
    """
    if not probs:
        return 0.0
    head = probs[:n_think] if n_think else probs[:CONFIDENCE_HEAD]
    return min(sum(probs) / len(probs), sum(head) / len(head))

FALLBACK_AR = "لسه ماتعلمتش الرد على السؤال ده. جرب تسأله بطريقة تانية، أو اسألني عن حاجة تانية."
FALLBACK_EN = "I haven't learned how to answer that yet. Try asking in another way, or ask me something else."

# ذاكرة الأسئلة (memory.py): التشابه من 0 لـ 1
MEMORY_STRONG = 0.60      # فوق كده الذاكرة بترد لوحدها
MEMORY_AGREE = 0.30       # فوق كده بترد لو النموذج فهم السؤال بنفس المعنى
MEMORY_SUGGEST = 0.35     # فوق كده (ومن غير اتفاق) بيسأل المستخدم "تقصد كذا؟"
UNKNOWN_MARGIN = 0.15     # لو السؤال أقرب بكتير لموضوع برّه اللي اتعلمه → مش عارف
MATH_LABEL = "حساب"       # نوع السؤال في سطر التفكير لما يكون مسألة حساب
DRAW_LABELS = ("رسم |", "drawing |")      # سطر التفكير لما يكون المطلوب رسمة
DRAW_WORDS = ("ارسم", "ارسملي", "ارسمي", "draw")
DRAW_MAX_WRONG = 20       # أقصى عدد مربعات غلط (من 256) عشان الرسمة تتعرض — وبنقول العدد
DRAW_MAX_OBJECTS = 3      # أقصى عدد رسومات جنب بعض في صورة واحدة


def draw_help(arabic):
    """لما المطلوب حاجة لسه ماتعلمش يرسمها: بيقول بصراحة هو بيعرف يرسم إيه"""
    names = "، ".join(drawings.known_names(True)) if arabic else ", ".join(drawings.known_names(False))
    if arabic:
        return ("دي لسه ماتعلمتش أرسمها. أنا برسم بنفسي مربع مربع، واللي اتعلمته لحد دلوقتي: "
                f"{names}. وأي واحدة فيهم بأي لون (أحمر، أخضر، أزرق، أصفر، برتقالي، بنفسجي، أسود، "
                "بمبي، أبيض، بني، رمادي)، وينفع لحد 3 رسومات مع بعض. جرب: ارسم قطة سودا وشمس.")
    return ("I have not learned to draw that yet. I draw by myself, cell by cell, and so far I know: "
            f"{names}. Any of them in red, green, blue, yellow, orange, purple, black, pink, white, "
            "brown or grey, and up to 3 drawings together. Try: draw a black cat and a sun.")

# ============================================================
# استخراج السؤال من رسالة طويلة
# ============================================================
# الناس لما تكتب رسالة طويلة بتحط التحية والكلام في الأول أو في الآخر والسؤال
# جواه. النموذج الصغير بيتلخبط لو اتجاهله الرسالة كلها (بيقرا أول جزء على
# إنه السؤال)، فالمحرك بيستخرج السؤال قبل ما يدخل للنموذج والذاكرة:
#  1. آخر كلمة سؤال في الرسالة غالبًا هي بداية السؤال الفعلي. وكلمات الطلب
#     (عايز/اكتب/احسب/what…) السؤال بيبدأ منها بالظبط، فبنمشي من عندها
#     مباشرة، وكلمات السؤال اللي بتيجي جوه السؤال (ايه/مين/كام/who…)
#     بناخد شوية كلام قبلها عشان ما نتقطعش من أوله.
#  2. مفيش كلمة سؤال → بناخد آخر جملة (تقسيم على السطور والترقيم) أو آخر
#     1000 حرف (INPUT_CAP). الرسايل القصيرة بتتحول زي ما هي.
_Q_STARTERS = {"عايز", "عاوز", "ممكن", "اكتب", "اكتبلي", "اعمللي", "قوللي", "قولى",
               "علمني", "احسب", "طب",
               "write", "give", "tell", "translate", "define", "print", "calculate", "how"}
_Q_WORD = re.compile(
    r"[؟?]"
    r"|(?<![\w])(ايه|مين|ازاي|امتى|فين|ليه|كام|ناتج|احسب|اكتب|اكتبلي|اعمللي|قوللي|قولى|علمني|يعني ايه|معنى|عايز|عاوز|ممكن|طب|بص)"
    r"|(?<![\w])(what|how|who|where|when|why|which|write|give|tell|translate|capital|define|print|calculate|mean)",
    re.I)


def extract_question(message):
    """بيرجع جزء الرسالة اللي فيه السؤال الفعلي (للنموذج والذاكرة).
    الرسالة القصيرة أو اللي مفيهاش كلام زيادة بترجع زي ما هي."""
    if len(message) <= 40:
        return message
    matches = list(_Q_WORD.finditer(message))
    if matches:
        last = matches[-1]
        if last.group(0).lower() in _Q_STARTERS:
            # كلمة الطلب نفسها هي بداية السؤال — من عندها على طول
            seg = message[last.start():]
        else:
            # كلمة سؤال جوه السؤال — كلمة أو اتنين قبلها عشان ما يتقطعش من أوله
            start = message.rfind(" ", 0, max(0, last.start() - 12))
            seg = message[start + 1:] if start > 0 else message
        seg = seg.strip()
        if len(seg) > INPUT_CAP:
            seg = seg[-INPUT_CAP:]
        if len(seg) >= 8:                     # الجزء المنستخرج معقول؟
            return seg
    # مفيش كلمة سؤال → آخر جملة (أو آخر 1000 حرف لو مفيش تقسيم واضح)
    parts = [p.strip() for p in re.split(r"[\n]+|[،؛,.!]+", message) if p.strip()]
    if len(parts) > 1:
        tail = parts[-1]
        if len(tail.split()) < 3 and len(parts) >= 2:
            tail = parts[-2] + " " + tail
        return tail[-INPUT_CAP:]
    return message[-INPUT_CAP:]

# سطر التفكير اللي النموذج بيكتبه لما يحس إن السؤال برّه اللي اتعلمه
UNKNOWN_AR = "مش معروف | سؤال برّه اللي اتعلمته"
UNKNOWN_EN = "unknown | outside what I have learned"


# ============================================================
# التحقق من ناتج الحساب
# ============================================================
# النموذج بيفهم المسألة صح (بيكتب الأرقام في سطر التفكير) لكن ممكن يغلط في
# حساب الناتج — خصوصًا الأرقام الكبيرة اللي ماتدربش عليها (الجمع والطرح
# متدرب لحد 20 والضرب لحد 12). هنا بنقرا المسألة من سطر التفكير ونحسبها
# بنفسنا كفحص ثاني، ولو الناتج المكتوب غلط بنصلحه. المسألة بتتعبر متحققة
# بس لو أرقام السؤال = أرقام سطر التفكير = أرقام الرد، وساعتها الرد مضمون.
_THINK_MATH = re.compile(r"^" + MATH_LABEL + r" \| (\d+) ([+\-*x×]) (\d+)$")
_ANSWER_MATH = re.compile(r"^(\d+) ([+\-*x×]) (\d+) = (\d+)$")
_QUESTION_MATH = re.compile(r"(\d+)\s*([+\-*x×])\s*(\d+)")


def _norm_sym(sym):
    """رمز الضرب في سطر التفكير ممكن يكون * أو x، والرد بييتكتب بـ ×"""
    return "×" if sym in ("*", "x", "×") else sym


def _math_from_question(question):
    """
    بيقرا المسألة من السؤال نفسه (لما النموذج يغلط في قراية الأرقام زي 154).
    بيرجع (a, sym, b) بس لو فيه مسألة واحدة نضيفة وأرقامها هي كل أرقام السؤال.
    """
    matches = _QUESTION_MATH.findall(question)
    if len(matches) != 1:
        return None
    a_s, sym, b_s = matches[0]
    if _numbers(question) != sorted([a_s, b_s]):
        return None
    return int(a_s), _norm_sym(sym), int(b_s)


def _solve(a, sym, b):
    return a + b if sym == "+" else a - b if sym == "-" else a * b


def _verify_math(question, thinking, text):
    """
    فحص ثاني للحساب: بيرجع (سطر التفكير، الرد) بعد التأكد من الناتج، أو None
    لو مش قادر يقرا المسألة. الأول بيجرب قراية النموذج (سطر التفكير)، ولو
    قرا الأرقام غلط بيقري المسألة من السؤال نفسه ويحسبها بنفسه.
    """
    if not thinking.startswith(MATH_LABEL + " |"):
        return None
    t = _THINK_MATH.match(thinking.strip())
    if t and _numbers(question) == _numbers(thinking):
        a, sym, b = int(t.group(1)), _norm_sym(t.group(2)), int(t.group(3))
        r = _ANSWER_MATH.match(text.strip())
        if not r or (int(r.group(1)), _norm_sym(r.group(2)), int(r.group(3))) != (a, sym, b):
            return None
        if int(r.group(4)) == _solve(a, sym, b):
            return thinking, text                        # الناتج كان صح من الأول
        return thinking, f"{a} {sym} {b} = {_solve(a, sym, b)}"
    m = _math_from_question(question)                    # قراية النموذج غلط أو ناقصة
    if not m or not _ANSWER_MATH.match(text.strip()):    # الرد لازم يكون شكله رد حساب
        return None
    a, sym, b = m
    return f"{MATH_LABEL} | {a} {sym} {b}", f"{a} {sym} {b} = {_solve(a, sym, b)}"


# النموذج بيكتب المسافات حرف حرف، فساعات بيغلط في عدّ مسافات أول السطر في الكود
# (3 بدل 4 مثلًا) وده بيبوّظ كود بايثون. هنا بنقرّب مسافات أول كل سطر لأقرب مضاعف
# لوحدة الإزاحة بتاعة اللغة. (حل مؤقت — الحل الجذري توكن خاص للإزاحة في التدريب.)
INDENT_UNIT = {"python": 4, "c": 4, "cpp": 4, "java": 4, "sql": 4,
               "javascript": 2, "html": 2, "css": 2, "bash": 2}


def fix_code_indent(text):
    parts = text.split("```")
    for i in range(1, len(parts), 2):
        lang, sep, body = parts[i].partition("\n")
        unit = INDENT_UNIT.get(lang.strip())
        if not unit or not sep:
            continue
        lines = []
        for line in body.split("\n"):
            code = line.lstrip(" ")
            n = len(line) - len(code)
            lines.append(" " * (int(n / unit + 0.5) * unit) + code if code else "")
        parts[i] = lang + "\n" + "\n".join(lines)
    return "```".join(parts)


# ============================================================
# الجزء المشترك بين المحركين
# ============================================================
class BaseSeshat:
    engine_name = "base"

    def _setup(self, ckpt):
        self.tok = CharTokenizer(ckpt["tokens"])
        self.config = dict(ckpt["config"])
        self.step = ckpt.get("step", 0)
        self.metrics = ckpt.get("metrics", {})
        self._lock = threading.Lock()
        self._banned = (self.tok.pad_id, self.tok.unk_id, self.tok.user_id, self.tok.bot_id)
        self.memory = None
        if os.environ.get("SESHAT_MEMORY", "1") != "0":
            try:
                from memory import QuestionMemory
                self.memory = QuestionMemory()
                _trim_memory()
            except Exception as e:                 # مجلد data مش موجود مثلًا
                print(f"⚠️  ذاكرة الأسئلة مش شغالة ({e}) — النموذج هيرد لوحده")

    def num_params(self):
        raise NotImplementedError

    def _generate(self, prompt, max_new_tokens, temperature):
        """بيرجع (التوكنز الجديدة، احتمال كل توكن) — بيتقفل بـ block_size لوحده"""
        raise NotImplementedError

    def info(self):
        c = self.config
        return {
            "params": self.num_params(),
            "n_layer": c["n_layer"],
            "n_head": c["n_head"],
            "n_embd": c["n_embd"],
            "block_size": c["block_size"],
            "vocab_size": c["vocab_size"],
            "trained_steps": self.step,
            "engine": self.engine_name,
            "memory_questions": len(self.memory) if self.memory else 0,
            # خ12 — السياق الممتد (إدخال وإخراج) بالنافذة المنزلقة
            "max_context": int(c.get("max_context", 2_000_000)),
            "n_sink": int(c.get("n_sink", 8)),
            "stream_window": int(c.get("stream_window", 768)),
        }

    def _generate_long(self, prompt, max_new_tokens, temperature):
        """التوليد الطويل (خ12) — بيتعرّف في المحركين"""
        raise NotImplementedError

    # ------------------------------------------------------------
    # خ12 — السياق الممتد: مستند ضخم + سؤال عنه
    # ------------------------------------------------------------
    def reply_long(self, document, question, temperature=0.0, max_new_tokens=2000,
                   use_fallback=True):
        """
        المستند بيتدفق كله في ذاكرة النافذة المنزلقة (لحد max_context = ٢٠٠٠٠٠٠
        حرف)، والسؤال بيتحط في الآخر — نفس شكل تدريب longdoc بالظبط
        (<user> المستند \n السؤال <bot> سطر التفكير الرد <end>).
        الرد بنفس شكل رد _model_reply (response/thinking/confidence/fallback)
        + doc_chars (عدد حروف المستند اللي اتعالجت فعلًا).
        """
        t0 = time.time()
        question = (question or "").strip()
        doc = (document or "").strip()
        cap = int(self.config.get("max_context", 2_000_000))
        if len(doc) > cap:                  # نحتفظ بالآخر — السؤال والسياق الجديد
            doc = doc[-cap:]
        if not question or not doc:
            return {"response": "ابعت المستند والسؤال مع بعض."
                    if is_arabic(doc + question) else
                    "Send the document and the question together.",
                    "thinking": "", "confidence": 1.0, "fallback": True,
                    "source": "long", "doc_chars": 0, "seconds": 0.0}
        # نفس صيغة التدريب: المستند ثم سطر جديد ثم السؤال
        body = doc + "\n" + question
        ids_body = self.tok.encode(body)[-cap:]
        prompt = [self.tok.user_id] + ids_body + [self.tok.bot_id]
        with self._lock:
            ids, probs = self._generate_long(prompt, max_new_tokens, temperature)
        # نفس تحليل سطر التفكير والثقة اللي في _model_reply
        thinking, n_think, lost = "", 0, False
        et = self.tok.end_think_id
        if et is not None and et in ids:
            n_think = ids.index(et) + 1
            thinking = self.tok.decode(ids[:n_think]).strip()
        elif et is not None and self.tok.think_id in ids:
            lost = True
        confidence = 0.0 if lost else confidence_from(probs, n_think)
        text = "" if lost else fix_code_indent(self.tok.decode(ids[n_think:]).strip())
        fallback = lost or thinking in (UNKNOWN_AR, UNKNOWN_EN)
        if fallback:
            text = FALLBACK_AR if is_arabic(question) else FALLBACK_EN
        elif use_fallback and (not text or confidence < CONFIDENCE_THRESHOLD):
            text = FALLBACK_AR if is_arabic(question) else FALLBACK_EN
            fallback = True
        return {"response": text, "thinking": thinking,
                "confidence": round(float(confidence), 3), "fallback": fallback,
                "source": "long", "matched": "", "suggestion": "",
                "doc_chars": len(doc),
                "prompt_tokens": len(prompt),
                "seconds": round(time.time() - t0, 2)}

    def _max_input_chars(self):
        """أقصى طول للسؤال اللي بيدخل النموذج فعلاً — آخر جزء من الرسالة.
        الناس بيكتبوا الرسالة الطويلة والسؤال في آخرها، فبناخد الآخر مش الأول.
        الحد متظبط مع التدريب: الأمثلة بأغلفة الكلام الزيادة بتوصل لحد ~١٠٠٠ حرف
        (شوف الأغلفة الطويلة في dataset.py). (الرسالة كاملة بتتقرا في ذاكرة
        الأسئلة والمخ — الحد ده للنموذج بس، والرد ليه غرفة كبيرة:
        block_size ناقص البرومبت)."""
        return max(120, min(INPUT_CAP, self.config["block_size"] // 4))

    # ------------------------------------------------------------
    # الرد = الذاكرة (أقرب سؤال) + النموذج (الفهم والحساب)
    # ------------------------------------------------------------
    def reply(self, message, temperature=0.0, max_new_tokens=2000, use_fallback=True,
              use_memory=True):
        """
        بيرجع dict فيه: response, thinking, confidence, seconds, fallback, source, matched,
        suggestion (سؤال بيقترحه لما يكون مش متأكد).

        source بيقول الرد جه منين:
          "memory"        سؤال قريب كفاية من سؤال متخزن
          "memory+model"  الذاكرة والنموذج اتفقوا على نفس السؤال
          "model"         النموذج لوحده (الحساب والرسم، أو لما الذاكرة مش شغالة)
          "none"          مفيش رد — السؤال برّه اللي اتعلمه
        use_memory=False بيرجع رد النموذج لوحده (ده اللي evaluate.py بيقيسه).
        """
        t0 = time.time()
        # استخراج السؤال من الرسالة (لو طويلة) وبعدين التنضيف — التحية والكلام
        # الزيادة مبيوصلوش للنموذج ولا للذاكرة
        question = normalize_question(extract_question(message))[-self._max_input_chars():]
        if not question:
            return {"response": "اكتب حاجة الأول 🙂", "thinking": "", "confidence": 1.0,
                    "seconds": 0.0, "fallback": False, "source": "none", "matched": "",
                    "suggestion": ""}

        memory = self.memory if use_memory else None
        if memory is None:
            out = self._model_reply(message, question, temperature, max_new_tokens, use_fallback)
            out.update(source="none" if out["fallback"] else "model", matched="", suggestion="")
        else:
            out = self._hybrid_reply(message, question, memory, temperature, max_new_tokens)
        out["seconds"] = round(time.time() - t0, 2)
        return out

    def _hybrid_reply(self, message, question, memory, temperature, max_new_tokens):
        arabic = is_arabic(message)
        # الذاكرة بتدور على السؤال المستخرج، ولو المطابقة ضعيفة بتجرب كمان
        # آخر جزء من الرسالة كلها — عشان السؤال ما يضيعش لو الاستخراج قصه
        best_q = question
        known, unknown_score = memory.lookup(question)
        tail_full = normalize_question(message)[-INPUT_CAP:]
        if tail_full and tail_full != question:
            known2, unknown2 = memory.lookup(tail_full)
            s1 = known.score if known else unknown_score
            s2 = known2.score if known2 else unknown2
            if s2 > s1:
                known, unknown_score = known2, unknown2
                best_q = tail_full
        near = memory.search(best_q, top=1, cat="unknown")
        near_unknown = near[0] if near else None
        model = None

        def answer(match, source, agreed=False):
            # سطر التفكير بيقول seshat فهم السؤال على إنه إيه (النوع + المعنى)
            pct = round(match.score * 100)
            kind = match.intent.think.split(" | ", 1)[0]
            if arabic:
                think = f"فهمت إنك بتسأل: «{match.intent.canonical}» — نوع السؤال: {kind}"
                think += " — النموذج والذاكرة متفقين" if agreed else ""
            else:
                think = f"I understood your question as: “{match.intent.canonical}” — type: {kind}"
                think += " — the model and the memory agree" if agreed else ""
            return {"response": match.intent.answer, "thinking": think,
                    "confidence": round(match.score, 3), "fallback": False,
                    "source": source, "matched": match.intent.canonical, "suggestion": "",
                    "match": pct}

        def dont_know():
            """
            السؤال برّه معلوماته. بدل جملة واحدة محفوظة، seshat بيفهم نوع السؤال وموضوعه
            ويكوّن رد عليهم (شوف data/understanding.py). الترتيب:
              1. فهم النموذج نفسه (سطر التفكير: النوع | الموضوع) — لو الموضوع فعلًا في السؤال
              2. قواعد القوالب (understanding.detect)
              3. نوع أقرب سؤال شبهه في الذاكرة
              4. يقول إنه مفهمش ويطلب صيغة تانية
            fallback=True معناها "مفيش معلومة" — المخ (brain.py) بيكمل منها بالبحث في النت.
            """
            suggestion = ""
            if (known is not None and known.score >= MEMORY_SUGGEST
                    and known.score >= unknown_score - UNKNOWN_MARGIN
                    and not understanding.other_subject(question, known.intent.canonical)):
                suggestion = known.intent.canonical
            kind, topic, by = None, "", ""
            got = understanding.parse_think(model["thinking"]) if model else None
            if got and got[1] and f" {got[1]} " not in f" {question} ":
                got = None                     # الموضوع اللي كتبه مش موجود في السؤال → مرفوض
            found = understanding.detect(question)
            fixed = (near_unknown is not None and near_unknown.intent.kind
                     and not near_unknown.intent.topic)       # شبه سؤال من نوع ملوش موضوع
            # النموذج لسه بيغلط في النوع أكتر من القوالب (القياس في README)، فلو السؤال
            # طابق قالب صريح والنموذج قال حاجة تانية → القالب هو اللي بيمشي.
            if got and (found is None or tuple(got[:2]) == tuple(found)) \
                    and not (found is None and fixed and near_unknown.score >= 0.72
                             and got[0] != near_unknown.intent.kind):
                kind, topic, by = got[0], got[1], "model"
            if kind is None and fixed and near_unknown.score >= 0.72:
                kind, by = near_unknown.intent.kind, "memory"
            if kind is None and found:
                kind, topic, by = found[0], found[1], "rules"
            if kind is None and fixed and near_unknown.score >= 0.5:
                kind, by = near_unknown.intent.kind, "memory"
            if kind is None and understanding.is_fact_question(question):
                kind, by = "fact", "rules"     # سؤال عن معلومة مش عنده: يقولها عن السؤال نفسه
            if kind is None:
                kind, by = "unclear", "none"
            if kind == "fact":
                asked = message.strip()[:120]
                text = (f"فهمت إنك عايز معلومة: «{asked}». المعلومة دي مش ضمن اللي اتعلمته، "
                        f"ومش هألف إجابة من عندي."
                        if arabic else
                        f"I understand you want a fact: “{asked}”. That is not among the things "
                        f"I learned, and I will not make up an answer.")
            else:
                text = understanding.compose(kind, topic, arabic)
            if by == "model" and model["response"] == text:
                by = "model-text"              # الجملة نفسها طلعت من النموذج حرف حرف وسليمة
            if kind == "unclear" and by == "none":
                text = (f"مش قادر أحدد قصدك من «{message.strip()[:80]}». ممكن تقولها بكلمات تانية؟"
                        if arabic else
                        f"I cannot tell what you mean by “{message.strip()[:80]}”. Could you say it in other words?")
            name = understanding.label(kind, arabic)
            if arabic:
                think = f"نوع السؤال: {name}" + (f" — الموضوع: «{topic}»" if topic else "")
            else:
                think = f"question type: {name}" + (f" — topic: “{topic}”" if topic else "")
            if suggestion and kind in ("unclear", "about", "fact"):
                # سؤال قريب من حاجة يعرفها → يسأل "تقصد كذا؟" (والمخ بيرد بيها لو البحث مجابش حاجة)
                pct = round(known.score * 100)
                if arabic:
                    text = f"مش متأكد إني فهمت سؤالك. تقصد: «{suggestion}»؟"
                else:
                    text = f"I'm not sure I understood. Did you mean: “{suggestion}”?"
            elif kind != "unclear":
                suggestion = ""
            return {"response": text, "thinking": think,
                    "confidence": round(known.score, 3) if known else 0.0, "fallback": True,
                    "source": "understand", "matched": "", "suggestion": suggestion,
                    "kind": kind, "topic": topic, "understood_by": by,
                    "searchable": kind in understanding.SEARCHABLE or kind == "fact"
                                  or by == "none"}

        # 1) نفس السؤال موجود بالحرف → مش محتاجين النموذج
        if known is not None and known.score >= 0.999:
            return answer(known, "memory")
        if known is None and unknown_score >= 0.999:
            return dont_know()

        # 2) النموذج يفكر: السؤال ده نوعه إيه ومعناه إيه
        model = self._model_reply(message, question, temperature, max_new_tokens, True)
        thinking = model["thinking"]

        # 2.5) رسم: النموذج هو اللي بيرسم (بيكتب الرسمة حرف حرف)، وإحنا بنتأكد إن الرسمة
        #      سليمة وإنها الحاجة واللون اللي اتطلبوا. ولو الاسم مش معروف → أقرب رسمة
        #      معروفة (خ4: مفيش "مش عارف أرسم" ولا "اختار من قايمة").
        objects = drawings.find_objects(question, DRAW_MAX_OBJECTS)
        wants_drawing = any(w in question.split() for w in DRAW_WORDS)
        thinks_drawing = thinking.startswith(DRAW_LABELS)
        if objects and (thinks_drawing or wants_drawing):
            return self._draw(objects, arabic, temperature, max_new_tokens, first=model)
        if wants_drawing or (thinks_drawing and (known is None or known.score < MEMORY_STRONG)):
            return self._draw_nearest(question, arabic, temperature, max_new_tokens, first=model)

        # 3) سؤال قريب جدًا من سؤال متخزن
        #    (بس مش لو السؤالين عن حاجتين مختلفتين: "سكان الصين" مش "سكان مصر")
        if (known is not None and known.score >= MEMORY_STRONG and known.score > unknown_score
                and not understanding.other_subject(question, known.intent.canonical)):
            agreed = thinking == known.intent.think
            return answer(known, "memory+model" if agreed else "memory", agreed=agreed)

        # 4) حساب: النموذج هو اللي بيحسب، بشرط إنه يكون قرا نفس الأرقام اللي في السؤال
        if thinking.startswith(MATH_LABEL + " |"):
            if model["fallback"] or _numbers(question) != _numbers(thinking):
                return dont_know()
            model.update(source="model", matched="", suggestion="")
            return model

        # 5) تشابه متوسط، بس النموذج فهم السؤال بنفس المعنى → الاتنين متفقين
        #    (إلا لو السؤالين عن حاجتين مختلفتين: "مين اخترع التليفون" مش "مين اخترع الصفر")
        if (known is not None and thinking == known.intent.think
                and known.score >= MEMORY_AGREE
                and known.score >= unknown_score - UNKNOWN_MARGIN
                and not understanding.other_subject(question, known.intent.canonical)):
            return answer(known, "memory+model", agreed=True)

        return dont_know()

    def _cannot_draw(self, arabic):
        return {"response": draw_help(arabic),
                "thinking": "طلب رسمة لحاجة لسه ماتعلمتش أرسمها" if arabic
                else "a drawing of something I have not learned to draw",
                "confidence": 0.0, "fallback": True, "source": "none", "matched": "",
                "suggestion": ""}

    def _draw_nearest(self, question, arabic, temperature, max_new_tokens, first=None):
        """خ4: أي طلب رسم لازم يطلع منه رسمة — أقرب رسمة معروفة للمطلوب،
        والرد بيتوضح إنها أقرب حاجة مش الرسمة الأصلية. الرفض اتلغى."""
        objects, asked = drawings.nearest_objects(question, 1)
        res = self._draw(objects, arabic, temperature, max_new_tokens, first=first)
        if asked and not res.get("fallback"):
            if arabic:
                res["response"] += ("\n(مفيش رسمة باسم «%s» عندي لسه، فدي أقرب حاجة عندي للمطلوب — "
                                    "وبكمل تعلم كل يوم.)" % asked)
            else:
                res["response"] += ("\n(I don't have a drawing named '%s' yet, so this is the "
                                    "closest one I know — I keep learning every day.)" % asked)
        return res

    def draw(self, message, temperature=0.0, max_new_tokens=600):
        """
        طلب رسم صريح (المخ اتأكد إنه طلب رسم). بيرجع رد فيه الرسمة دايمًا —
        لو الاسم مش من اللي اتعلمها بيرسم أقرب حاجة وبيقول كده (خ4: مفيش رفض).
        """
        from tokenizer import normalize_question
        arabic = is_arabic(message)
        q = normalize_question(message)
        objects = drawings.find_objects(q, DRAW_MAX_OBJECTS)
        if not objects:
            return self._draw_nearest(q, arabic, temperature, max_new_tokens)
        return self._draw(objects, arabic, temperature, max_new_tokens)

    def _draw_one(self, key, code, arabic, temperature, max_new_tokens, first=None):
        """
        النموذج بيرسم حاجة واحدة. بنراجع اللي كتبه: رسمة سليمة (16 سطر مجموع كل سطر 16)،
        واللون الأساسي هو المطلوب، ومش بعيدة عن الشكل الصح بأكتر من DRAW_MAX_WRONG مربع.
        بيرجع (الشبكة، عدد المربعات الغلط، عدد السطور اللي طولها اتظبط) أو None.
        first = رد النموذج على رسالة المستخدم نفسها (لو موجود بنجربه الأول)، وبعده بنسأل
        النموذج بالصيغة الأساسية ("ارسم قطه سودا").
        """
        want = drawings.grid_for(key, code)
        tries = [first] if first is not None else []
        for attempt in range(len(tries) + 1):
            if attempt < len(tries):
                out = tries[attempt]
            else:
                q = drawings.canonical(key, code, arabic)
                out = self._model_reply(q, q, temperature, max_new_tokens, False)
            if out.get("fallback"):
                continue
            got = drawings.decode(out["response"])
            if got is None or got[1] != code:
                continue
            wrong = sum(a != b for x, y in zip(got[0], want) for a, b in zip(x, y))
            if wrong <= DRAW_MAX_WRONG:
                return got[0], wrong, got[2]
        return None

    def _draw(self, objects, arabic, temperature, max_new_tokens, first=None):
        """
        بيرسم المطلوب (رسمة أو أكتر جنب بعض). كل رسمة النموذج بيكتبها لوحدها، والمحرك
        بيحطهم جنب بعض في صورة واحدة. الرسمة اللي بتتعرض هي اللي النموذج كتبها من غير
        تصليح، وعدد المربعات اللي مش في مكانها بيتكتب في الرد.
        """
        done, failed, wrong_total, padded = [], [], 0, 0
        for i, (key, color) in enumerate(objects):
            code = color or drawings.OBJECTS[key][6]
            got = self._draw_one(key, code, arabic, temperature, max_new_tokens,
                                 first=first if (i == 0 and len(objects) == 1) else None)
            name = drawings.title(key, code, arabic)
            if got is None:
                failed.append(name)
                continue
            done.append((name, got[0]))
            wrong_total += got[1]
            padded += got[2]
        think = ("رسم | " if arabic else "drawing | ") + (" + ".join(
            drawings.title(k, c or drawings.OBJECTS[k][6], arabic) for k, c in objects))
        if not done:
            return {"response": ("حاولت أرسم " + " و".join(failed) + " بس الرسمة طلعت مش مظبوطة، "
                                 "فمش هعرضها. جرب تاني أو اطلب حاجة تانية."
                                 if arabic else
                                 "I tried to draw " + " and ".join(failed) + " but the drawing came "
                                 "out wrong, so I will not show it. Try again or ask for something else."),
                    "thinking": think, "confidence": 0.0, "fallback": True, "source": "none",
                    "matched": "", "suggestion": ""}
        names = [n for n, _ in done]
        cells = drawings.N * drawings.N * len(done)
        if arabic:
            text = "دي رسمة " + " و".join(names) + ":\n" + drawings.full_block([g for _, g in done])
            if wrong_total:
                text += f"\n(رسمتها أنا مربع مربع، وفيها {wrong_total} مربع من {cells} مش في مكانه.)"
            if failed:
                text += "\nمعرفتش أظبط رسمة " + " و".join(failed) + "، فمعرضتهاش."
        else:
            text = "Here is a " + " and a ".join(names) + ":\n" + drawings.full_block([g for _, g in done])
            if wrong_total:
                text += f"\n(I drew it cell by cell; {wrong_total} of {cells} cells are off.)"
            if failed:
                text += "\nI could not get " + " and ".join(failed) + " right, so I left it out."
        return {"response": text, "thinking": think,
                "confidence": round(1.0 - wrong_total / cells, 3), "fallback": False,
                "source": "model", "matched": "", "suggestion": "",
                "wrong_cells": wrong_total, "rows_padded": padded,
                "drawn": len(done), "not_drawn": len(failed)}

    def _model_reply(self, message, question, temperature, max_new_tokens, use_fallback):
        """رد النموذج لوحده: بيكتب سطر التفكير وبعدين الرد"""
        prompt = self.tok.encode_prompt(question)
        max_prompt = self.config["block_size"] // 2
        if len(prompt) > max_prompt:               # نسيب مكان للرد
            prompt = prompt[:max_prompt - 1] + [self.tok.bot_id]

        with self._lock:                            # طلب واحد في المرة
            ids, probs = self._generate(prompt, max_new_tokens, temperature)

        # النموذج بيكتب الأول سطر تفكير (فهم السؤال إزاي) وبعدين الرد
        thinking, n_think, lost = "", 0, False
        et = self.tok.end_think_id
        if et is not None and et in ids:
            n_think = ids.index(et) + 1
            thinking = self.tok.decode(ids[:n_think]).strip()
        elif et is not None and self.tok.think_id in ids:
            lost = True                        # بدأ يفكر ومقفلش التفكير → تاه
        confidence = 0.0 if lost else confidence_from(probs, n_think)

        text = "" if lost else fix_code_indent(self.tok.decode(ids[n_think:]).strip())
        fallback = lost or thinking in (UNKNOWN_AR, UNKNOWN_EN)   # النموذج نفسه قال مش عارف
        if fallback:
            text = FALLBACK_AR if is_arabic(message) else FALLBACK_EN
        # شبكة أمان الحساب: لو السؤال نفسه فيه مسألة واحدة نضيفة (كل أرقامه هي
        # أرقام المسألة)، المحرك بيحسبها بنفسه ويرد مضمون — سوا النموذج تاه أو
        # فهم غلط أو قال مش مفهوم، وده مهم للرسايل الطويلة اللي فيها حساب في الآخر
        m = _math_from_question(question)
        if m:
            a, sym, b = m
            return {"response": f"{a} {sym} {b} = {_solve(a, sym, b)}",
                    "thinking": f"{MATH_LABEL} | {a} {sym} {b}",
                    "confidence": 1.0, "fallback": False}
        understood = None if fallback else understanding.parse_think(thinking)
        if understood is not None:
            # النموذج فهم إن السؤال برّه معلوماته وكوّن رد من نوعه وموضوعه
            return {"response": text, "thinking": thinking,
                    "confidence": round(float(confidence), 3), "fallback": True,
                    "kind": understood[0], "topic": understood[1]}
        # فحص ثاني للحساب: لو المسألة اتقرت صح، المحرك بيحسب الناتج بنفسه وبيصلح
        # الرد — ساعتها الرد مضمون مهما كانت ثقة النموذج (لما النموذج فهمها فعلاً)
        verified = False
        if not fallback:
            fixed = _verify_math(question, thinking, text)
            if fixed is not None:
                thinking, text, verified = fixed[0], fixed[1], True
                confidence = 1.0
        if use_fallback and not fallback and not verified and (not text or confidence < CONFIDENCE_THRESHOLD):
            text = FALLBACK_AR if is_arabic(message) else FALLBACK_EN
            fallback = True
        return {"response": text, "thinking": thinking,
                "confidence": round(float(confidence), 3), "fallback": fallback}


# ============================================================
# محرك NumPy — نفس حسابات model.py بالظبط بس من غير torch
# ============================================================
_erf = np.vectorize(math.erf, otypes=[np.float64])


def _gelu(x):
    return (0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))).astype(np.float32)


def _layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


class NumpySeshat(BaseSeshat):
    engine_name = "numpy"

    def __init__(self, path=DEFAULT_MODEL):
        ckpt = load_checkpoint(path)
        self._setup(ckpt)
        w = ckpt["weights"]
        self.n_layer = self.config["n_layer"]
        self.n_head = self.config["n_head"]
        self.n_embd = self.config["n_embd"]
        self.head_dim = self.n_embd // self.n_head
        self.tok_emb = w["tok_emb.weight"]
        self.pos_emb = w["pos_emb.weight"]
        self.out_w = np.ascontiguousarray(self.tok_emb.T)       # Weight tying
        self.ln_f = (w["ln_f.gamma"], w["ln_f.beta"])
        self.layers = []
        for i in range(self.n_layer):
            p = f"blocks.{i}."
            self.layers.append({
                "ln1": (w[p + "ln1.gamma"], w[p + "ln1.beta"]),
                "ln2": (w[p + "ln2.gamma"], w[p + "ln2.beta"]),
                "c_attn": np.ascontiguousarray(w[p + "attn.c_attn.weight"].T),
                "c_proj": np.ascontiguousarray(w[p + "attn.c_proj.weight"].T),
                "fc1": np.ascontiguousarray(w[p + "ff.fc1.weight"].T),
                "fc2": np.ascontiguousarray(w[p + "ff.fc2.weight"].T),
            })
        self._n_params = sum(int(v.size) for k, v in w.items() if k != "lm_head.weight")
        self._rng = np.random.default_rng()

    def num_params(self):
        return self._n_params

    def _new_kv_cache(self):
        """خ28 — ذاكرة KV محجوزة مسبقًا بمقاعد block_size: بنكتب k وv في مكانها
        بدل concatenate ونسخ كامل لكل توكن لكل طبقة (نفس القيم بالظبط —
        مخرجات greedy متطابقة، والفرق بس إن مفيش تخصيص ونسخ متكرر)."""
        cap = int(self.config["block_size"])
        return {"k": np.zeros((self.n_head, cap, self.head_dim), dtype=np.float32),
                "v": np.zeros((self.n_head, cap, self.head_dim), dtype=np.float32),
                "n": 0}

    def _forward(self, ids, caches):
        """بيدخّل توكنز جديدة ويرجّع logits آخر توكن. caches فيها k و v بتوع اللي فات."""
        H, hd, C = self.n_head, self.head_dim, self.n_embd
        past = int(caches[0].get("n", 0)) if caches else 0
        T = len(ids)
        x = self.tok_emb[ids] + self.pos_emb[past:past + T]                 # (T, C)
        mask = None
        if T > 1:                                                            # قناع سببي
            mask = np.triu(np.ones((T, past + T), dtype=bool), k=past + 1)
        for layer, cache in zip(self.layers, caches):
            h = _layer_norm(x, *layer["ln1"])
            q, k, v = np.split(h @ layer["c_attn"], 3, axis=1)
            q = q.reshape(T, H, hd).transpose(1, 0, 2)                       # (H, T, hd)
            k = k.reshape(T, H, hd).transpose(1, 0, 2)
            v = v.reshape(T, H, hd).transpose(1, 0, 2)
            # خ28: كتابة مباشرة في الذاكرة المحجوزة + عرض على الجزء المكتوب بس
            cache["k"][:, past:past + T] = k
            cache["v"][:, past:past + T] = v
            cache["n"] = past + T
            k = cache["k"][:, :past + T]
            v = cache["v"][:, :past + T]

            att = (q @ k.transpose(0, 2, 1)) / math.sqrt(hd)                 # (H, T, past+T)
            if mask is not None:
                att[:, mask] = -np.inf
            att = np.exp(att - att.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            y = (att @ v).transpose(1, 0, 2).reshape(T, C)
            x = x + y @ layer["c_proj"]

            h = _layer_norm(x, *layer["ln2"])
            x = x + _gelu(h @ layer["fc1"]) @ layer["fc2"]
        x = _layer_norm(x[-1], *self.ln_f)
        return x @ self.out_w

    def _generate(self, prompt, max_new_tokens, temperature, top_k=10):
        block = self.config["block_size"]
        prompt = list(prompt)[-(block - 1):]
        caches = [self._new_kv_cache() for _ in self.layers]     # خ28: ذاكرة محجوزة
        step_in, out, out_probs = prompt, [], []
        banned = list(self._banned)
        for _ in range(max(0, min(max_new_tokens, block - len(prompt)))):
            logits = self._forward(step_in, caches).astype(np.float64)
            logits[banned] = -np.inf
            e = np.exp(logits - logits.max())
            probs = e / e.sum()
            if temperature <= 0:
                nxt = int(np.argmax(probs))
            else:
                scaled = logits / temperature
                if top_k and top_k < scaled.size:
                    kth = np.partition(scaled, -top_k)[-top_k]
                    scaled[scaled < kth] = -np.inf
                e = np.exp(scaled - scaled.max())
                nxt = int(self._rng.choice(scaled.size, p=e / e.sum()))
            if nxt == self.tok.end_id:
                break
            out_probs.append(float(probs[nxt]))
            out.append(nxt)
            step_in = [nxt]
        return out, out_probs

    # ------------------------------------------------------------
    # خ12 — التدفق الطويل: نفس حسابات model.py بس بـ numpy
    # ------------------------------------------------------------
    def _stream_cache(self):
        """ذاكرة التدفق: مقاعد ثابتة = المراسي + النافذة (نفس ظروف التدريب)"""
        n_sink = max(1, min(int(self.config.get("n_sink", 8)), self.config["block_size"] - 1))
        win = int(self.config.get("stream_window", 0)) or (self.config["block_size"] - n_sink)
        win = max(1, min(win, self.config["block_size"] - n_sink))
        return {"n_sink": n_sink, "window": win, "capacity": n_sink + win,
                "count": 0, "order": [],
                "k": [np.zeros((self.n_head, n_sink + win, self.head_dim), dtype=np.float32)
                      for _ in self.layers],
                "v": [np.zeros((self.n_head, n_sink + win, self.head_dim), dtype=np.float32)
                      for _ in self.layers]}

    @staticmethod
    def _slot_of(t, n_sink, window):
        return t if t < n_sink else n_sink + (t - n_sink) % window

    def _stream_plan(self, cache, T):
        """خطة كتابة T حرف: بتحدّث count والترتيب مرة واحدة (زي model.plan)"""
        t0 = cache["count"]
        cap, n_sink = cache["capacity"], cache["n_sink"]
        slots = [self._slot_of(t, n_sink, cache["window"]) for t in range(t0, t0 + T)]
        fill_end = min(t0 + T, cap)
        split = max(0, min(T, fill_end - t0))
        if split > 0:
            cache["order"] += slots[:split]
        if split < T:
            wrap = slots[split:]
            del cache["order"][n_sink:n_sink + len(wrap)]
            cache["order"] += wrap
        cache["count"] += T
        return t0, slots, fill_end

    def _stream_write(self, cache, layer, k, v, slots, fill_end, t0):
        """بيكتب k و v في مقاعدهم (k, v: (H, T, hd))"""
        split = max(0, min(len(slots), fill_end - t0))
        if split > 0:
            cache["k"][layer][:, slots[:split]] = k[:, :split]
            cache["v"][layer][:, slots[:split]] = v[:, :split]
        if split < len(slots):
            cache["k"][layer][:, slots[split:]] = k[:, split:]
            cache["v"][layer][:, slots[split:]] = v[:, split:]

    def _forward_stream(self, ids, cache):
        """خطوة تدفق: ids حروف جديدة → logits آخر حرف (نفس _long_step في model.py)"""
        H, hd, C = self.n_head, self.head_dim, self.n_embd
        T = len(ids)
        t0, slots, fill_end = self._stream_plan(cache, T)
        occ_after = min(t0 + T, cache["capacity"])
        q_pos0 = occ_after - T
        x = self.tok_emb[ids] + self.pos_emb[q_pos0:occ_after]          # (T, C)
        for layer_i, layer in enumerate(self.layers):
            h = _layer_norm(x, *layer["ln1"])
            q, k, v = np.split(h @ layer["c_attn"], 3, axis=1)
            q = q.reshape(T, H, hd).transpose(1, 0, 2)                   # (H, T, hd)
            k = k.reshape(T, H, hd).transpose(1, 0, 2)
            v = v.reshape(T, H, hd).transpose(1, 0, 2)
            self._stream_write(cache, layer_i, k.astype(np.float32), v.astype(np.float32),
                               slots, fill_end, t0)
            order = cache["order"]
            ko = cache["k"][layer_i][:, order]                           # (H, occ, hd)
            vo = cache["v"][layer_i][:, order]
            occ = ko.shape[1]
            att = (q @ ko.transpose(0, 2, 1)) / math.sqrt(hd)            # (H, T, occ)
            mask = np.triu(np.ones((T, occ), dtype=bool), k=q_pos0 + 1)  # قناع سببي
            att[:, mask] = -np.inf
            att = np.exp(att - att.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            y = (att @ vo).transpose(1, 0, 2).reshape(T, C)
            x = x + y @ layer["c_proj"]
            h = _layer_norm(x, *layer["ln2"])
            x = x + _gelu(h @ layer["fc1"]) @ layer["fc2"]
        x = _layer_norm(x[-1], *self.ln_f)
        return x @ self.out_w

    def _generate_long(self, prompt, max_new_tokens, temperature, top_k=10, chunk=512):
        """
        التوليد الطويل (خ12) بفرعين زي معمارية StreamingLLM:
          • البرومبت ≤ block_size → المسار الكلاسيكي بنفس انتباه التدريب بالظبط
            (ده حال مستندات التدريب كلها — طولها ≤ ٢١٠٠ حرف).
          • أطول من كده → ذاكرة النافذة المنزلقة + المراسي: تدفق بذاكرة ثابتة
            لحد max_context (٢م حرف) — النافذة بتحتفظ بآخر stream_window حرف.
        """
        block = self.config["block_size"]
        prompt = list(prompt)
        cap = int(self.config.get("max_context", 2_000_000))
        prompt = prompt[:cap]
        if not prompt:
            return [], []
        # هامش صغير للردود القصيرة في المسار الكلاسيكي
        if len(prompt) + max(0, min(max_new_tokens, 128)) <= block:
            return self._generate(prompt, max_new_tokens, temperature)
        cache = self._stream_cache()
        max_new = max(0, min(max_new_tokens, cap - len(prompt)))
        chunk = max(1, min(chunk, cache["window"]))
        banned = list(self._banned)
        out, out_probs = [], []
        logits = None
        for s in range(0, len(prompt), chunk):
            logits = self._forward_stream(prompt[s:s + chunk], cache)
        for _ in range(max_new):
            logits = logits.astype(np.float64)
            logits[banned] = -np.inf
            e = np.exp(logits - logits.max())
            probs = e / e.sum()
            if temperature <= 0:
                nxt = int(np.argmax(probs))
            else:
                scaled = logits / temperature
                if top_k and top_k < scaled.size:
                    kth = np.partition(scaled, -top_k)[-top_k]
                    scaled[scaled < kth] = -np.inf
                e = np.exp(scaled - scaled.max())
                nxt = int(self._rng.choice(scaled.size, p=e / e.sum()))
            if nxt == self.tok.end_id:
                break
            out_probs.append(float(probs[nxt]))
            out.append(nxt)
            logits = self._forward_stream([nxt], cache)
        return out, out_probs


# ============================================================
# محرك torch — نفس الملف، بس بيشغّل model.py (ينفع GPU)
# ============================================================
class TorchSeshat(BaseSeshat):
    engine_name = "torch"

    def __init__(self, path=DEFAULT_MODEL, device=None):
        import torch
        from model import Config, MiniGPT

        ckpt = load_checkpoint(path)
        self._setup(ckpt)
        self.model = MiniGPT(Config(**self.config))
        self.model.load_state_dict({k: torch.from_numpy(np.array(v))
                                    for k, v in ckpt["weights"].items()})
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.engine_name = f"torch-{self.device}"

    def num_params(self):
        return self.model.num_params()

    def _generate(self, prompt, max_new_tokens, temperature):
        return self.model.generate(prompt, max_new_tokens=max_new_tokens,
                                   temperature=temperature, end_id=self.tok.end_id,
                                   banned_ids=self._banned)

    def _generate_long(self, prompt, max_new_tokens, temperature):
        """نفس الفرعين: ≤ block_size كلاسيكي، أطول → تدفق بالنافذة المنزلقة"""
        prompt = list(prompt)
        block = self.config["block_size"]
        cap = int(self.config.get("max_context", 2_000_000))
        if len(prompt) + max(0, min(max_new_tokens, 128)) <= block:
            return self._generate(prompt, max_new_tokens, temperature)
        return self.model.generate_long(prompt[:cap], max_new_tokens=max_new_tokens,
                                        temperature=temperature, end_id=self.tok.end_id,
                                        banned_ids=self._banned)


def Seshat(path=DEFAULT_MODEL, engine=None):
    """بيحمّل النموذج بالمحرك المناسب (NumPy افتراضيًا)"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ملف النموذج مش موجود: {path}\nشغّل الأول:  python train_model.py")
    engine = (engine or os.environ.get("SESHAT_ENGINE", "numpy")).lower()
    if engine == "torch":
        return TorchSeshat(path)
    return NumpySeshat(path)
