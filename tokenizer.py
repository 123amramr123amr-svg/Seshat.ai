"""
seshat.chet — Tokenizer على مستوى الحروف، مكتوب من الصفر

الفرق عن النسخة القديمة:
- المفردات ثابتة وواسعة (عربي + إنجليزي + أرقام + رموز برمجة) مش مأخوذة من
  بيانات التدريب بس، فالنموذج يقدر يقرا ويكتب أي حرف منها.
- فيه توكنز خاصة: بداية كلام المستخدم، بداية رد النموذج، ونهاية الرد.
- فيه توكنز للتفكير: النموذج بيكتب الأول هو فهم السؤال إزاي، وبعدين يرد.
- مسافات الإزاحة في الكود (4 مسافات / مسافتين) كل واحدة توكن واحد، عشان النموذج
  ميغلطش في عدّ المسافات والكود يبقى أقصر.
- الحرف الغريب مبيتمسحش في صمت، بيتحول لتوكن <unk>.
- فيه تنضيف (normalize) لسؤال المستخدم عشان اختلاف الكتابة ميلخبطش النموذج.
"""

import re

PAD, UNK, USER, BOT, END = "<pad>", "<unk>", "<user>", "<bot>", "<end>"
SPECIALS = [PAD, UNK, USER, BOT, END]

# توكنز اتضافت بعدين — بتتحط في آخر المفردات عشان أرقام التوكنز القديمة متتغيرش
THINK, END_THINK = "<think>", "</think>"
INDENT4, INDENT2 = " " * 4, " " * 2
EXTRA_TOKENS = [THINK, END_THINK, INDENT4, INDENT2]
HIDDEN = set(SPECIALS + [THINK, END_THINK])          # مبتظهرش في النص


def _base_chars():
    chars = ["\n", "\t"]
    chars += [chr(c) for c in range(32, 127)]            # ASCII: إنجليزي + أرقام + رموز الكود
    chars += [chr(c) for c in range(0x0621, 0x063B)]      # ء .. غ
    chars += [chr(c) for c in range(0x0641, 0x064B)]      # ف .. ي
    chars += list("،؛؟")                                   # ترقيم عربي
    chars += [chr(c) for c in range(0x0660, 0x066A)]      # ٠ .. ٩
    chars += list("×÷«»…—–°")                              # رموز إضافية
    return chars


class CharTokenizer:
    def __init__(self, tokens=None):
        self.tokens = list(tokens) if tokens else SPECIALS + _base_chars() + EXTRA_TOKENS
        self._index()

    def _index(self):
        self.stoi = {t: i for i, t in enumerate(self.tokens)}
        self.pad_id = self.stoi[PAD]
        self.unk_id = self.stoi[UNK]
        self.user_id = self.stoi[USER]
        self.bot_id = self.stoi[BOT]
        self.end_id = self.stoi[END]
        # النماذج القديمة مفيهاش التوكنز دي → None
        self.think_id = self.stoi.get(THINK)
        self.end_think_id = self.stoi.get(END_THINK)
        self._i4 = self.stoi.get(INDENT4)
        self._i2 = self.stoi.get(INDENT2)

    def add_extra_tokens(self):
        """بيضيف توكنز التفكير والإزاحة لنموذج قديم مكانتش فيه. بيرجع عدد اللي اتضاف."""
        missing = [t for t in EXTRA_TOKENS if t not in self.stoi]
        self.tokens += missing
        self._index()
        return len(missing)

    @property
    def vocab_size(self):
        return len(self.tokens)

    def add_chars_from(self, text):
        """يضيف أي حرف جديد موجود في بيانات التدريب ومش في المفردات"""
        added = 0
        for ch in sorted(set(text)):
            if ch not in self.stoi:
                self.stoi[ch] = len(self.tokens)
                self.tokens.append(ch)
                added += 1
        return added

    def encode(self, text):
        ids, i, n, stoi = [], 0, len(text), self.stoi
        i4, i2 = self._i4, self._i2
        while i < n:
            ch = text[i]
            if ch == " " and i4 is not None:
                if text.startswith(INDENT4, i):
                    ids.append(i4)
                    i += 4
                    continue
                if text.startswith(INDENT2, i):
                    ids.append(i2)
                    i += 2
                    continue
            ids.append(stoi.get(ch, self.unk_id))
            i += 1
        return ids

    def decode(self, ids):
        out = []
        for i in ids:
            tok = self.tokens[i] if 0 <= i < len(self.tokens) else ""
            if tok not in HIDDEN:
                out.append(tok)
        return "".join(out)

    def encode_prompt(self, question):
        """<user> السؤال <bot>  — النموذج بيكمل من هنا لحد <end>"""
        return [self.user_id] + self.encode(question) + [self.bot_id]


# ============================================================
# تنضيف سؤال المستخدم
# ============================================================
_TASHKEEL = re.compile(r"[\u064B-\u0652\u0670\u0640]")     # التشكيل + التطويل
_DROP = re.compile(r"[?!؟،؛\"'`,]")
_REPEAT = re.compile(r"([^\W\d_])\1{2,}")                   # حرف متكرر 3 مرات أو أكتر
_SPACES = re.compile(r"\s+")
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize_question(text):
    """
    بيوحّد شكل السؤال قبل ما يدخل النموذج (وقت التدريب ووقت التشغيل):
    أ إ آ → ا   |   ى → ي   |   ة → ه   |   حروف صغيرة   |   من غير تشكيل ولا علامات استفهام
    """
    t = text.strip().lower()
    t = _TASHKEEL.sub("", t)
    t = t.translate(_AR_DIGITS)
    t = re.sub("[أإآٱ]", "ا", t)
    t = t.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    t = _DROP.sub("", t)
    t = _REPEAT.sub(r"\1", t)
    t = _SPACES.sub(" ", t).strip()
    t = t.rstrip(". ")
    return t


def is_arabic(text):
    return any("\u0600" <= ch <= "\u06FF" for ch in text)
