"""
seshat.chet — Tokenizer على مستوى الحروف، مكتوب من الصفر

الفرق عن النسخة القديمة:
- المفردات ثابتة وواسعة (عربي + إنجليزي + أرقام + رموز برمجة) مش مأخوذة من
  بيانات التدريب بس، فالنموذج يقدر يقرا ويكتب أي حرف منها.
- فيه توكنز خاصة: بداية كلام المستخدم، بداية رد النموذج، ونهاية الرد.
- الحرف الغريب مبيتمسحش في صمت، بيتحول لتوكن <unk>.
- فيه تنضيف (normalize) لسؤال المستخدم عشان اختلاف الكتابة ميلخبطش النموذج.
"""

import re

PAD, UNK, USER, BOT, END = "<pad>", "<unk>", "<user>", "<bot>", "<end>"
SPECIALS = [PAD, UNK, USER, BOT, END]


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
        self.tokens = list(tokens) if tokens else SPECIALS + _base_chars()
        self.stoi = {t: i for i, t in enumerate(self.tokens)}
        self.pad_id = self.stoi[PAD]
        self.unk_id = self.stoi[UNK]
        self.user_id = self.stoi[USER]
        self.bot_id = self.stoi[BOT]
        self.end_id = self.stoi[END]

    @property
    def vocab_size(self):
        return len(self.tokens)

    def add_chars_from(self, text):
        """يضيف أي حرف جديد موجود في بيانات التدريب ومش في المفردات"""
        for ch in sorted(set(text)):
            if ch not in self.stoi:
                self.stoi[ch] = len(self.tokens)
                self.tokens.append(ch)

    def encode(self, text):
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, ids):
        out = []
        for i in ids:
            tok = self.tokens[i] if 0 <= i < len(self.tokens) else ""
            if tok not in SPECIALS:
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
