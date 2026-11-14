"""
seshat.chet — تجهيز بيانات التدريب

مصادر البيانات (كلها في فولدر data/):
  1. ملفات .txt مكتوبة بالإيد (محادثات ومعلومات عربي/إنجليزي)
  2. code_snippets.py — أكواد، وبيتولّد لكل كود أسئلة عربي وإنجليزي
  3. wordlists.py — ترجمة كلمات + عواصم
  4. unknown_questions.py — أسئلة برّه اللي اتعلمه (عشان يقول "مش عارف")
  5. عمليات حسابية بتتولّد هنا

التفكير: كل رد بيتدرب عليه النموذج بيبدأ بسطر تفكير فيه نوع السؤال + السؤال
بصيغته الأساسية. يعني النموذج بيتعلم الأول "السؤال ده معناه إيه" وبعدين يرد.

كل "Intent" = مجموعة صيغ لنفس السؤال + رد واحد.
وقت التدريب بنختار صيغة عشوائية ونضيف عليها تغييرات بسيطة (كلمة زيادة، غلطة كتابة)
عشان النموذج يتعلم المعنى مش شكل الجملة بالحرف.
"""

import glob
import hashlib
import os
import random
import sys

# torch بيتستورد جوه دوال التدريب بس — عشان التشغيل (والذاكرة) يفضلوا من غير torch

from engine import FALLBACK_AR, FALLBACK_EN, UNKNOWN_AR, UNKNOWN_EN
from tokenizer import CharTokenizer, normalize_question, is_arabic

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# نسبة ظهور كل نوع في التدريب (لكل رد). المحادثة والمعلومات ردودها طويلة ومختلفة
# عن بعض فمحتاجة تتكرر أكتر، والحساب والترجمة شبه بعض فبيتعلموا أسرع.
CAT_WEIGHTS = {
    "chat": 4.0,
    "knowledge": 3.0,
    "programming": 3.0,
    "code": 1.2,
    "translate": 0.5,
    "capital": 0.8,
    "math": 0.3,
    "draw": 8.0,          # الرسومات طويلة ولازم تتكتب صح رمز رمز
    "unknown": 120.0,     # ردّين بس (عربي/إنجليزي) لكن وراهم مئات الأسئلة
}

# أسئلة الهوية (الاسم والمطوّر) لازم تتحفظ كويس جدًا → وزنها بيتضرب في الرقم ده
IDENTITY_BOOST = 10.0
IDENTITY_QUESTIONS = {
    "اسمك ايه", "مين طورك", "انت مين", "مين عمرو احمد",
    "what is your name", "who developed you", "who are you", "who is amr ahmed",
}

# اسم النوع اللي بيتكتب في سطر التفكير
THINK_LABELS = {
    "chat": ("محادثة", "chat"),
    "knowledge": ("معلومة", "fact"),
    "programming": ("برمجة", "programming"),
    "code": ("كود", "code"),
    "translate": ("ترجمة", "translation"),
    "capital": ("عاصمة", "capital"),
    "math": ("حساب", "math"),
    "draw": ("رسم", "drawing"),
}


class Intent:
    __slots__ = ("questions", "answer", "cat", "weight", "aug", "typo", "holdout",
                 "canonical", "think")

    def __init__(self, questions, answer, cat, weight=1.0, aug=True, typo=True):
        self.canonical = questions[0].strip() if questions else ""   # الصيغة الأساسية للسؤال
        self.think = ""
        seen, qs = set(), []
        for q in questions:
            q = normalize_question(q)
            if q and q not in seen:
                seen.add(q)
                qs.append(q)
        self.questions = qs
        self.answer = answer.strip()
        self.cat = cat
        self.weight = weight
        self.aug = aug
        self.typo = typo
        self.holdout = []       # صيغ متشالة من التدريب عشان نختبر بيها


# ============================================================
# 1) الملفات المكتوبة بالإيد
# ============================================================
Q_PREFIXES = ("س:", "Q:")
A_PREFIXES = ("ج:", "A:")


def load_dialog_file(path, cat):
    intents, questions, answer_lines = [], None, None

    def flush():
        if questions and answer_lines is not None:
            answer = "\n".join(answer_lines).strip()
            if answer:
                intents.append(Intent(questions, answer, cat))

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("#") and answer_lines is None:
                continue
            if line.startswith(Q_PREFIXES):
                if answer_lines is not None:      # بلوك جديد
                    flush()
                    questions, answer_lines = None, None
                parts = [p.strip() for p in line[2:].split("|")]
                questions = (questions or []) + [p for p in parts if p]
            elif line.startswith(A_PREFIXES):
                answer_lines = [line[2:].strip()]
            elif answer_lines is not None:
                if line.startswith("#"):
                    continue
                answer_lines.append(line)
    flush()
    return intents


# ============================================================
# 2) الأكواد
# ============================================================
LANGS = {
    # المفتاح: (الاسم في الرد العربي، الاسم في الرد الإنجليزي، وسم الكود، أسماء في السؤال العربي، أسماء في السؤال الإنجليزي)
    "python":     ("بايثون", "Python", "python", ["بايثون", "python"], ["python"]),
    "javascript": ("جافاسكريبت", "JavaScript", "javascript",
                   ["جافاسكريبت", "جافا سكريبت", "javascript", "js"], ["javascript", "js"]),
    "c":          ("سي", "C", "c", ["سي", "c"], ["c"]),
    "c++":        ("سي بلس بلس", "C++", "cpp", ["سي بلس بلس", "c++"], ["c++", "cpp"]),
    "java":       ("جافا", "Java", "java", ["جافا", "java"], ["java"]),
    "sql":        ("SQL", "SQL", "sql", ["sql"], ["sql"]),
    "html":       ("HTML", "HTML", "html", ["html"], ["html"]),
    "css":        ("CSS", "CSS", "css", ["css"], ["css"]),
    "bash":       ("باش", "Bash", "bash", ["باش", "bash"], ["bash"]),
}

AR_CODE_TEMPLATES = [
    "اكتب كود {L} {t}",
    "عايز كود {L} {t}",
    "اكتبلي برنامج {L} {t}",
    "كود {L} {t}",
    "ممكن كود {L} {t}",
    "اعمل كود {t} بلغه {L}",
    "اكتب برنامج {t} ب {L}",
]
AR_CODE_TEMPLATES_NOLANG = ["اكتب كود {t}", "عايز كود {t}", "اكتبلي برنامج {t}", "اعمل برنامج {t}"]

EN_CODE_TEMPLATES = [
    "write {L} code to {t}",
    "write a {L} program to {t}",
    "{L} code to {t}",
    "how to {t} in {L}",
    "how do i {t} in {L}",
    "{t} in {L}",
    "write a {L} function to {t}",
    "give me {L} code to {t}",
]
EN_CODE_TEMPLATES_NOLANG = ["write code to {t}", "write a program to {t}", "how to {t}", "code to {t}"]


def load_code_intents():
    sys.path.insert(0, DATA_DIR)
    from code_snippets import SNIPPETS
    intents = []
    for lang, ar_task, en_task, code in SNIPPETS:
        ar_name, en_name, fence, ar_q_names, en_q_names = LANGS[lang]
        block = f"```{fence}\n{code}\n```"

        ar_qs = [tpl.format(L=n, t=ar_task) for tpl in AR_CODE_TEMPLATES for n in ar_q_names]
        en_qs = [tpl.format(L=n, t=en_task) for tpl in EN_CODE_TEMPLATES for n in en_q_names]
        if lang == "python":            # لو المستخدم مقالش لغة، بنرد ببايثون
            ar_qs += [tpl.format(t=ar_task) for tpl in AR_CODE_TEMPLATES_NOLANG]
            en_qs += [tpl.format(t=en_task) for tpl in EN_CODE_TEMPLATES_NOLANG]

        intents.append(Intent(ar_qs, f"ده كود {ar_name} {ar_task}:\n{block}", "code", weight=2.0))
        intents.append(Intent(en_qs, f"Here is {en_name} code to {en_task}:\n{block}", "code", weight=2.0))
    return intents


# ============================================================
# 3) الترجمة والعواصم
# ============================================================
def load_wordlist_intents():
    sys.path.insert(0, DATA_DIR)
    from wordlists import WORDS, CAPITALS, COUNTRY_ALIASES
    intents = []
    for en, ar in WORDS:
        intents.append(Intent(
            [f"يعني ايه {en}", f"{en} يعني ايه", f"معنى {en}", f"ترجم {en}",
             f"{en} بالعربي", f"ترجمه {en}", f"معنى كلمه {en}", f"يعني ايه {en} بالعربي"],
            f"{en} يعني «{ar}» بالعربي.", "translate", weight=0.6, typo=False))
        intents.append(Intent(
            [f"{ar} بالانجليزي", f"يعني ايه {ar} بالانجليزي", f"ترجم {ar} للانجليزي",
             f"ترجم {ar}", f"{ar} بالانجليزي ايه", f"كلمه {ar} بالانجليزي"],
            f"«{ar}» بالإنجليزي: {en}", "translate", weight=0.6, typo=False))
        intents.append(Intent(
            [f"translate {en} to arabic", f"{en} in arabic", f"how do you say {en} in arabic",
             f"what is {en} in arabic", f"translate {en}", f"what does {en} mean in arabic"],
            f"\"{en}\" in Arabic is \"{ar}\".", "translate", weight=0.6, typo=False))
        intents.append(Intent(
            [f"translate {ar} to english", f"{ar} in english", f"what does {ar} mean",
             f"what is {ar} in english", f"how do you say {ar} in english"],
            f"\"{ar}\" in English is \"{en}\".", "translate", weight=0.6, typo=False))

    for ar_c, ar_cap, en_c, en_cap in CAPITALS:
        intents.append(Intent(
            [f"عاصمه {ar_c} ايه", f"ايه عاصمه {ar_c}", f"ما هي عاصمه {ar_c}",
             f"عاصمه {ar_c}", f"ايه هي عاصمه {ar_c}", f"عاصمه {ar_c} اسمها ايه"],
            f"عاصمة {ar_c} هي {ar_cap}.", "capital", weight=0.8, typo=False))
        names = COUNTRY_ALIASES.get(en_c, [en_c])
        qs = []
        for n in names:
            qs += [f"what is the capital of {n}", f"capital of {n}", f"{n} capital",
                   f"what's the capital of {n}", f"the capital of {n}"]
        intents.append(Intent(qs, f"The capital of {en_c} is {en_cap}.", "capital",
                              weight=0.8, typo=False))
    return intents


# ============================================================
# 4) الحساب
# ============================================================
def make_math_intents():
    intents = []

    def mk(a, b, sym, res, ar_words, en_words, syms):
        qs = []
        for s in syms:
            qs += [f"{a} {s} {b}", f"{a}{s}{b}", f"{a} {s} {b} كام", f"كام {a} {s} {b}",
                   f"ناتج {a} {s} {b}", f"what is {a} {s} {b}", f"{a} {s} {b} =",
                   f"calculate {a} {s} {b}", f"احسب {a} {s} {b}"]
        for w in ar_words:
            qs += [f"{a} {w} {b}", f"{a} {w} {b} كام", f"كام {a} {w} {b}"]
        for w in en_words:
            qs += [f"{a} {w} {b}", f"what is {a} {w} {b}"]
        intents.append(Intent(qs, f"{a} {sym} {b} = {res}", "math", weight=0.35,
                              aug=True, typo=False))

    for a in range(0, 21):
        for b in range(0, 21):
            mk(a, b, "+", a + b, ["زائد", "زايد"], ["plus"], ["+"])
            if a >= b:
                mk(a, b, "-", a - b, ["ناقص"], ["minus"], ["-"])
    for a in range(0, 13):
        for b in range(0, 13):
            mk(a, b, "×", a * b, ["في", "ضرب"], ["times", "multiplied by"], ["*", "x", "×"])
    return intents


# ============================================================
# 5) الرسم — شبكة رموز 12 × 12
# ============================================================
def load_draw_intents(held_out=False):
    """held_out=True: التركيبات اللي النموذج مبيتدربش عليها (للاختبار بس)"""
    sys.path.insert(0, DATA_DIR)
    import drawings
    return [Intent(qs, answer, "draw", aug=False, typo=False)
            for qs, answer in drawings.build(include_held_out=held_out)]


# ============================================================
# 6) أسئلة برّه اللي اتعلمه
# ============================================================
def make_unknown_intents():
    sys.path.insert(0, DATA_DIR)
    import unknown_questions as uq
    ar_qs, en_qs = uq.build()
    out = []
    for qs, answer, think, test in ((ar_qs, FALLBACK_AR, UNKNOWN_AR, uq.AR_TEST),
                                    (en_qs, FALLBACK_EN, UNKNOWN_EN, uq.EN_TEST)):
        it = Intent(qs, answer, "unknown")
        it.think = think
        test = {normalize_question(q) for q in test}
        it.questions = [q for q in it.questions if q not in test]
        it.holdout = sorted(test)              # للاختبار بس، مبيتدربش عليها
        out.append(it)
    return out


# ============================================================
# تجميع كل البيانات
# ============================================================
def _stable_hash(text):
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def build_intents(holdout=True, verbose=True):
    intents = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.txt"))):
        name = os.path.basename(path)
        cat = ("programming" if "programming" in name
               else "knowledge" if "knowledge" in name else "chat")
        got = load_dialog_file(path, cat)
        intents += got
        if verbose:
            print(f"  {name:<22} {len(got):>5} رد")
    for label, loader in (("أكواد", load_code_intents), ("ترجمة وعواصم", load_wordlist_intents),
                          ("حساب", make_math_intents), ("رسومات", load_draw_intents),
                          ("برّه اللي اتعلمه", make_unknown_intents)):
        got = loader()
        intents += got
        if verbose:
            print(f"  {label:<22} {len(got):>5} رد")

    # لو نفس السؤال موجود في أكتر من Intent، أول واحد بس اللي ياخده
    owner, dup = {}, 0
    for it in intents:
        keep = []
        for q in it.questions:
            if q in owner:
                dup += 1
            else:
                owner[q] = it
                keep.append(q)
        it.questions = keep
    intents = [it for it in intents if it.questions]
    for it in intents:
        it.weight = CAT_WEIGHTS.get(it.cat, 1.0)
        if normalize_question(it.canonical) in IDENTITY_QUESTIONS:
            it.weight *= IDENTITY_BOOST
        if not it.think:                       # سطر التفكير: النوع | السؤال بصيغته الأساسية
            ar_label, en_label = THINK_LABELS[it.cat]
            label = ar_label if is_arabic(it.canonical) or it.cat == "math" else en_label
            it.think = f"{label} | {it.canonical}"

    # نشيل صيغة واحدة من بعض الأسئلة عشان نختبر بيها النموذج على صيغ ماشافهاش
    if holdout:
        for it in intents:
            if it.cat in ("chat", "knowledge", "programming", "code") and len(it.questions) >= 4:
                h = _stable_hash(it.answer)
                if h % 3 == 0:
                    it.holdout = [it.questions.pop(h % len(it.questions))]

    if verbose:
        nq = sum(len(it.questions) for it in intents)
        nh = sum(len(it.holdout) for it in intents)
        print(f"  المجموع: {len(intents)} رد مختلف | {nq} صيغة سؤال للتدريب | {nh} صيغة للاختبار"
              + (f" | {dup} سؤال مكرر اتشال" if dup else ""))
    return intents


# ============================================================
# تغييرات عشوائية على السؤال (Augmentation)
# ============================================================
AR_PRE = ["لو سمحت", "ممكن", "طيب", "يا سشات", "بص", "معلش", "من فضلك", "قولي"]
AR_SUF = ["لو سمحت", "يا سشات", "بسرعه", "من فضلك", "يا باشا", "بقى"]
EN_PRE = ["please", "hey", "ok", "hi seshat", "so", "seshat", "can you tell me", "well"]
EN_SUF = ["please", "thanks", "seshat", "now", "pls"]


def _typo(q, rng):
    idxs = [i for i, ch in enumerate(q) if ch.isalpha()]
    if len(idxs) < 5:
        return q
    i = rng.choice(idxs)
    kind = rng.random()
    if kind < 0.4:                                     # حذف حرف
        return q[:i] + q[i + 1:]
    if kind < 0.7 and i + 1 < len(q) and q[i + 1].isalpha():   # تبديل حرفين
        return q[:i] + q[i + 1] + q[i] + q[i + 2:]
    return q[:i] + q[i] + q[i:]                        # تكرار حرف


def augment(q, intent, rng):
    if not intent.aug:
        return q
    ar = is_arabic(q)
    r = rng.random()
    if r < 0.12:
        q = rng.choice(AR_PRE if ar else EN_PRE) + " " + q
    elif r < 0.20:
        q = q + " " + rng.choice(AR_SUF if ar else EN_SUF)
    if intent.typo and rng.random() < 0.12:
        q = _typo(q, rng)
    return q


# ============================================================
# تحويل لتنسورات
# ============================================================
class Dataset:
    def __init__(self, tokenizer=None, holdout=True, seed=1234, verbose=True):
        self.intents = build_intents(holdout=holdout, verbose=verbose)
        self.tok = tokenizer or CharTokenizer()
        self.tok.add_chars_from("".join(it.answer + it.think + "".join(it.questions + it.holdout)
                                        for it in self.intents))
        self.rng = random.Random(seed)
        self.weights = [it.weight for it in self.intents]
        self._answer_ids = {}
        self._pad_limit = 512
        self.max_answer_len = max(len(it.answer) for it in self.intents)

    def encode_pair(self, question, intent, block_size):
        """بيرجع (x, y): الخسارة بتتحسب على حروف الرد بس، مش على السؤال"""
        a = self._answer_ids.get(id(intent))
        if a is None:
            # <think> النوع | السؤال الأساسي </think> الرد <end>
            a = ([self.tok.think_id] + self.tok.encode(intent.think) + [self.tok.end_think_id]
                 + self.tok.encode(intent.answer) + [self.tok.end_id])
            self._answer_ids[id(intent)] = a
        max_q = block_size - len(a) - 1
        q = self.tok.encode(question)[:max(1, max_q - 1)]
        seq = [self.tok.user_id] + q + [self.tok.bot_id] + a
        seq = seq[:block_size + 1]
        x = seq[:-1]
        y = [-1] * (len(q) + 1) + seq[len(q) + 2:]
        return x, y

    def _collate(self, pairs, device):
        import torch
        T = max(len(x) for x, _ in pairs)
        T = min(-(-T // 16) * 16, max(T, self._pad_limit))   # تقريب الطول لمضاعفات 16 (أرحم للذاكرة)
        X = torch.full((len(pairs), T), self.tok.pad_id, dtype=torch.long)
        Y = torch.full((len(pairs), T), -1, dtype=torch.long)
        for i, (x, y) in enumerate(pairs):
            X[i, :len(x)] = torch.tensor(x)
            Y[i, :len(y)] = torch.tensor(y)
        return X.to(device), Y.to(device)

    def batches(self, batch_size, block_size, device, max_cells=1_000_000, pool=16):
        """
        مولّد لا نهائي للباتشات. بنجمع الأمثلة اللي أطوالها قريبة من بعض في نفس الباتش
        عشان منضيعش وقت على الحشو (الأكواد طويلة والمحادثات قصيرة).

        max_cells = أقصى (عدد الأمثلة × الطول²) في الباتش الواحد. مصفوفة الانتباه حجمها
        بيزيد مع مربع الطول، فالأمثلة الطويلة لازم تتاخد بعدد أقل عشان الرام متتمليش.
        """
        self._pad_limit = block_size
        while True:
            chosen = self.rng.choices(self.intents, weights=self.weights, k=batch_size * pool)
            pairs = []
            for it in chosen:
                q = augment(self.rng.choice(it.questions), it, self.rng)
                pairs.append(self.encode_pair(q, it, block_size))
            pairs.sort(key=lambda p: len(p[0]))
            chunks, cur = [], []
            for p in pairs:
                T = len(p[0])
                if cur and (len(cur) >= batch_size or (len(cur) + 1) * T * T > max_cells):
                    chunks.append(cur)
                    cur = []
                cur.append(p)
            if cur:
                chunks.append(cur)
            self.rng.shuffle(chunks)
            for chunk in chunks:
                yield self._collate(chunk, device)

    def eval_sets(self, n_train=200, seed=7):
        """(أسئلة من التدريب، أسئلة ماتشافتش) — كل عنصر (سؤال، Intent)"""
        rng = random.Random(seed)
        pool = [it for it in self.intents if it.questions]
        seen = [(rng.choice(it.questions), it) for it in rng.sample(pool, min(n_train, len(pool)))]
        unseen = [(q, it) for it in self.intents for q in it.holdout]
        return seen, unseen

    def evaluate(self, model, pairs, device, batch_size=8):
        """
        بيرجع (الخسارة، نسبة الردود الصح بالحرف).
        الرد يعتبر صح لو أعلى احتمال في كل موضع هو الحرف المطلوب —
        وده بالظبط اللي هيطلع لو ولّدنا الرد بـ temperature = 0.
        """
        import torch
        if not pairs:
            return 0.0, 0.0
        model.eval()
        enc = [self.encode_pair(q, it, model.cfg.block_size) for q, it in pairs]
        order = sorted(range(len(enc)), key=lambda i: len(enc[i][0]))
        loss_sum, n_tok, exact = 0.0, 0, 0
        with torch.no_grad():
            for s in range(0, len(order), batch_size):
                X, Y = self._collate([enc[i] for i in order[s:s + batch_size]], device)
                logits, _ = model(X)
                mask = Y != -1
                lp = torch.log_softmax(logits, dim=-1)
                tok_lp = lp.gather(-1, Y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
                loss_sum += float(-(tok_lp * mask).sum())
                n_tok += int(mask.sum())
                correct = (logits.argmax(-1) == Y) | ~mask
                exact += int(correct.all(dim=1).sum())
        model.train()
        return loss_sum / max(1, n_tok), exact / len(pairs)

if __name__ == "__main__":
    ds = Dataset()
    print(f"حجم المفردات: {ds.tok.vocab_size} | أطول رد: {ds.max_answer_len} حرف")
    by_cat = {}
    for it in ds.intents:
        by_cat[it.cat] = by_cat.get(it.cat, 0) + 1
    print("التقسيم:", by_cat)
    for it in ds.rng.sample(ds.intents, 6):
        q = augment(ds.rng.choice(it.questions), it, ds.rng)
        print("—" * 30, f"\nس: {q}\nج: {it.answer}")
