"""
ذاكرة الأسئلة — بتدوّر على أقرب سؤال النموذج يعرفه
======================================================
مكتوبة من الصفر بـ NumPy (من غير أي مكتبة بحث ولا نموذج جاهز).

الفكرة: كل سؤال في بيانات التدريب بيتحول لـ "بصمة": كلماته + مقاطع من 3 حروف
جوه كل كلمة. كل مقطع ليه وزن — المقطع النادر (زي "هرم") وزنه عالي، والمقطع
اللي في كل الأسئلة (زي "ايه") وزنه قليل. لما المستخدم يبعت سؤال، بنعمله بصمة
بنفس الطريقة ونقارنها ببصمات كل الأسئلة (تشابه جيب التمام) ونرجّع الأقرب.

مقاطع الحروف هي اللي بتخلي "الدوال" تقرب من "داله" و"مبرمجك" من "برمجك"،
فالمستخدم مش لازم يكتب نفس السؤال بالحرف.

بعد كده بنراجع أقرب 15 سؤال: الكلمات المهمة في السؤال المتخزن لازم تكون موجودة
في كلام المستخدم. ده اللي بيفرّق "عدد سكان الصين" عن "عدد سكان مصر" — نفس
الصيغة بس الكلمة المهمة مختلفة، فالتشابه بينزل.

الحساب مش بيتخزن هنا (النموذج هو اللي بيحسب).
"""

import math
from array import array

import numpy as np

from tokenizer import normalize_question

# الحساب مالوش "أقرب سؤال" (15 + 27 مش قريبة من 15 + 2)، والرسم النموذج هو اللي بيرسمه
SKIP_CATS = ("math", "draw")
WORD_WEIGHT = 1.5              # الكلمة الكاملة أهم من مقطع منها
NGRAM = 3
RERANK = 15                    # عدد الأسئلة اللي بنراجع كلماتها المهمة
WORD_MATCH = 0.5               # كلمتين يعتبروا "نفس الكلمة" لو نص مقاطعهم مشترك


def _grams(word):
    padded = "^" + word + "$"
    return {padded[i:i + NGRAM] for i in range(len(padded) - NGRAM + 1)}


def _same_word(a, b):
    if a == b:
        return True
    ga, gb = _grams(a), _grams(b)
    return 2 * len(ga & gb) / (len(ga) + len(gb)) >= WORD_MATCH


def features(question):
    """بيرجع dict: الخاصية → عدد مرات ظهورها. السؤال لازم يكون متنضف."""
    feats = {}
    for word in question.split():
        key = "w:" + word
        feats[key] = feats.get(key, 0.0) + WORD_WEIGHT
        padded = "^" + word + "$"
        for i in range(len(padded) - NGRAM + 1):
            g = padded[i:i + NGRAM]
            feats[g] = feats.get(g, 0.0) + 1.0
    return feats


class Match:
    __slots__ = ("intent", "question", "score")

    def __init__(self, intent, question, score):
        self.intent, self.question, self.score = intent, question, score

    def __repr__(self):
        return f"Match({self.score:.2f} | {self.question!r} → {self.intent.canonical!r})"


class QuestionMemory:
    def __init__(self, intents=None, include_holdout=True):
        """
        include_holdout: الصيغ اللي النموذج مبيتدربش عليها (عشان نختبره بيها) بتدخل
        الذاكرة في التشغيل العادي. في القياس بنشيلها عشان الاختبار يفضل عادل.
        """
        if intents is None:
            from dataset import build_intents          # من غير torch
            intents = build_intents(verbose=False)
        self.intents = [it for it in intents
                        if it.cat not in SKIP_CATS and getattr(it, "mem", True)]

        self.questions, owner, self.exact = [], [], {}
        for idx, it in enumerate(self.intents):
            extra = it.holdout if include_holdout and it.cat != "unknown" else []
            for q in list(it.questions) + list(extra):
                if q not in self.exact:
                    self.exact[q] = idx
                    self.questions.append(q)
                    owner.append(idx)
        self.owner = np.array(owner, dtype=np.int32)
        n_docs = len(self.questions)

        # 1) عدد الأسئلة اللي فيها كل خاصية → الوزن (IDF)
        df = {}
        for q in self.questions:
            for f in features(q):
                df[f] = df.get(f, 0) + 1
        self.idf = {f: math.log((n_docs + 1) / (c + 1)) + 1.0 for f, c in df.items()}
        self.idf_unknown = math.log(n_docs + 1) + 1.0      # خاصية عمرها ما ظهرت = أندر حاجة

        # 2) فهرس مقلوب: لكل خاصية، أرقام الأسئلة اللي فيها ووزنها هناك (بعد التطبيع).
        #    بنجمّع في مصفوفات مضغوطة (مش قوايم بايثون) عشان الرام تفضل قليلة.
        feat_id = {f: i for i, f in enumerate(self.idf)}
        feat_col, doc_col, val_col = array("i"), array("i"), array("f")
        for d, q in enumerate(self.questions):
            vec = {f: tf * self.idf[f] for f, tf in features(q).items()}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            for f, v in vec.items():
                feat_col.append(feat_id[f])
                doc_col.append(d)
                val_col.append(v / norm)
        feat_col = np.frombuffer(feat_col, dtype=np.int32)
        order = np.argsort(feat_col, kind="stable")
        self._docs = np.frombuffer(doc_col, dtype=np.int32)[order]
        self._vals = np.frombuffer(val_col, dtype=np.float32)[order]
        starts = np.searchsorted(feat_col[order], np.arange(len(feat_id) + 1))
        self._span = {f: (int(starts[i]), int(starts[i + 1])) for f, i in feat_id.items()}

    def __len__(self):
        return len(self.questions)

    def _coverage(self, stored, asked):
        """نسبة (بالوزن) كلمات السؤال المتخزن اللي موجودة في كلام المستخدم"""
        total = hit = 0.0
        for w in stored:
            weight = self.idf.get("w:" + w, self.idf_unknown)
            total += weight
            if any(_same_word(w, v) for v in asked):
                hit += weight
        return hit / total if total else 0.0

    def search(self, message, top=3, cat=None):
        """
        بيرجع أقرب الأسئلة (سؤال واحد لكل رد مختلف) مترتبين من الأقرب.
        score من 0 لـ 1: 1 يعني نفس السؤال بالظبط.
        cat: لو اتحدد، بيرجع الأسئلة اللي من النوع ده بس (أو كل الأنواع ماعدا "!نوع").
        """
        q = normalize_question(message)
        if not q:
            return []
        feats = features(q)
        scores = np.zeros(len(self.questions), dtype=np.float32)
        norm = 0.0
        for f, tf in feats.items():
            w = tf * self.idf.get(f, self.idf_unknown)
            norm += w * w
            span = self._span.get(f)
            if span is not None:
                scores[self._docs[span[0]:span[1]]] += w * self._vals[span[0]:span[1]]
        scores /= math.sqrt(norm) or 1.0

        k = min(len(scores), 60)
        best = np.argpartition(-scores, k - 1)[:k]
        best = best[np.argsort(-scores[best])]

        asked = q.split()
        found, reviewed = {}, 0
        for d in best:
            intent = self.intents[int(self.owner[d])]
            if cat and (intent.cat == cat[1:] if cat.startswith("!") else intent.cat != cat):
                continue
            if reviewed >= RERANK:
                break
            reviewed += 1
            cosine = float(scores[d])
            cover = self._coverage(self.questions[d].split(), asked)
            score = cosine * (0.5 + 0.5 * cover)
            old = found.get(id(intent))
            if old is None or score > old.score:
                found[id(intent)] = Match(intent, self.questions[d], score)
        return sorted(found.values(), key=lambda m: -m.score)[:top]

    def lookup(self, message):
        """
        بيرجع (أقرب سؤال يعرف رده أو None، أعلى تشابه مع سؤال "برّه اللي اتعلمه").
        """
        q = normalize_question(message)
        idx = self.exact.get(q)
        if idx is not None:                               # نفس السؤال بالحرف
            intent = self.intents[idx]
            if intent.cat == "unknown":
                return None, 1.0
            return Match(intent, q, 1.0), 0.0
        known = self.search(message, top=1, cat="!unknown")
        unknown = self.search(message, top=1, cat="unknown")
        return (known[0] if known else None), (unknown[0].score if unknown else 0.0)

if __name__ == "__main__":
    import sys
    mem = QuestionMemory()
    print(f"الذاكرة: {len(mem)} سؤال | {len(mem.idf)} خاصية")
    for msg in sys.argv[1:] or ["مين اللي برمجك", "ايه الحاجات اللي تقدر تعملها", "what's an api"]:
        print(f"\nس: {msg}")
        for m in mem.search(msg):
            print(f"   {m.score:.2f}  {m.question}  →  [{m.intent.cat}] {m.intent.canonical}")
