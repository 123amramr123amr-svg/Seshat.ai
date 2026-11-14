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
import threading
import time

import numpy as np

from checkpoint import load_checkpoint
from tokenizer import CharTokenizer, normalize_question, is_arabic

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "seshat_chet_model.npz")

# لو ثقة النموذج في الرد أقل من الحد ده، بنعتبر إنه مش عارف الإجابة وبنقول كده
# بصراحة بدل ما نطلّع كلام ملخبط.
# الثقة = متوسط احتمال الحروف اللي اختارها، مع تركيز على أول 12 حرف: دول اللي
# النموذج بيقرر فيهم هيرد بإيه، ولو متردد فيهم يبقى غالبًا السؤال برّه اللي اتعلمه.
# الرقم متظبط بالقياس: بيسيب ~97% من الردود الصح، وبيمسك حوالي تلتين الأسئلة
# اللي برّه التدريب (مش كلها — النموذج الصغير بيبقى واثق حتى وهو غلطان).
CONFIDENCE_THRESHOLD = 0.93
CONFIDENCE_HEAD = 12


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

# سطر التفكير اللي النموذج بيكتبه لما يحس إن السؤال برّه اللي اتعلمه
UNKNOWN_AR = "مش معروف | سؤال برّه اللي اتعلمته"
UNKNOWN_EN = "unknown | outside what I have learned"


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
        """بيرجع (التوكنز الجديدة، احتمال كل توكن)"""
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
        }

    # ------------------------------------------------------------
    # الرد = الذاكرة (أقرب سؤال) + النموذج (الفهم والحساب)
    # ------------------------------------------------------------
    def reply(self, message, temperature=0.0, max_new_tokens=600, use_fallback=True,
              use_memory=True):
        """
        بيرجع dict فيه: response, thinking, confidence, seconds, fallback, source, matched,
        suggestion (سؤال بيقترحه لما يكون مش متأكد).

        source بيقول الرد جه منين:
          "memory"        سؤال قريب كفاية من سؤال متخزن
          "memory+model"  الذاكرة والنموذج اتفقوا على نفس السؤال
          "model"         النموذج لوحده (الحساب، أو لما الذاكرة مش شغالة)
          "none"          مفيش رد — السؤال برّه اللي اتعلمه
        use_memory=False بيرجع رد النموذج لوحده (ده اللي evaluate.py بيقيسه).
        """
        t0 = time.time()
        question = normalize_question(message)[:300]
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
        known, unknown_score = memory.lookup(message)

        def answer(match, source, agreed=False):
            pct = round(match.score * 100)
            if arabic:
                think = f"أقرب سؤال أعرفه: «{match.intent.canonical}» — تشابه {pct}%"
                think += " — والنموذج فهمه بنفس المعنى" if agreed else ""
            else:
                think = f"closest question I know: “{match.intent.canonical}” — {pct}% similar"
                think += " — the model read it the same way" if agreed else ""
            return {"response": match.intent.answer, "thinking": think,
                    "confidence": round(match.score, 3), "fallback": False,
                    "source": source, "matched": match.intent.canonical, "suggestion": ""}

        def dont_know():
            # مش متأكد → منألّفش رد. لو فيه سؤال قريب شوية بنقترحه على المستخدم.
            text, think, suggestion = (FALLBACK_AR if arabic else FALLBACK_EN), "", ""
            if (known is not None and known.score >= MEMORY_SUGGEST
                    and known.score >= unknown_score - UNKNOWN_MARGIN):
                suggestion = known.intent.canonical
                pct = round(known.score * 100)
                if arabic:
                    text = f"مش متأكد إني فهمت سؤالك. تقصد: «{suggestion}»؟"
                    think = f"أقرب سؤال أعرفه: «{suggestion}» — تشابه {pct}% بس، مش كفاية عشان أرد"
                else:
                    text = f"I'm not sure I understood. Did you mean: “{suggestion}”?"
                    think = f"closest question I know: “{suggestion}” — only {pct}% similar, not enough to answer"
            return {"response": text, "thinking": think,
                    "confidence": round(known.score, 3) if known else 0.0, "fallback": True,
                    "source": "none", "matched": "", "suggestion": suggestion}

        # 1) نفس السؤال موجود بالحرف → مش محتاجين النموذج
        if known is not None and known.score >= 0.999:
            return answer(known, "memory")
        if known is None and unknown_score >= 0.999:
            return dont_know()

        # 2) النموذج يفكر: السؤال ده نوعه إيه ومعناه إيه
        model = self._model_reply(message, question, temperature, max_new_tokens, True)
        thinking = model["thinking"]

        # 3) سؤال قريب جدًا من سؤال متخزن
        if known is not None and known.score >= MEMORY_STRONG and known.score > unknown_score:
            agreed = thinking == known.intent.think
            return answer(known, "memory+model" if agreed else "memory", agreed=agreed)

        # 4) حساب: النموذج هو اللي بيحسب، بشرط إنه يكون قرا نفس الأرقام اللي في السؤال
        if thinking.startswith(MATH_LABEL + " |"):
            if model["fallback"] or _numbers(question) != _numbers(thinking):
                return dont_know()
            model.update(source="model", matched="", suggestion="")
            return model

        # 5) تشابه متوسط، بس النموذج فهم السؤال بنفس المعنى → الاتنين متفقين
        if (known is not None and thinking == known.intent.think
                and known.score >= MEMORY_AGREE
                and known.score >= unknown_score - UNKNOWN_MARGIN):
            return answer(known, "memory+model", agreed=True)

        return dont_know()

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
        if use_fallback and not fallback and (not text or confidence < CONFIDENCE_THRESHOLD):
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

    def _forward(self, ids, caches):
        """بيدخّل توكنز جديدة ويرجّع logits آخر توكن. caches فيها k و v بتوع اللي فات."""
        H, hd, C = self.n_head, self.head_dim, self.n_embd
        past = caches[0]["k"].shape[1] if "k" in caches[0] else 0
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
            if "k" in cache:
                k = np.concatenate([cache["k"], k], axis=1)
                v = np.concatenate([cache["v"], v], axis=1)
            cache["k"], cache["v"] = k, v

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
        caches = [{} for _ in self.layers]
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


def Seshat(path=DEFAULT_MODEL, engine=None):
    """بيحمّل النموذج بالمحرك المناسب (NumPy افتراضيًا)"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ملف النموذج مش موجود: {path}\nشغّل الأول:  python train_model.py")
    engine = (engine or os.environ.get("SESHAT_ENGINE", "numpy")).lower()
    if engine == "torch":
        return TorchSeshat(path)
    return NumpySeshat(path)
