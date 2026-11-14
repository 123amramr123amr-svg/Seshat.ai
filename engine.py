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


def confidence_from(probs):
    if not probs:
        return 0.0
    head = probs[:CONFIDENCE_HEAD]
    return min(sum(probs) / len(probs), sum(head) / len(head))

FALLBACK_AR = "لسه ماتعلمتش الرد على السؤال ده. جرب تسأله بطريقة تانية، أو اسألني عن حاجة تانية."
FALLBACK_EN = "I haven't learned how to answer that yet. Try asking in another way, or ask me something else."


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
        }

    def reply(self, message, temperature=0.0, max_new_tokens=600, use_fallback=True):
        """بيرجع dict فيه: response, confidence, seconds, fallback"""
        t0 = time.time()
        question = normalize_question(message)[:300]
        if not question:
            return {"response": "اكتب حاجة الأول 🙂", "confidence": 1.0,
                    "seconds": 0.0, "fallback": False}

        prompt = self.tok.encode_prompt(question)
        max_prompt = self.config["block_size"] // 2
        if len(prompt) > max_prompt:               # نسيب مكان للرد
            prompt = prompt[:max_prompt - 1] + [self.tok.bot_id]

        with self._lock:                            # طلب واحد في المرة
            ids, probs = self._generate(prompt, max_new_tokens, temperature)
        confidence = confidence_from(probs)

        text = fix_code_indent(self.tok.decode(ids).strip())
        fallback = False
        if use_fallback and (not text or confidence < CONFIDENCE_THRESHOLD):
            text = FALLBACK_AR if is_arabic(message) else FALLBACK_EN
            fallback = True
        return {"response": text, "confidence": round(float(confidence), 3),
                "seconds": round(time.time() - t0, 2), "fallback": fallback}


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
