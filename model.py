"""
seshat.chet — نموذج GPT عربي/إنجليزي مبني بالكامل من الصفر
كل مكوّن هنا مكتوب يدويًا: Attention، LayerNorm، Transformer Block، والتوليد
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# الإعدادات
# ============================================================
# جولة خ12 — السياق الممتد: أقصى عدد حروف (توكنز) إدخال وإخراج.
# المعمارية بتشتغل بنافذة منزلقة + مراسي انتباه (نفس فكرة StreamingLLM):
# الذاكرة ما بتكبرش مع طول النص، فالنموذج بيقرا وبيكتب ملايين الحروف
# من غير ما الرام تتملي. سياق التدريب الفعلي (block_size) بيفضل 4096.
MAX_CONTEXT = 2_000_000
# مراسي الانتباه (attention sinks): أول حروف السياق بتفضل محفوظة دايمًا —
# بتنزّل توزيع الانتباه وبتخلي التدفق اللانهائي مستقر.
N_SINK = 8
# نافذة الانزلاق: عدد الحروف الأخيرة اللي النموذج شايفها بالتفصيل وقت التدفق
# على نصوص أطول من block_size (StreamingLLM). التدريب بيفضل انتباه كامل،
# والمحرك بيستخدم النافذة بس للمستندات الضخمة (شوف engine.reply_long).
STREAM_WINDOW = 768

# أحجام جاهزة — كل ما الحجم يكبر النموذج يحفظ ويفهم أكتر، بس محتاج وقت/جهاز أقوى
PRESETS = {
    "tiny":   dict(n_layer=4,  n_head=4,  n_embd=128),   # ~0.9M  — CPU ضعيف
    "small":  dict(n_layer=6,  n_head=6,  n_embd=192),   # ~2.8M  — CPU كويس
    "medium": dict(n_layer=8,  n_head=8,  n_embd=256),   # ~6.5M  — GPU أو صبر
    "large":  dict(n_layer=12, n_head=12, n_embd=384),   # ~21M   — GPU
    # خ23/خ24 — أمر المستخدم: «نموذج واحد بس، أي تزويد معاملات يكون على نفس النموذج»
    # 16 رأس (head_dim 32 زي النموذج الأصلي) عشان grow_model.py ينقل المعرفة بتطابق تام.
    # مسار الأحجام على نفس الملف: خ24 وصل 52.6M ثم خ26 وصل 102.9M (32 طبقة).
    "50m":    dict(n_layer=16, n_head=16, n_embd=512),   # ~52.6M — حالة تاريخية (خ24–خ25)
    # خ26 — أمر المستخدم: «دربه لغايه 100M معامله هوهو نفس النموذج» —
    # نفس الملف اتوسع 16→32 طبقة (grow_model.py، طبقات هوية) = 102,917,632 معامل.
    "100m":   dict(n_layer=32, n_head=16, n_embd=512),   # ~102.9M — تعدى الـ100M
}


class Config:
    def __init__(self, **kw):
        # 4096 حرف: السياق الاتسع. المصفوفة وقت التدريب بتاخد حوالي 24 بايت ×
        # عدد الرؤوس × الطبقات × الطول الفعلي للنص² — وبما إن أمثلة التدريب
        # (سؤال + غلاف + رد) مش بتعدّي ~1500 حرف، التدريب على سياق 4096
        # بيكلف نفس رام 2048 بالظبط، والزيادة دي كلها مساحة مفتوحة للردود
        # والرسايل الأطول. النموذج القديم كان 512 واتوسّع بضرب جدول المواقع
        # (توسيع دوره — شوف train_model.py).
        self.block_size = 4096     # أقصى عدد حروف (سؤال + رد) النموذج يشوفها
        # خ12 — السياق الممتد (نفس فكرة النافذة المنزلقة في التدفق):
        self.max_context = MAX_CONTEXT      # أقصى إدخال وإخراج (٢ مليون حرف)
        self.n_sink = N_SINK                # مراسي الانتباه (أول حروف السياق)
        self.stream_window = STREAM_WINDOW  # النافذة المنزلقة للتدفق
        self.n_layer = 4
        self.n_head = 4
        self.n_embd = 128
        self.dropout = 0.1
        self.vocab_size = None
        for k, v in kw.items():
            if not hasattr(self, k):
                raise ValueError(f"إعداد مش معروف: {k}")
            setattr(self, k, v)

    def to_dict(self):
        # بنحفظ كل الإعدادات (النسخة القديمة كانت بتحفظ vocab_size بس)
        return dict(vars(self))


# ============================================================
# المكوّنات
# ============================================================
class LayerNorm(nn.Module):
    """تطبيع يدوي — من غير استخدام nn.LayerNorm"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class MultiHeadAttention(nn.Module):
    """انتباه متعدد الرؤوس مع قناع سببي"""
    # خ12: كاش لأقنعة النافذة — بنفس (T, past, window, n_sink) القناع واحد
    # دايمًا، فبنبنيه مرة ونستخدمه في كل الطبقات وكل الخطوات (توفير وقت كبير)
    _win_cache = {}
    # خ26: قناع سببي واحد مشترك لكل الطبقات بنفس block_size — قبل كده كل طبقة
    # كانت بتحجز لوحدها [1,1,block,block] float32 (67MB عند سياق 4096) يعني
    # 32 طبقة = 2.15GB عالفاضي في البافرات، وده كان القاتل الحقيقي لجلسات
    # تدريب الـ100M (الرسم البياني للتدرجات فوق البافرات دي كان بيعدي حد
    # الرام والعملية بتتقتل). القناع للقراءة فقط فنسخة مشتركة كافية وصح.
    _causal_mask_cache = {}

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        if cfg.block_size not in MultiHeadAttention._causal_mask_cache:
            m = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
            MultiHeadAttention._causal_mask_cache[cfg.block_size] = \
                m.view(1, 1, cfg.block_size, cfg.block_size)
        self.register_buffer("mask", MultiHeadAttention._causal_mask_cache[cfg.block_size],
                             persistent=False)

    @classmethod
    def _window_mask(cls, T, past, window, n_sink, device):
        key = (T, past, window, n_sink, str(device))
        m = cls._win_cache.get(key)
        if m is None:
            key_pos = torch.arange(past + T, device=device)
            q_pos = torch.arange(past, past + T, device=device)
            allow = (key_pos[None, :] <= q_pos[:, None]) | (key_pos[None, :] < n_sink)
            in_win = key_pos[None, :] > q_pos[:, None] - window
            m = (allow & in_win)[None, None]
            if len(cls._win_cache) > 64 and T * (past + T) > 4_000_000:
                return m                      # القناع الضخم مش بنخزنه عشان الرام
            if len(cls._win_cache) >= 64:
                cls._win_cache.clear()
            cls._win_cache[key] = m
        return m

    def forward(self, x, cache=None, window=None, n_sink=0):
        """
        window (خ12): نافذة منزلقة في التدريب — كل حرف بيشوف آخر `window` حرف
        + المراسي الأولى (n_sink). بتعادل ذاكرة التدفق في التشغيل، فالتدريب
        والتشغيل في نفس الوضع بالظبط.
        """
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # ذاكرة التوليد (KV cache): وقت التوليد بندخل الحرف الجديد بس، والـ k و v
        # بتوع الحروف اللي فاتت متخزنين، فمبنعيدش حسابهم كل مرة.
        past = 0
        if cache is not None:
            if "k" in cache:
                past = cache["k"].size(2)
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if window is None:
            att = att.masked_fill(self.mask[:, :, past:past + T, :past + T] == 0, float("-inf"))
        else:
            att = att.masked_fill(self._window_mask(T, past, window, n_sink, x.device) == 0,
                                  float("-inf"))
        att = self.attn_drop(F.softmax(att, dim=-1))

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.fc2 = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    """طبقة Transformer كاملة: Attention + FeedForward مع Residual"""
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = LayerNorm(cfg.n_embd)
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = LayerNorm(cfg.n_embd)
        self.ff = FeedForward(cfg)

    def forward(self, x, cache=None, window=None, n_sink=0):
        x = x + self.attn(self.ln1(x), cache, window=window, n_sink=n_sink)
        x = x + self.ff(self.ln2(x))
        return x


# ============================================================
# النموذج الكامل
# ============================================================
class MiniGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.vocab_size is not None, "لازم تحدد vocab_size"
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight tying — نفس الوزن بين التضمين والمخرج
        self.lm_head.weight = self.tok_emb.weight

        # تهيئة الأوزان يدويًا (بتفرق جدًا في ثبات التدريب)
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, caches=None, window=None):
        B, T = idx.shape
        past = caches[0]["k"].size(2) if caches and "k" in caches[0] else 0
        assert past + T <= self.cfg.block_size, \
            f"طول النص {past + T} أكبر من block_size {self.cfg.block_size}"
        pos = torch.arange(past, past + T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for i, block in enumerate(self.blocks):
            x = block(x, caches[i] if caches else None,
                      window=window, n_sink=self.cfg.n_sink)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    @torch.no_grad()
    def generate_stream(self, prompt_ids, max_new_tokens=2400, temperature=0.0, top_k=10,
                        end_id=None, banned_ids=()):
        """
        بيولّد الرد حرف ورا حرف لحد توكن النهاية، وبيطلّع (التوكن، احتماله) أول بأول.
        temperature = 0 يعني اختيار أعلى احتمال دايمًا (أدق حاجة للأكواد).
        """
        self.eval()
        device = next(self.parameters()).device
        prompt_ids = list(prompt_ids)[-(self.cfg.block_size - 1):]
        caches = [{} for _ in self.blocks]
        step_in = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        room = self.cfg.block_size - len(prompt_ids)
        for _ in range(max(0, min(max_new_tokens, room))):
            logits, _ = self(step_in, caches=caches)
            logits = logits[0, -1, :]
            for b in banned_ids:
                logits[b] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            if temperature <= 0:
                nxt = int(torch.argmax(probs))
            else:
                scaled = logits / temperature
                if top_k:
                    kth = torch.topk(scaled, min(top_k, scaled.numel())).values[-1]
                    scaled = scaled.masked_fill(scaled < kth, float("-inf"))
                nxt = int(torch.multinomial(F.softmax(scaled, dim=-1), 1))
            if end_id is not None and nxt == end_id:
                return
            yield nxt, float(probs[nxt])
            step_in = torch.tensor([[nxt]], dtype=torch.long, device=device)

    def generate(self, prompt_ids, **kw):
        """بيرجع (التوكنز الجديدة، احتمال كل توكن اختاره)"""
        out, probs = [], []
        for tok, p in self.generate_stream(prompt_ids, **kw):
            out.append(tok)
            probs.append(p)
        return out, probs


# ============================================================
# خ12 — التدفق الطويل: نافذة منزلقة + مراسي انتباه
# ============================================================
# نفس مبدأ StreamingLLM: الذاكرة فيها عدد مقاعد ثابت (مراسي + نافذة)، والحروف
# الجديدة بتتكتب في المقاعد بدورها، ومواقع الانتباه بتتقاس على ترتيب الحرف في
# الذاكرة مش على مكانه الأصلي في النص. النتيجة: النموذج بيقرا وبيكتب حتى
# max_context حرف بذاكرة ثابتة الحجم، ومواقع الحروف دايمًا في حدود جدول
# المواقع المدرّب — فمفيش موقع جديد عمره ما اتدرب عليه.
# حجم الذاكرة = المراسي + النافذة (776). التدريب بيفضل انتباه كامل على
# مستندات ≤ block_size (ودي بتتمشي بالمسار الكلاسيكي في المحرك)، والذاكرة دي
# بتشتغل بس في التدفق على النصوص الأطول من block_size.
class LongStreamCache:
    """ذاكرة التدفق: مقاعد ثابتة = المراسي + النافذة، والحروف بتلف عليها بالترتيب"""

    def __init__(self, model):
        cfg = model.cfg
        self.n_sink = max(1, min(cfg.n_sink, cfg.block_size - 1))
        win = getattr(cfg, "stream_window", 0) or (cfg.block_size - self.n_sink)
        self.window = max(1, min(win, cfg.block_size - self.n_sink))
        self.capacity = self.n_sink + self.window
        self.count = 0                       # إجمالي الحروف اللي عدت من هنا
        self.order = []                      # المقاعد بترتيب الإدخال (للمواقع)
        self.k = [None] * len(model.blocks)  # (H, capacity, hd) لكل طبقة
        self.v = [None] * len(model.blocks)
        for i, block in enumerate(model.blocks):
            H, hd = block.attn.n_head, block.attn.head_dim
            dev = block.attn.c_attn.weight.device
            self.k[i] = torch.zeros(H, self.capacity, hd, device=dev)
            self.v[i] = torch.zeros(H, self.capacity, hd, device=dev)

    def _slot(self, t):
        """مقعد الحرف رقم t (المراسي الأول، وبعدين تلف على النافذة)"""
        if t < self.n_sink:
            return t
        return self.n_sink + (t - self.n_sink) % self.window

    def plan(self, T):
        """
        خطة كتابة T حرف جديد: بترجع المقاعد وبتحدّث count والترتيب مرة واحدة
        (بتتنادى مرة في كل خطوة — الطبقات كلها بتكتب في نفس المقاعد).
        """
        t0 = self.count
        slots = [self._slot(t) for t in range(t0, t0 + T)]
        fill_end = min(t0 + T, self.capacity)
        split = max(0, min(T, fill_end - t0))   # اللي بيملي مقاعد فاضية
        if split > 0:                            # مقاعد فاضية بتتملي بالترتيب
            self.order += slots[:split]
        if split < T:                            # الذاكرة اتملت → لف على النافذة:
            # أقدم حروف النافذة هي اللي مقاعدهم بتتكتب دلوقتي — بيتشالوا من
            # أول ترتيب النافذة، والحروف الجديدة بتتحط في آخره (FIFO)
            wrap = slots[split:]
            del self.order[self.n_sink:self.n_sink + len(wrap)]
            self.order += wrap
        self.count += T
        return {"t0": t0, "slots": slots, "fill_end": fill_end}

    def write(self, layer, k, v, slots, fill_end, t0):
        """بيكتب k و v في مقاعدهم (tensor write واحدة لكل مرحلة)"""
        dev = self.k[layer].device
        split = max(0, min(len(slots), fill_end - t0))
        if split > 0:
            idx = torch.tensor(slots[:split], device=dev)
            self.k[layer][:, idx] = k[:, :split]
            self.v[layer][:, idx] = v[:, :split]
        if split < len(slots):
            idx = torch.tensor(slots[split:], device=dev)
            self.k[layer][:, idx] = k[:, split:]
            self.v[layer][:, idx] = v[:, split:]

    def ordered_kv(self, layer):
        """(k, v) مرتّبين بترتيب الإدخال — المواقع 0..occupied-1"""
        idx = torch.tensor(self.order, device=self.k[layer].device)
        return self.k[layer][:, idx], self.v[layer][:, idx]

    @property
    def occupied(self):
        return len(self.order)


def _long_step(model, ids, cache):
    """
    خطوة في التدفق الطويل: ids حروف جديدة (chunk ≤ النافذة)، بيتكتبوا في
    الذاكرة وبيترصّوا بالترتيب، والانتباه بيحصل جوه الchunk وبينهم وبين
    الذاكرة القديمة. بيرجع logits آخر حرف (توزيع الحرف الجاي).
    """
    dev = ids.device
    T = len(ids)
    # خطة الكتابة مرة واحدة (الطبقات كلها نفس المقاعد والمواقع)
    p = cache.plan(T)
    occ_after = min(p["t0"] + T, cache.capacity)
    q_pos0 = occ_after - T
    x = model.tok_emb(ids) + model.pos_emb(torch.arange(q_pos0, occ_after, device=dev))[None]
    for i, block in enumerate(model.blocks):
        h = block.ln1(x[0])
        C = h.shape[-1]
        q, k, v = (block.attn.c_attn(h)).split(C, dim=1)
        H, hd = block.attn.n_head, block.attn.head_dim
        q = q.view(T, H, hd).transpose(0, 1)     # (H, T, hd)
        k = k.view(T, H, hd).transpose(0, 1)
        v = v.view(T, H, hd).transpose(0, 1)
        cache.write(i, k, v, p["slots"], p["fill_end"], p["t0"])
        ko, vo = cache.ordered_kv(i)             # (H, occ, hd)
        occ = ko.shape[1]
        att = (q @ ko.transpose(1, 2)) / math.sqrt(hd)          # (H, T, occ)
        # قناع سببي: query j (موقعه q_pos0+j) بيشوف المفاتيح لحد موقعه
        q_pos = torch.arange(q_pos0, occ, device=dev)
        k_pos = torch.arange(occ, device=dev)
        att = att.masked_fill(k_pos[None, None, :] > q_pos[None, :, None], float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ vo).transpose(0, 1).reshape(T, C)
        x = x + block.attn.resid_drop(block.attn.c_proj(y))[None]
        h = block.ln2(x[0])
        x = x + block.ff(h)[None]
    x = model.ln_f(x[0][-1])
    return model.lm_head(x)


def _long_methods(cls):
    """بيضيف دوال التدفق الطويل على MiniGPT (بتتستدعي من engine.py كمان)"""

    def prefill_stream(self, prompt_ids, cache, chunk=512):
        """تدفق نص طويل في الذاكرة — chunk chunk، والمقاعد ما بتعداش capacity أبدًا"""
        ids = list(prompt_ids)
        chunk = max(1, min(chunk, cache.window))
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            for s in range(0, len(ids), chunk):
                _long_step(self, torch.tensor(ids[s:s + chunk], dtype=torch.long,
                                              device=device), cache)
        return cache

    def generate_stream_long(self, prompt_ids, max_new_tokens=2400, temperature=0.0,
                             top_k=10, end_id=None, banned_ids=(), chunk=512):
        """
        التوليد الطويل: البرومبت كله بيتدفق في الذاكرة (مهما كان طوله — لحد
        max_context)، وبعدين التوليد بيكمل من غير حد block_size: لحد end_id
        أو max_new_tokens (بحد أقصى max_context = ٢ مليون توكن).
        بيرجّع (توكن، احتماله) واحد ورا التاني زي generate_stream بالظبط.
        """
        self.eval()
        device = next(self.parameters()).device
        prompt_ids = list(prompt_ids)
        cap = getattr(self.cfg, "max_context", MAX_CONTEXT)
        max_new_tokens = max(0, min(max_new_tokens, cap - max(1, len(prompt_ids))))
        cache = LongStreamCache(self)
        logits = None
        c = max(1, min(chunk, cache.window))
        with torch.no_grad():
            for s in range(0, len(prompt_ids), c):
                logits = _long_step(self, torch.tensor(prompt_ids[s:s + c],
                                                       dtype=torch.long, device=device), cache)
            if logits is None:                      # برومبت فاضي — مفيش حاجة تتولد
                return
            for _ in range(max_new_tokens):
                for b in banned_ids:
                    logits[b] = float("-inf")
                probs = F.softmax(logits.float(), dim=-1)
                if temperature <= 0:
                    nxt = int(torch.argmax(probs))
                else:
                    scaled = logits / temperature
                    if top_k:
                        kth = torch.topk(scaled, min(top_k, scaled.numel())).values[-1]
                        scaled = scaled.masked_fill(scaled < kth, float("-inf"))
                    nxt = int(torch.multinomial(F.softmax(scaled.float(), dim=-1), 1))
                if end_id is not None and nxt == end_id:
                    return
                yield nxt, float(probs[nxt])
                # الحرف المولّد بيدخل الذاكرة ويجيب توزيع اللي بعده
                logits = _long_step(self, torch.tensor([nxt], dtype=torch.long,
                                                       device=device), cache)

    def generate_long(self, prompt_ids, **kw):
        out, probs = [], []
        for tok, p in self.generate_stream_long(prompt_ids, **kw):
            out.append(tok)
            probs.append(p)
        return out, probs

    cls.prefill_stream = prefill_stream
    cls.generate_stream_long = generate_stream_long
    cls.generate_long = generate_long
    return cls


MiniGPT = _long_methods(MiniGPT)
