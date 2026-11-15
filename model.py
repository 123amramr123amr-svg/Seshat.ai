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
# أحجام جاهزة — كل ما الحجم يكبر النموذج يحفظ ويفهم أكتر، بس محتاج وقت/جهاز أقوى
PRESETS = {
    "tiny":   dict(n_layer=4,  n_head=4,  n_embd=128),   # ~0.9M  — CPU ضعيف
    "small":  dict(n_layer=6,  n_head=6,  n_embd=192),   # ~2.8M  — CPU كويس
    "medium": dict(n_layer=8,  n_head=8,  n_embd=256),   # ~6.5M  — GPU أو صبر
    "large":  dict(n_layer=12, n_head=12, n_embd=384),   # ~21M   — GPU
}


class Config:
    def __init__(self, **kw):
        # 2048 حرف = أكبر سياق ينفع يتدرب على جهاز بـ ٣ جيجا رام (مصفوفة الانتباه
        # وقت التدريب بتاخد حوالي 24 بايت × عدد الرؤوس × الطبقات × الطول² —
        # عند 2048 و16 رأس×طبقة ≈ 1.6 جيجا). النموذج القديم كان 512 واتوسّع
        # بضرب جدول المواقع (توسيع دوره — شوف train_model.py).
        self.block_size = 2048     # أقصى عدد حروف (سؤال + رد) النموذج يشوفها
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
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size),
                             persistent=False)

    def forward(self, x, cache=None):
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
        att = att.masked_fill(self.mask[:, :, past:past + T, :past + T] == 0, float("-inf"))
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

    def forward(self, x, cache=None):
        x = x + self.attn(self.ln1(x), cache)
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

    def forward(self, idx, targets=None, caches=None):
        B, T = idx.shape
        past = caches[0]["k"].size(2) if caches and "k" in caches[0] else 0
        assert past + T <= self.cfg.block_size, \
            f"طول النص {past + T} أكبر من block_size {self.cfg.block_size}"
        pos = torch.arange(past, past + T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for i, block in enumerate(self.blocks):
            x = block(x, caches[i] if caches else None)
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
    def generate_stream(self, prompt_ids, max_new_tokens=1200, temperature=0.0, top_k=10,
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
