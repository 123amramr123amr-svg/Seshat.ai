"""
seshat.chet — توسيع نموذج واحد (grow_model.py)

أمر المستخدم (خ24): «نموذج واحد بس — أي تزويد معاملات يكون على نفس النموذج».
السكربت ده بيوسّع النموذج الرئيسي نفسه من غير ما يفقد أي معرفة:

الفكرة (Net2Net-style بالنقل الدقيق):
  - كل الأبعاد بتتربّع بعامل صحيح f (مثلًا 128 → 512 = ×4): كل مجموعة قنوات
    بتبقى نسخة طبق الأصل من القنوات القديمة، فالشبكة الموسّعة تحسب *نفس
    الدالة بالظبط* لحظة التوسيع.
  - الطبقات الجديدة فوق القديمة بتبدأ هوية (مخرجات الصفر) — ما بتضيفش حاجة
    لحظة الدمج وتتعلم بعدين.
  - ln_f (gamma وbeta معًا) بيتقسم على f عشان الـlogits ترجع بنفس المقياس
    الأصلي بالضبط (الرأس مربوط بالتضمين المُربَّع وجزء الـbias لازم يتوازن كمان)
    — فالتوليد في المحرك متطابق حرفيًا.
  - head_dim ثابت (32) وعدد الرؤوس بيكبر تلقائي: كل رأس قديم بيتكرر f مرة.

التحقق مدمج: بيقارن logits النموذج القديم والجديد على أسئلة فعلية ويطبع أقصى
فرق — لازم ~صفر. التدريب بعدها على نفس الملف بيخصص النسخ المكررة والطبقات الجديدة.

أمثلة:
  python3 grow_model.py --src backups/seshat_k23_static_69408.npz \
      --out seshat_chet_model.npz --target-n-embd 512 --target-n-layer 16
  python3 grow_model.py --src seshat_chet_model.npz --out seshat_chet_model.npz \
      --target-n-embd 1024 --target-n-layer 24        # تكبير لاحق على نفس النموذج
"""

import argparse
import json
import os

import numpy as np

from checkpoint import load_checkpoint, save_checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description="توسيع نموذج seshat الواحد مع نقل كامل للمعرفة")
    p.add_argument("--src", required=True, help="ملف النموذج المصدر")
    p.add_argument("--out", required=True, help="ملف النموذج بعد التوسيع (ممكن يكون نفس المصدر)")
    p.add_argument("--target-n-embd", type=int, required=True, help="البعد الجديد (لازم مضاعفة صحيحة للقديم)")
    p.add_argument("--target-n-layer", type=int, required=True, help="عدد الطبقات الجديد (≥ القديم)")
    p.add_argument("--target-block-size", type=int, default=0, help="السياق الجديد (0 = زي القديم)")
    p.add_argument("--new-weight-std", type=float, default=0.02, help="انحراف تهيئة الطبقات الجديدة")
    p.add_argument("--no-verify", action="store_true", help="تخطي فحص تطابق الـlogits")
    return p.parse_args()


def tile_cols(w, f):
    """توسيع أفقي: [.., d] → [.., d*f] بمجموعات متطابقة"""
    return np.tile(w, (1, f))


def tile_vec(w, f, scale=1.0):
    """توسيع متجه LN: [d] → [d*f] (مع قسمة اختيارية لموازنة الرأس المربوط)"""
    return np.tile(w, f) * scale


def tile_matrix(w, f, scale=1.0):
    """توسيع مصفوفة خطية في الاتجاهين: [out, in] → [out*f, in*f] مع قسمة f
    عشان مجموع المساهمات المتطابقة يرجع بنفس القيمة"""
    return np.tile(w, (f, f)) * scale


def tile_qkv(w, f, scale=1.0):
    """c_attn [3*d, d]: التوسيع لازم يبقى *جوه* كل قسم q/k/v عشان الترتيب يفضل صحيح،
    وفي الاتجاهين (out وin) مع قسمة f عشان المجموع يرجع بنفس القيمة"""
    sections = np.split(w, 3, axis=0)
    return np.vstack([np.tile(s, (f, f)) * scale for s in sections])


def widen_weights(weights, cfg, tgt_layer, tgt_embd, tgt_block, new_std):
    """يبني أوزان النموذج الموسّع من أوزان المصدر — نقل دقيق بدون فقدان"""
    f = tgt_embd // cfg["n_embd"]
    assert f >= 1 and tgt_embd % cfg["n_embd"] == 0, "البعد الجديد لازم مضاعفة صحيحة"
    src_layer = cfg["n_layer"]
    assert tgt_layer >= src_layer, "عدد الطبقات الجديد لازم ≥ القديم"
    head_dim = cfg["n_embd"] // cfg["n_head"]
    assert head_dim * cfg["n_head"] == cfg["n_embd"]
    new = {}

    new["tok_emb.weight"] = tile_cols(weights["tok_emb.weight"], f)
    new["pos_emb.weight"] = tile_cols(weights["pos_emb.weight"], f)

    for i in range(src_layer):
        p = f"blocks.{i}."
        new[p + "ln1.gamma"] = tile_vec(weights[p + "ln1.gamma"], f)
        new[p + "ln1.beta"] = tile_vec(weights[p + "ln1.beta"], f)
        new[p + "attn.c_attn.weight"] = tile_qkv(weights[p + "attn.c_attn.weight"], f, 1.0 / f)
        new[p + "attn.c_proj.weight"] = tile_matrix(weights[p + "attn.c_proj.weight"], f, 1.0 / f)
        new[p + "ln2.gamma"] = tile_vec(weights[p + "ln2.gamma"], f)
        new[p + "ln2.beta"] = tile_vec(weights[p + "ln2.beta"], f)
        new[p + "ff.fc1.weight"] = tile_matrix(weights[p + "ff.fc1.weight"], f, 1.0 / f)
        new[p + "ff.fc2.weight"] = tile_matrix(weights[p + "ff.fc2.weight"], f, 1.0 / f)

    rng = np.random.default_rng(cfg.get("seed", 1234) + 24)
    for i in range(src_layer, tgt_layer):
        p = f"blocks.{i}."
        # طبقة هوية: mlp/bn محايدة ومخرجات الانتباه صفر → البلوك مبيضيفش حاجة
        new[p + "ln1.gamma"] = np.ones(tgt_embd, dtype=np.float32)
        new[p + "ln1.beta"] = np.zeros(tgt_embd, dtype=np.float32)
        new[p + "ln2.gamma"] = np.ones(tgt_embd, dtype=np.float32)
        new[p + "ln2.beta"] = np.zeros(tgt_embd, dtype=np.float32)
        new[p + "attn.c_attn.weight"] = (rng.normal(0, new_std, (3 * tgt_embd, tgt_embd))).astype(np.float32)
        new[p + "attn.c_proj.weight"] = np.zeros((tgt_embd, tgt_embd), dtype=np.float32)
        new[p + "ff.fc1.weight"] = (rng.normal(0, new_std, (4 * tgt_embd, tgt_embd))).astype(np.float32)
        new[p + "ff.fc2.weight"] = np.zeros((tgt_embd, 4 * tgt_embd), dtype=np.float32)

    new["ln_f.gamma"] = tile_vec(weights["ln_f.gamma"], f, 1.0 / f)
    new["ln_f.beta"] = tile_vec(weights["ln_f.beta"], f, 1.0 / f)
    return new, f


def verify_exact(src_path, new_weights, new_cfg, tokens, prompts):
    """يقارن logits القديم والجديد — لازم الفرق ~صفر والاختيار متطابق"""
    import torch
    from model import Config, MiniGPT
    from tokenizer import CharTokenizer

    tok = CharTokenizer(tokens)

    def build(cfg_dict, weights):
        c = Config(**cfg_dict)
        m = MiniGPT(c)
        sd = {k: torch.from_numpy(np.asarray(v, dtype=np.float32)) for k, v in weights.items()}
        res = m.load_state_dict(sd, strict=False)
        assert not res.unexpected_keys, f"مفاتيح زايدة: {res.unexpected_keys}"
        m.eval()
        return m

    src = load_checkpoint(src_path)
    m_src = build(src["config"], src["weights"])
    m_new = build(new_cfg, new_weights)
    max_diff = 0.0
    same_argmax = True
    with torch.no_grad():
        for q in prompts:
            ids = tok.encode(q)
            if not ids:
                continue
            ids_t = torch.tensor([ids], dtype=torch.long)
            lo, _ = m_src(ids_t)
            ln, _ = m_new(ids_t)
            a, b = lo[0, -1], ln[0, -1]
            max_diff = max(max_diff, float((a - b).abs().max()))
            same_argmax &= int(a.argmax()) == int(b.argmax())
    return max_diff, same_argmax


def main():
    args = parse_args()
    src = load_checkpoint(args.src)
    cfg = dict(src["config"])
    weights = src["weights"]
    vocab = weights["tok_emb.weight"].shape[0]
    tgt_block = args.target_block_size or cfg["block_size"]
    print(f"المصدر: {args.src} | {cfg['n_layer']} طبقة × {cfg['n_embd']} بعد × {cfg['n_head']} رؤوس "
          f"| {src['step']:,} خطوة | مفردات {vocab}")

    new_cfg = dict(cfg)
    new_cfg["n_layer"] = args.target_n_layer
    new_cfg["n_embd"] = args.target_n_embd
    new_cfg["n_head"] = args.target_n_embd // (cfg["n_embd"] // cfg["n_head"])
    new_cfg["block_size"] = tgt_block
    new_cfg["vocab_size"] = vocab
    if tgt_block > cfg["block_size"]:
        reps = -(-tgt_block // cfg["block_size"])
        pos = np.tile(weights["pos_emb.weight"], (reps, 1))[:tgt_block]
        weights = dict(weights)
        weights["pos_emb.weight"] = pos

    new_weights, f = widen_weights(weights, cfg, args.target_n_layer, args.target_n_embd,
                                   tgt_block, args.new_weight_std)
    total = int(sum(v.size for v in new_weights.values()))
    print(f"الهدف: {new_cfg['n_layer']} طبقة × {new_cfg['n_embd']} بعد × {new_cfg['n_head']} رؤوس "
          f"(head_dim {cfg['n_embd'] // cfg['n_head']}) | عامل التربيع ×{f} | {total:,} معامل")

    if not args.no_verify:
        prompts = ["ازيك", "الشمس بتطلع من", "2+2=", "اسمك ايه"]
        diff, same = verify_exact(args.src, new_weights, new_cfg, src["tokens"], prompts)
        print(f"التحقق: أقصى فرق logits = {diff:.2e} | نفس الاختيار: {same}")
        assert same and diff < 1e-3, "النقل مش متطابق — ممنوع الحفظ"

    metrics = dict(src["metrics"] or {})
    metrics.update({"widened_from": src["step"],
                    "widened_to": f"{new_cfg['n_layer']}L{new_cfg['n_embd']}d{new_cfg['n_head']}h",
                    "note": "خ24: توسيع النموذج الواحد بنقل دقيق (grow_model.py)"})
    save_checkpoint(args.out, new_weights, new_cfg, src["tokens"], src["step"], metrics)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"✅ اتحفظ في {args.out} ({size_mb:.0f} MB) بنفس خطوات التدريب {src['step']:,}")


if __name__ == "__main__":
    main()
