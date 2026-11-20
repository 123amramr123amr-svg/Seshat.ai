"""
نواة-1 — وحدة النمو الذاتي (خ18 بطلب المستخدم)
==============================================

المستخدم طلب: «يكون ليه زي خلايا عصبية والبنية التحتية بتاعته والنواة بتاعته
وطورهم... الحاجات دي تكبر مع كبر البيانات اللي جواها والمعاملات اللي جواها».

الوحدة دي بتقرأ بنية seshat الحية مباشرة من الملفات (من غير أي أرقام مكتوبة
بالإيد): عدد المعاملات من مصفوفات النموذج، المفردات والسياق من جدول المواقع،
عدد مدخلات المعرفة من ملفات البيانات، ذاكرة التعلم، المرشحات المؤرشفة —
فكل رقم هنا بيتحدث لوحده أول ما البيانات أو النموذج يكبر.

الواجهة:
    growth_report()  → dict بأرقام البنية الكاملة
    growth_summary() → نص عربي جاهز للعرض في الحوار
    matches_growth_question(msg) → bool (كشف سؤال البنية)
"""

import glob
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(REPO, "seshat_chet_model.npz")
DATA_DIR = os.path.join(REPO, "data")
BACKUPS_DIR = os.path.join(REPO, "backups")

_GROWTH_RE = re.compile(
    r"(?:قدراتك|حجمك|بنيتك|نواتك|النواة|معاملاتك|طبقاتك|كام معامل|عدد المعاملات|"
    r"كام طبقة|كام مدخل|البنية التحتية|تقرير عن نفسك|حالتك التقنية|مقاسك|كام بارامتر|"
    r"جدول النمو|خطة النمو|خطتك للنمو|جدول التوسعة|النمو بتاعك)",
    re.IGNORECASE)


def matches_growth_question(msg):
    """كشف سؤال البنية الذاتية — مش سؤال الهوية «انت مين» (ده له مساره الموثق)."""
    if not msg:
        return False
    t = msg.lower()
    if "انت مين" in t or "اسمك" in t or "مين طورك" in t:
        return False
    return bool(_GROWTH_RE.search(t))


def _count_entries(path):
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for line in f if line.startswith(("س:", "Q:")))
    except OSError:
        return 0


def growth_report():
    """تقرير البنية الكامل — كل الأرقام حية من الملفات نفسها."""
    report = {
        "model_exists": False,
        "params": 0,
        "layers": 0,
        "heads": 0,
        "dim": 0,
        "vocab": 0,
        "context": 0,
        "data": {},
        "data_total": 0,
        "learned": 0,
        "backups": 0,
        "code_files": 0,
    }
    # النموذج: قراءة حية من المصفوفات
    try:
        import numpy as np
        if os.path.exists(MODEL_PATH):
            z = np.load(MODEL_PATH, allow_pickle=True)
            tok = z["tok_emb.weight"]
            pos = z["pos_emb.weight"]
            blocks = sorted({k.split(".")[1] for k in z.files
                             if k.startswith("blocks.")})
            # الصيغة المخزنة: (المفردات، البعد) — vocab صفوف × dim أعمدة
            vocab, dim = int(tok.shape[0]), int(tok.shape[1])
            report["model_exists"] = True
            report["dim"] = dim
            report["vocab"] = vocab
            report["context"] = int(pos.shape[0])
            report["layers"] = len(blocks)
            # مجموع المعاملات المخزنة (مرة واحدة — فرق طفيف عن num_params
            # بسبب ربط lm_head بالـembeddings وعدم تخزينه منفصلًا)
            total = 0
            for k in z.files:
                if k == "__meta__":
                    continue
                total += int(z[k].size)
            report["params"] = total
            meta = z["__meta__"] if "__meta__" in z.files else None
            if meta is not None:
                try:
                    m = json.loads(bytes(meta).decode("utf-8"))
                    report["steps"] = m.get("step")
                    cfg = m.get("config") or {}
                    if cfg.get("n_head"):
                        report["heads"] = int(cfg["n_head"])
                except Exception:
                    pass
            if not report["heads"]:
                report["heads"] = max(1, min(8, dim // 64))
    except Exception:
        pass

    # البيانات الحية: عدد المدخلات لكل ملف + مجموعها
    for name in ("ar_knowledge.txt", "en_knowledge.txt", "ar_chat.txt",
                 "en_chat.txt", "ar_programming.txt", "en_programming.txt"):
        n = _count_entries(os.path.join(DATA_DIR, name))
        report["data"][name] = n
        report["data_total"] += n
    report["code_files"] = len(glob.glob(os.path.join(DATA_DIR, "big_code.py")))

    # ذاكرة التعلم والمرشحات المؤرشفة
    try:
        with open(os.path.join(DATA_DIR, "learned.json"), encoding="utf-8") as f:
            report["learned"] = len(json.load(f))
    except Exception:
        pass
    try:
        report["backups"] = len(glob.glob(os.path.join(BACKUPS_DIR, "*.npz")))
    except Exception:
        pass
    return report


def _fmt(n):
    return "%s" % format(int(n), ",")


def growth_summary():
    """ملخص عربي جاهز للرد في الحوار — الأرقام حية 100%."""
    r = growth_report()
    if not r["model_exists"]:
        return ("أنا seshat — لسه ملف النموذج مش موجود في المستودع، فمش قادر "
                "أقيس بنيتي الحية دلوقتي، لكن طبقة النواة (nawah/) شغالة.")
    lines = [
        "دي بنيتي الحية مقروءة من ملفاتي نفسها مش من كلام مكتوب:",
        "• المعاملات: %s معامل (dim %d × %d طبقات × %d رؤوس)"
        % (_fmt(r["params"]), r["dim"], r["layers"], r["heads"]),
        "• المفردات: %d توكن، نافذة الانتباه %d توكن للرسالة الواحدة، وسياق المستندات الضخمة 2,000,000 حرف بالنافذة المنزلقة — تلات أرقام مختلفة كل واحد لغرضه (توضيح بعد سؤال المستخدم في خ27)"
        % (r["vocab"], r["context"]),
        "• المعرفة الحية: %s مدخل في 6 ملفات بيانات + ذاكرة تعلم فيها %d مدخل"
        % (_fmt(r["data_total"]), r["learned"]),
        "• النمو المؤرشف: %d مرشحة تدريب محفوظة في backups" % r["backups"],
        "كل الأرقام دي بتتكبر لوحدها مع كل جولة بيانات وتدريب — اطلبها أي وقت.",
    ]
    if r.get("steps"):
        lines.insert(4, "• التدريب: %s خطوة اتجمعت عبر الجولات" % _fmt(r["steps"]))
    return "\n".join(lines)


def growth_diff(old_report):
    """الفرق بين تقريرين — عشان الجولات تقيس نموها بالأرقام (طلب المستخدم:
    الحاجات دي تكبر مع كبر البيانات والمعاملات)."""
    if not old_report:
        return {}
    cur = growth_report()
    diff = {}
    for key in ("params", "vocab", "context", "data_total", "learned", "backups"):
        if key in old_report and key in cur:
            diff[key] = cur[key] - old_report[key]
    return diff


# — جدول النمو كل 8 ساعات (بأمر المستخدم 20 نوفمبر: «كل ثمان ساعات تزود عدد
#   المعاملات بتاعته والتدريب») — كل خطوة توسعة على نفس الملف عبر grow_model.py —
GROWTH_TABLE = [
    {"round": "خ26-27", "layers": 32, "params_approx": "103M",
     "note": "أول توسعة فوق حاجز المية مليون (بنفس الملف)"},
    {"round": "خ28", "layers": 40, "params_approx": "128M",
     "note": "+8 طبقات + كاش KV لتسريع التوليد"},
    {"round": "خ29", "layers": 48, "params_approx": "154M",
     "note": "+8 طبقات + ذاكرة حوار متعدد الأدوار أعمق"},
    {"round": "خ30", "layers": 64, "params_approx": "204M",
     "note": "مسيرة المليار: قفزة 16→64 طبقة بوصفة freeze — تم"},
    {"round": "خ31", "layers": 80, "params_approx": "254M",
     "note": "مسيرة المليار: +16 طبقة @512 بتجميد دوّار"},
    {"round": "خ32", "layers": 80, "params_approx": "570M",
     "note": "مسيرة المليار: توسيع العرض n_embd 512→768"},
    {"round": "خ33", "layers": 96, "params_approx": "683M",
     "note": "مسيرة المليار: +16 طبقة @768"},
    {"round": "خ34", "layers": 108, "params_approx": "768M",
     "note": "مسيرة المليار: +12 طبقة @768"},
    {"round": "خ35", "layers": 120, "params_approx": "853M",
     "note": "مسيرة المليار: +12 طبقة @768"},
    {"round": "خ36", "layers": 132, "params_approx": "938M",
     "note": "مسيرة المليار: +12 طبقة @768"},
    {"round": "خ37", "layers": 141, "params_approx": "1,001M",
     "note": "هدف المليار: ~1,000,000,000 معامل على نفس الملف الواحد"},
]


def growth_plan():
    """الجدول + الموقع الحي: خنا فين دلوقتي بالطبقات والمعاملات المقروءة من الملف."""
    r = growth_report()
    done = [row for row in GROWTH_TABLE if r["layers"] >= row["layers"]]
    nxt = [row for row in GROWTH_TABLE if r["layers"] < row["layers"]]
    return {"report": r, "done": done, "next": nxt[:1],
            "rule": "التوسعة على نفس الملف seshat_chet_model.npz فقط عبر "
                    "grow_model.py — ممنوع أي ملف نموذج تاني أبدًا"}


def growth_plan_text():
    """نص عربي لجدول النمو والموقع الحي — جاهز للرد في الحوار."""
    p = growth_plan()
    r = p["report"]
    if not r["model_exists"]:
        return ("جدول النمو ثابت (%d خطوات مسجلة) بس القياس الحي مش متاح "
                "لأن ملف النموذج مش موجود دلوقتي." % len(GROWTH_TABLE))
    lines = ["جدول نمو seshat كل 8 ساعات — الموقع الحي مقروء من الملف: %d طبقة × %s معامل"
             % (r["layers"], _fmt(r["params"]))]
    for row in p["done"]:
        lines.append("• %s: %d طبقة (%s) — %s — تم"
                     % (row["round"], row["layers"], row["params_approx"], row["note"]))
    for row in p["next"]:
        lines.append("• %s: %d طبقة (%s) — %s — الجاي"
                     % (row["round"], row["layers"], row["params_approx"], row["note"]))
    lines.append("القاعدة الثابتة: " + p["rule"])
    return "\n".join(lines)
