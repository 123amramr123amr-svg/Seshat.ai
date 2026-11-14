"""
سيرفر seshat.chet — بيعرض الواجهة ويرد على رسائل المستخدم

التشغيل:
    python app.py
وبعدين افتح  http://localhost:5000
"""

import os

from flask import Flask, jsonify, render_template, request

from engine import Seshat
from brain import Brain

app = Flask(__name__)
app.json.ensure_ascii = False           # عشان العربي يطلع عربي في الـ JSON

print("⏳ جاري تحميل النموذج...")
seshat = Seshat()
print(f"✅ اتحمل النموذج — {seshat.num_params():,} معامل | المحرك: {seshat.engine_name}")

print("⏳ تجهيز المخ (البحث + التعلم الذاتي + الصور + الأدب)...")
brain = Brain(seshat)
st = brain.status()
print(f"✅ المخ جاهز — بحث النت: {'شغال' if st['search'] else 'واقف'} | "
      f"مخ خارجي: {st['api_model'] if st['api'] else 'مش مفعل'} | "
      f"اتعلم {st['learned']} حاجة قبل كده")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/info")
def info():
    out = seshat.info()
    out["brain"] = brain.status()
    return jsonify(out)


@app.route("/config", methods=["GET", "POST"])
def config_route():
    """إعدادات المخ: مفتاح API اختياري + البحث. المفتاح بيتخزن في
    data/config.local.json (مش بيترفع على جيت هاب) ومش بيتعرض أبدًا."""
    if request.method == "GET":
        out = brain.cfg.public()
        out["learned"] = len(brain.learned.items)
        return jsonify(out)
    data = request.get_json(silent=True) or {}
    brain.cfg.save(**{k: data[k] for k in ("api_key", "api_base", "api_model",
                                           "search_enabled") if k in data})
    out = brain.cfg.public()
    out["learned"] = len(brain.learned.items)
    return jsonify(out)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    msg = str(data.get("message", "")).strip()
    if not msg:
        return jsonify({"response": "اكتب حاجة الأول 🙂", "thinking": "", "confidence": 1.0,
                        "seconds": 0, "fallback": False, "source": "none", "matched": "",
                        "suggestion": ""})
    try:
        temperature = min(1.5, max(0.0, float(data.get("temperature", 0.0))))
    except (TypeError, ValueError):
        temperature = 0.0
    try:
        # المخ الجديد: أدب + حساب + رسم + تعلم + نموذج محلي + مخ خارجي + بحث
        # SESHAT_LOCAL_ONLY=1 يرجّع سلوك النموذج المحلي القديم لوحده
        if os.environ.get("SESHAT_LOCAL_ONLY", "0") != "1":
            return jsonify(brain.reply(msg, temperature=temperature))
        return jsonify(seshat.reply(msg, temperature=temperature))
    except Exception:                       # غلط غير متوقع → JSON واضح بدل صفحة HTML
        app.logger.exception("حصلت مشكلة وقت الرد على: %r", msg[:200])
        return jsonify({"response": "حصلت مشكلة جوه السيرفر. جرب تاني بعد شوية.",
                        "thinking": "", "confidence": 0.0, "seconds": 0, "fallback": True,
                        "source": "none", "matched": "", "suggestion": ""}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 شغال على http://localhost:{port}")
    # debug=False: وضع الـ debug بيفتح ثغرة تنفيذ أكواد لو السيرفر اتنشر على النت
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
