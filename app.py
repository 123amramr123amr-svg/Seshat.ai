"""
سيرفر seshat.chet — بيعرض الواجهة ويرد على رسائل المستخدم

التشغيل:
    python app.py
وبعدين افتح  http://localhost:5000
"""

import os

from flask import Flask, jsonify, render_template, request

from engine import Seshat

app = Flask(__name__)
app.json.ensure_ascii = False           # عشان العربي يطلع عربي في الـ JSON

print("⏳ جاري تحميل النموذج...")
seshat = Seshat()
print(f"✅ اتحمل النموذج — {seshat.num_params():,} معامل | المحرك: {seshat.engine_name}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/info")
def info():
    return jsonify(seshat.info())


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
    return jsonify(seshat.reply(msg, temperature=temperature))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 شغال على http://localhost:{port}")
    # debug=False: وضع الـ debug بيفتح ثغرة تنفيذ أكواد لو السيرفر اتنشر على النت
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
