"""
سيرفر seshat.chet — بيعرض الواجهة ويرد على رسائل المستخدم

التشغيل:
    python app.py
وبعدين افتح  http://localhost:5000
"""

import os
import time

from flask import Flask, jsonify, render_template, request

import vision
from engine import Seshat, drawings, _trim_memory
from brain import Brain

app = Flask(__name__)
app.json.ensure_ascii = False           # عشان العربي يطلع عربي في الـ JSON
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024      # أكبر طلب: 8 ميجا (صورة أو لقطات فيديو)

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
    return render_template("index.html", palette=drawings.PALETTE, grid=drawings.N)


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
    """
    الطلب: {"message": "...", "image": {w, h, data}, "video": {duration, w, h, frames: [...]}}
    الصورة واللقطات بتيجي بكسلات خام RGBA (base64) — المتصفح هو اللي بيفك الملف ويصغّره.
    """
    data = request.get_json(silent=True) or {}
    msg = str(data.get("message", "")).strip()
    image, video = data.get("image"), data.get("video")
    t0 = time.time()

    if isinstance(video, dict) or isinstance(image, dict):
        try:
            out = None
            if isinstance(image, dict) and not isinstance(video, dict) \
                    and brain.cfg.api_key and vision.wants_look(msg):
                # فيه مخ خارجي بمفتاح → هو اللي بيشوف الصورة ويقول فيها إيه بجد
                from brain import llm_vision
                seen = llm_vision(msg, vision.jpeg_free_data_url(image), brain.cfg)
                if seen:
                    out = {"response": seen, "thinking": "المخ الخارجي شاف الصورة ووصفها",
                           "confidence": 0.9, "fallback": False, "source": "llm",
                           "source_label": "شبكة عصبية خارجية", "matched": "", "suggestion": ""}
            if out is None:
                out = (vision.handle_video(msg, video) if isinstance(video, dict)
                       else vision.handle_image(msg, image))
        except Exception as err:                    # ملف بايظ أو بيانات ناقصة
            out = {"response": f"معرفتش أقرا الملف ده ({err}). جرب صورة أو فيديو تاني.",
                   "thinking": "", "confidence": 0.0, "fallback": True, "source": "none",
                   "matched": "", "suggestion": ""}
        if out is not None:
            out["seconds"] = round(time.time() - t0, 2)
            out.setdefault("source_label", "صور" if out.get("source") == "vision" else "")
            response = jsonify(out)
            del data, image, video, out
            _trim_memory()                          # نرجّع رام البكسلات للنظام (حد الاستضافة 100 ميجا)
            return response
        # الكلام ملوش علاقة بالصورة (مثلاً "اسمك ايه") → رد الشات العادي

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
            out = brain.reply(msg, temperature=temperature)
        else:
            out = seshat.reply(msg, temperature=temperature)
    except Exception:                       # غلط غير متوقع → JSON واضح بدل صفحة HTML
        app.logger.exception("حصلت مشكلة وقت الرد على: %r", msg[:200])
        return jsonify({"response": "حصلت مشكلة جوه السيرفر. جرب تاني بعد شوية.",
                        "thinking": "", "confidence": 0.0, "seconds": 0, "fallback": True,
                        "source": "none", "matched": "", "suggestion": ""}), 500
    if isinstance(image, dict) and out.get("source") == "none" and not out.get("suggestion"):
        # معاه صورة وكلام مفهمناهوش → نقوله إيه اللي ينفع يتعمل في الصورة
        out["response"] = vision.help_text(vision.is_arabic(msg))
    return jsonify(out)


@app.errorhandler(413)
def too_large(_):
    return jsonify({"response": "الملف كبير أوي. جرب صورة أصغر.", "thinking": "", "confidence": 0.0,
                    "seconds": 0, "fallback": True, "source": "none", "matched": "",
                    "suggestion": ""}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 شغال على http://localhost:{port}")
    # debug=False: وضع الـ debug بيفتح ثغرة تنفيذ أكواد لو السيرفر اتنشر على النت
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
