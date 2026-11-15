"""
سيرفر Seshat.ai — واجهة Flask.

التشغيل:
    python app.py
ثم افتح http://localhost:5000

إعدادات النشر:
    SESHAT_ADMIN_TOKEN=<long-random-secret>  # مطلوب لتعديل /config
    SESHAT_CHAT_RATE_LIMIT=30                # طلبًا لكل IP خلال 60 ثانية
    SESHAT_MAX_MESSAGE_CHARS=4000
"""

import hmac
import os
import threading
import time
from collections import deque

from flask import Flask, jsonify, render_template, request

import vision
from engine import Seshat, drawings, _trim_memory
from brain import Brain
from models import list_models

app = Flask(__name__)
app.json.ensure_ascii = False
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

# محدد معدل بسيط داخل العملية. في النشر متعدد العمال استخدم محددًا مشتركًا
# (مثل Redis) أمام التطبيق، ولا تعتمد على هذا المحدد وحده.
_RATE_WINDOW_SECONDS = 60
_rate_lock = threading.Lock()
_chat_requests = {}


def _chat_rate_limit():
    """يسمح بعدد محدود من طلبات الشات لكل عنوان IP في نافذة زمنية متحركة."""
    try:
        limit = max(1, min(300, int(os.environ.get("SESHAT_CHAT_RATE_LIMIT", "30"))))
    except ValueError:
        limit = 30
    now = time.monotonic()
    client = request.remote_addr or "unknown"
    with _rate_lock:
        # تنظيف دوري للمدخلات القديمة والحد من نمو الذاكرة.
        stale_before = now - _RATE_WINDOW_SECONDS
        for ip in list(_chat_requests):
            q = _chat_requests[ip]
            while q and q[0] <= stale_before:
                q.popleft()
            if not q:
                del _chat_requests[ip]
        if client not in _chat_requests and len(_chat_requests) >= 10000:
            return False
        q = _chat_requests.setdefault(client, deque())
        if len(q) >= limit:
            return False
        q.append(now)
    return True


def _authorized_config_write():
    """لا تسمح بتغيير إعدادات المخ إلا بتوكن إداري مضبوط في بيئة التشغيل."""
    expected = os.environ.get("SESHAT_ADMIN_TOKEN", "")
    supplied = request.headers.get("Authorization", "")
    if not expected:
        return False
    prefix = "Bearer "
    if not supplied.startswith(prefix):
        return False
    return hmac.compare_digest(supplied[len(prefix):], expected)


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


@app.route("/models")
def models_route():
    """List only model versions that have a registered runtime."""
    return jsonify({"models": list_models(), "default": "khufu-1.0"})


@app.route("/info")
def info():
    out = seshat.info()
    out["brain"] = brain.status()
    return jsonify(out)


@app.route("/config", methods=["GET", "POST"])
def config_route():
    """GET يعرض الإعدادات العامة؛ POST يتطلب Authorization: Bearer."""
    if request.method == "GET":
        out = brain.cfg.public()
        out["learned"] = len(brain.learned.items)
        return jsonify(out)

    if not _authorized_config_write():
        # لا نكشف هل التوكن غير مضبوط أم غير صحيح.
        return jsonify({"error": "غير مصرح بتعديل الإعدادات."}), 401

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "محتوى JSON غير صالح."}), 400
    allowed = ("api_key", "api_base", "api_model", "search_enabled")
    brain.cfg.save(**{k: data[k] for k in allowed if k in data})
    out = brain.cfg.public()
    out["learned"] = len(brain.learned.items)
    return jsonify(out)


@app.route("/chat", methods=["POST"])
def chat():
    """
    الطلب: {"message": "...", "image": {...}, "video": {...}}
    الصور والفيديو تظل خاضعة لحد حجم الطلب العام (8 ميجا).
    """
    if not _chat_rate_limit():
        return jsonify({"error": "طلبات كثيرة. انتظر قليلًا ثم حاول مرة أخرى."}), 429

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "محتوى JSON غير صالح."}), 400

    raw_msg = data.get("message", "")
    if not isinstance(raw_msg, str):
        return jsonify({"error": "يجب أن تكون الرسالة نصًا."}), 400
    msg = raw_msg.strip()
    try:
        max_chars = max(100, min(20000, int(os.environ.get("SESHAT_MAX_MESSAGE_CHARS", "4000"))))
    except ValueError:
        max_chars = 4000
    if len(msg) > max_chars:
        return jsonify({"error": f"الرسالة أطول من الحد المسموح ({max_chars} حرف)."}), 413

    image, video = data.get("image"), data.get("video")
    t0 = time.time()

    if isinstance(video, dict) or isinstance(image, dict):
        try:
            out = None
            if isinstance(image, dict) and not isinstance(video, dict) \
                    and brain.cfg.api_key and vision.wants_look(msg):
                from brain import llm_vision
                seen = llm_vision(msg, vision.jpeg_free_data_url(image), brain.cfg)
                if seen:
                    out = {"response": seen, "thinking": "المخ الخارجي شاف الصورة ووصفها",
                           "confidence": 0.9, "fallback": False, "source": "llm",
                           "source_label": "شبكة عصبية خارجية", "matched": "", "suggestion": ""}
            if out is None:
                out = (vision.handle_video(msg, video) if isinstance(video, dict)
                       else vision.handle_image(msg, image))
        except Exception:
            app.logger.exception("فشل تحليل صورة أو فيديو")
            out = {"response": "معرفتش أقرا الملف ده. جرب صورة أو فيديو تاني.",
                   "thinking": "", "confidence": 0.0, "fallback": True, "source": "none",
                   "matched": "", "suggestion": ""}
        if out is not None:
            out["seconds"] = round(time.time() - t0, 2)
            out.setdefault("source_label", "صور" if out.get("source") == "vision" else "")
            response = jsonify(out)
            del data, image, video, out
            _trim_memory()
            return response

    if not msg:
        return jsonify({"response": "اكتب حاجة الأول 🙂", "thinking": "", "confidence": 1.0,
                        "seconds": 0, "fallback": False, "source": "none", "matched": "",
                        "suggestion": ""})
    try:
        temperature = min(1.5, max(0.0, float(data.get("temperature", 0.0))))
    except (TypeError, ValueError):
        temperature = 0.0
    try:
        if os.environ.get("SESHAT_LOCAL_ONLY", "0") != "1":
            out = brain.reply(msg, temperature=temperature)
        else:
            out = seshat.reply(msg, temperature=temperature)
    except Exception:
        app.logger.exception("حصلت مشكلة وقت معالجة طلب الشات")
        return jsonify({"response": "حصلت مشكلة جوه السيرفر. جرب تاني بعد شوية.",
                        "thinking": "", "confidence": 0.0, "seconds": 0, "fallback": True,
                        "source": "none", "matched": "", "suggestion": ""}), 500
    if isinstance(image, dict) and out.get("source") == "none" and not out.get("suggestion"):
        out["response"] = vision.help_text(vision.is_arabic(msg))
    return jsonify(out)


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "الملف أو الرسالة أكبر من الحد المسموح."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 شغال على http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
