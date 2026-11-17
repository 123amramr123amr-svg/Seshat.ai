"""
سيرفر Seshat.ai — واجهة Flask.

التشغيل:
    python app.py
ثم افتح http://localhost:5000

إعدادات النشر:
    SESHAT_ADMIN_TOKEN=<long-random-secret>  # مطلوب لتعديل /config
    SESHAT_CHAT_RATE_LIMIT=30                # طلبًا لكل IP خلال 60 ثانية
    SESHAT_MAX_MESSAGE_CHARS=8000            # أقصى طول للرسالة (السياق الاتسع)
"""

import hmac
import os
import threading
import time
from collections import deque

from flask import Flask, jsonify, render_template, request, send_file

import vision
from engine import Seshat, drawings, _trim_memory
from brain import Brain
from models import list_models

# ---------------- خوفو (نموذج الصور) ----------------
import sys
import uuid
KHUFU_DIR = os.path.join(os.path.dirname(__file__), "khufu")
if os.path.exists(KHUFU_DIR):
    sys.path.insert(0, KHUFU_DIR)
    try:
        import scenes as khufu_scenes
        from generate import generate as khufu_generate
        KHUFU_READY = True
    except Exception:
        KHUFU_READY = False
else:
    KHUFU_READY = False

KHUFU_GEN_DIR = os.path.join(KHUFU_DIR, "data", "generated") if KHUFU_READY else None
if KHUFU_GEN_DIR:
    os.makedirs(KHUFU_GEN_DIR, exist_ok=True)
_khufu_jobs = {}
_khufu_lock = threading.Lock()

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


@app.route("/generated/<name>")
def generated_image(name):
    """
    الصورة المتولدة. بتتولد أول مرة حد يطلبها (ممكن تاخد لحد دقيقة أو اتنين لو فيه دور)،
    وبعد كده بتتبعت من الملف المتخزن. لو فشلت → JSON فيه رسالة للمستخدم.
    الطلبات بتتسجل من /chat بس، فالعنوان ده ميقدرش يولّد حاجة من عنده.
    """
    image_id = name[:-4] if name.endswith(".jpg") else ""
    if brain.images.meta(image_id) is None:
        return jsonify({"error": "الصورة دي مش موجودة."}), 404
    path, error = brain.images.ensure(image_id)
    if path is None:
        return jsonify({"error": error}), 503
    kind = (brain.images.meta(image_id) or {}).get("kind", "jpeg")
    response = send_file(path, mimetype="image/" + kind, max_age=3600)
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.route("/models")
def models_route():
    """List only model versions that have a registered runtime."""
    return jsonify({"models": list_models(), "default": "seshat-1.0"})


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
        max_chars = max(100, min(20000, int(os.environ.get("SESHAT_MAX_MESSAGE_CHARS", "8000"))))
    except ValueError:
        max_chars = 8000
    if len(msg) > max_chars:
        return jsonify({"error": f"الرسالة أطول من الحد المسموح ({max_chars} حرف)."}), 413

    image, video = data.get("image"), data.get("video")
    t0 = time.time()

    last_image = data.get("image_id") if isinstance(data.get("image_id"), str) else None
    if isinstance(image, dict) and not isinstance(video, dict) and msg \
            and os.environ.get("SESHAT_LOCAL_ONLY", "0") != "1":
        # فيه صورة مرفوعة والكلام طلب صورة جديدة ("ارسم قطة") → مش تعديل على المرفوعة.
        # الصيغ المحتملة ("اعمل صورة أبيض وأسود") بتفضل تعديل لو فيها تعديل معروف.
        from brain import try_draw
        wanted = try_draw(msg)
        if wanted and (wanted["strong"] or not vision.find_edits(msg)[0]):
            image = None

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
            out = brain.reply(msg, temperature=temperature, last_image=last_image)
        else:
            out = seshat.reply(msg, temperature=temperature)
    except Exception:
        app.logger.exception("حصلت مشكلة وقت معالجة طلب الشات")
        return jsonify({"response": "حصلت مشكلة جوه السيرفر. جرب تاني بعد شوية.",
                        "thinking": "", "confidence": 0.0, "seconds": 0, "fallback": True,
                        "source": "none", "matched": "", "suggestion": ""}), 500
    
    # Optional: Enhance with Nawah (البنية التحتية المتقدمة)
    if os.environ.get("SESHAT_USE_NAWAH", "0") == "1" and out.get("response"):
        try:
            from nawah_integration import get_nawah_brain
            nawah_brain = get_nawah_brain()
            enhanced = nawah_brain.enhance_reply(msg, out.get("response", ""), temperature, is_fallback=bool(out.get("fallback")))
            out["response"] = enhanced["reply"]
            out["nawah_analysis"] = enhanced["analysis"]
        except Exception as e:
            app.logger.warning(f"Nawah enhancement failed: {e}")
    
    if isinstance(image, dict) and not last_image and out.get("source") == "none" \
            and not out.get("suggestion"):
        out["response"] = vision.help_text(vision.is_arabic(msg))
    return jsonify(out)


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "الملف أو الرسالة أكبر من الحد المسموح."}), 413


# ---------------- routes لخوفو (نموذج الصور) ----------------

@app.route("/khufu-app")
def khufu_app():
    if not KHUFU_READY:
        return "خوفو مش جاهز — تأكد إن مجلد khufu موجود", 500
    khufu_html = os.path.join(app.static_folder, "khufu", "index.html")
    if not os.path.exists(khufu_html):
        return "ملف واجهة خوفو مش موجود", 404
    return send_file(khufu_html, mimetype="text/html")


@app.route("/khufu-generate", methods=["POST"])
def khufu_gen():
    if not KHUFU_READY:
        return jsonify(error="خوفو مش جاهز"), 500
    data = request.get_json(force=True)
    subject = str(data.get("subject", "")).strip()
    size = int(data.get("target_size") or data.get("size", 64))   # خ2: المقاس النهائي نفسه (حتى 750 —
    # المكبّر Lanczos جوا generate.py بيكبّر من قاعدة 48)
    steps = int(data.get("steps", 50))
    seed = data.get("seed")
    seed = int(seed) if seed not in (None, "") else None
    if size not in khufu_scenes.SIZES:          # خوفو بيحدد المقاسات بنفسه (حتى 750 في خ2)
        return jsonify(error="الأبعاد لازم تكون واحدة من "
                             + " أو ".join(str(s) for s in khufu_scenes.SIZES)), 400
    if steps not in (20, 50, 100):
        return jsonify(error="الجودة لازم تكون 20 أو 50 أو 100 خطوة"), 400
    # خ6: النص الخام بيمر لخوفو زي ما هو — generate جوه خوفو بيحل بنفسه:
    # صنف واحد → مسار الصنف العادي، ووصف حر فيه أكتر من عنصر → تركيبة مركبة.
    # (خوفو عمره ما بيرفض — أقرب صنف دايمًا موجود جوّاه).
    cid = str(subject).strip()
    jid = uuid.uuid4().hex[:12]
    with _khufu_lock:
        _khufu_jobs[jid] = {"status": "running", "cid": cid, "size": size,
                           "steps": steps, "seed": seed}

    def run():
        try:
            img, _cid = khufu_generate(cid, size=size, steps=steps, seed=seed)
            path = os.path.join(KHUFU_GEN_DIR, jid + ".png")
            img.save(path)
            with _khufu_lock:
                _khufu_jobs[jid].update(status="done", path=path,
                                         kb=max(1, os.path.getsize(path) // 1024))
        except Exception as e:
            with _khufu_lock:
                _khufu_jobs[jid].update(status="error", error=str(e))

    threading.Thread(target=run, daemon=True).start()
    return jsonify(id=jid)


@app.route("/khufu-generated/<jid>.png")
def khufu_get_img(jid):
    with _khufu_lock:
        job = _khufu_jobs.get(jid)
    if not job:
        return "مش موجود", 404
    if job["status"] == "running":
        return jsonify(status="running"), 202
    if job["status"] == "error":
        return jsonify(error=job.get("error", "حصل خطأ")), 500
    return send_file(job["path"])


@app.route("/khufu-status")
def khufu_status():
    if not KHUFU_READY:
        return jsonify(ready=False, error="خوفو مش جاهز")
    status_path = os.path.join(KHUFU_DIR, "data", "status.json")
    try:
        import json as _json
        with open(status_path, encoding="utf-8") as f:
            st = _json.load(f)
    except Exception:
        st = {"step": 0, "loss": None}
    st["ready"] = os.path.exists(os.path.join(KHUFU_DIR, "data", "khufu_latest.pt"))
    st["classes"] = [{"id": c[0], "ar": c[1]} for c in khufu_scenes.CLASSES]
    return jsonify(st)




# ---------------- routes لنواة-1 (البنية التحتية المتقدمة) ----------------

@app.route("/nawah-status")
def nawah_status():
    try:
        from nawah_integration import get_nawah_brain
        nb = get_nawah_brain()
        st = nb.get_stats()
        st["ready"] = True
        st["enabled_in_chat"] = os.environ.get("SESHAT_USE_NAWAH", "0") == "1"
        return jsonify(st)
    except Exception as e:
        return jsonify(ready=False, error=str(e)), 500


@app.route("/nawah-analyze", methods=["POST"])
def nawah_analyze():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text") or data.get("question") or "").strip()
    if not text:
        return jsonify(error="اكتب نصًا أو سؤالًا للتحليل أولًا"), 400
    try:
        temperature = min(1.5, max(0.0, float(data.get("temperature", 0.75))))
    except (TypeError, ValueError):
        temperature = 0.75
    try:
        from nawah_integration import get_nawah_brain
        nb = get_nawah_brain()
        result = nb.analyze_only(text, temperature=temperature, save_to_db=True)
        return jsonify(result)
    except Exception as e:
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 شغال على http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
