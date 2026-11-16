"""Khufu web app: generate images from the local model, with size + quality controls."""
import json
import os
import threading
import uuid

from flask import Flask, abort, jsonify, request, send_file

import scenes
from generate import generate

app = Flask(__name__)
GEN_DIR = "data/generated"
CKPT = "data/khufu_latest.pt"
os.makedirs(GEN_DIR, exist_ok=True)
jobs = {}
lock = threading.Lock()


@app.route("/")
def index():
    return send_file("templates/index.html")


@app.route("/generate", methods=["POST"])
def gen():
    data = request.get_json(force=True)
    subject = str(data.get("subject", "")).strip()
    size = int(data.get("size", 64))
    steps = int(data.get("steps", 50))
    seed = data.get("seed")
    seed = int(seed) if seed not in (None, "") else None
    if size not in scenes.SIZES:            # خ2: المقاسات حتى 750 من scenes نفسها
        return jsonify(error="الأبعاد لازم تكون واحدة من "
                             + " أو ".join(str(s) for s in scenes.SIZES)), 400
    if steps not in (20, 50, 100):
        return jsonify(error="الجودة لازم تكون 20 أو 50 أو 100 خطوة"), 400
    cid = subject if subject in scenes.CLASS_IDS else scenes.class_from_text(subject)
    if cid is None:
        return jsonify(error="مش عارف أرسم «" + subject + "» — اختار حاجة من القائمة"), 400
    if not os.path.exists(CKPT):
        return jsonify(error="خوفو لسه بيتدرب — مفيش نموذج جاهز بعد"), 400
    jid = uuid.uuid4().hex[:12]
    with lock:
        jobs[jid] = {"status": "running", "cid": cid, "size": size, "steps": steps, "seed": seed}

    def run():
        try:
            img, _cid = generate(cid, size=size, steps=steps, seed=seed)
            path = os.path.join(GEN_DIR, jid + ".png")
            img.save(path)
            with lock:
                jobs[jid].update(status="done", path=path,
                                 kb=max(1, os.path.getsize(path) // 1024))
        except Exception as e:  # noqa: BLE001
            with lock:
                jobs[jid].update(status="error", error=str(e))

    threading.Thread(target=run, daemon=True).start()
    return jsonify(id=jid)


@app.route("/generated/<jid>.png")
def get_img(jid):
    with lock:
        job = jobs.get(jid)
    if not job:
        abort(404)
    if job["status"] == "running":
        return jsonify(status="running"), 202
    if job["status"] == "error":
        return jsonify(error=job.get("error", "حصل خطأ")), 500
    return send_file(job["path"])


@app.route("/status")
def status():
    try:
        with open("data/status.json", encoding="utf-8") as f:
            st = json.load(f)
    except Exception:
        st = {"step": 0, "loss": None}
    st["ready"] = os.path.exists(CKPT)
    st["classes"] = [{"id": c[0], "ar": c[1]} for c in scenes.CLASSES]
    return jsonify(st)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))
