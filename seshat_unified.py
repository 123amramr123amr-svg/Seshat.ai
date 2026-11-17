# -*- coding: utf-8 -*-
"""
seshat_unified.py — النموذج الموحّد seshat 1.0
==============================================
جولة خ8 (التوحيد): سي شات (الفهم والحوار والحساب والتعلم والبحث) وخوفو (توليد
الصور DDPM + الرسم الإجرائي) بقوا نموذج واحد باسم **seshat 1.0** — مدخل واحد،
تحليل نية واحد، وذاكرة واحدة:

  1. تحليل موحّد: كل رسالة بتتحلل مرة واحدة (أدب → حساب → صورة → تعلم →
     معرفة → بحث)، وطلب الصور بيمر على نفس خط الفهم قبل ما يوصل لخوفو.
  2. صور محلية 100%: خوفو بيفهم الوصف العربي/الإنجليزي مباشرة (مرادفات +
     تركيبات مركبة + دلوّات دلالية) ويرسم على 750×750 — مفيش pollinations
     ولا أي خدمة خارجية ولا مفاتيح ولا انتظار شبكة (الأسرع والمخصوص).
  3. العقل والذاكرة مشتركين: اللي اتعلمه seshat (learned.json) وبذور الصور
     ومعرفت الأصناف كلها في نفس المستودع ونفس السيرفر.

الاستخدام:
    from seshat_unified import SeshatOne
    one = SeshatOne(engine)          # engine = محرك seshat النصي (يمكن None)
    out = one.reply("ارسم قارب في البحر وقت الغروب")
    out["image"]                     # → "generated/<id>.png" من خوفو نفسه
"""

import os

from brain import Brain, BrainConfig, LearnedMemory

MODEL_NAME = "seshat 1.0"
MODEL_TAGLINE = ("النموذج الموحّد: فهم سي شات النصي + توليد صور خوفو — "
                 "محلي بالكامل من غير أي خدمة خارجية")


class SeshatOne(Brain):
    """نموذج واحد بيسمع ويفهم ويرد ويرسم. كل عقد Brain القديم شغال زي ما هو
    (reply / images / cfg / learned / status) — والجديد: هوية موحدة وحالة
    بتوضح إن الصور جاية من خوفو المحلي."""

    def __init__(self, engine=None):
        super().__init__(engine)
        self.model_name = MODEL_NAME
        # خ8: هوية seshat 1.0 — بتجاوب على «انت مين» من هنا على طول
        # (تعريف موحد واحد للنص والصور معًا).
        self.IDENTITY_AR = (
            "أنا seshat 1.0 — النموذج الموحّد: عقل سي شات للفهم والحوار والحساب "
            "والتعلم والبحث، ومولّد الصور خوفو شغال جوايا وبرسم بنفسي على 750×750 "
            "من غير أي موقع خارجي. مطوّريني عمرو أحمد، وكل حاجة فيا محلية.")
        self.IDENTITY_EN = (
            "I am seshat 1.0 — the unified model: the seshat text brain (chat, math, "
            "learning, search) plus the khufu image generator in one. I draw everything "
            "myself, fully local. Developed by Amr Ahmed.")

    def status(self):
        st = super().status()
        st["model"] = MODEL_NAME
        st["unified"] = True
        st["text_engine"] = self.engine.engine_name if (
            self.engine is not None and hasattr(self.engine, "engine_name")) else "محلي"
        st["image_engine"] = "خوفو (محلي — مفيش خدمات خارجية)"
        return st

    def identify(self):
        return {
            "name": MODEL_NAME,
            "developer": "عمرو أحمد (Amr Ahmed)",
            "tagline": MODEL_TAGLINE,
            "parts": {
                "فهم وحوار وحساب وتعلم": "سي شات (النموذج النصي المحلي)",
                "توليد الصور": "خوفو (DDPM + الرسم الإجرائي الهجين — 750×750)",
            },
            "external_services": "ولا حاجة — كل حاجة محلية",
        }
