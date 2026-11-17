"""Central catalog for the seshat.chet model family.

Only models that are actually wired to an inference backend should be marked active.
Add future model versions here when their checkpoints and runtime adapters exist.
"""

MODEL_CATALOG = (
    {
        "id": "seshat-1.0",
        "name": "seshat 1.0",
        "family": "seshat.chet",
        "status": "active",
        "engine": "local",
        "description": "نموذج المحادثة المحلي (1.33M معامل، سياق 4096 حرف) مع المخ والبحث والتعلم الذاتي.",
    },
    {
        "id": "nawah-1.0",
        "name": "نواة-1 (nawah 1.0)",
        "family": "seshat.chet",
        "status": "active",
        "engine": "local-nlp-pipeline",
        "description": "البنية التحتية المتقدمة لتحليل الكلمات العربية والانتباه الذاتي 16D واستخراج المفاهيم وقياس الأصالة.",
    },
)


def list_models():
    """Return JSON-safe model metadata for the web interface."""
    return [dict(model) for model in MODEL_CATALOG]


def get_model(model_id):
    """Return a registered model or None; never silently select an unknown model."""
    for model in MODEL_CATALOG:
        if model["id"] == model_id:
            return dict(model)
    return None
