"""Central catalog for the C Chat model family.

Only models that are actually wired to an inference backend should be marked active.
Add future model versions here when their checkpoints and runtime adapters exist.
"""

MODEL_CATALOG = (
    {
        "id": "khufu-1.0",
        "name": "خوفو 1.0",
        "family": "C Chat",
        "status": "active",
        "engine": "local",
        "description": "أول نموذج محلي تجريبي ضمن منصة C Chat.",
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
