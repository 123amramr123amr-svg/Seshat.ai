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
        "description": "أول نموذج محلي تجريبي ضمن منصة seshat.chet.",
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
