import re
from pathlib import Path

import flask
from catboost import CatBoostClassifier
from flask import jsonify, request

app = flask.Flask(__name__)
app.config["DEBUG"] = False

MODEL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
MODELS_ROOT = Path(__file__).resolve().parent / "models"
MODEL_CACHE = {}


def _error(message: str, status_code: int = 400):
    return jsonify({"error": message}), status_code


def _normalize_features(payload):
    if not isinstance(payload, list) or not payload:
        raise ValueError("Request body must be a non-empty JSON array.")

    # Support both one sample ([...]) and batch ([[...], [...]]).
    if isinstance(payload[0], list):
        rows = payload
    else:
        rows = [payload]

    normalized = []
    for row in rows:
        if not isinstance(row, list) or not row:
            raise ValueError(
                "Each sample must be a non-empty array of numeric features."
            )
        try:
            normalized.append([float(value) for value in row])
        except (TypeError, ValueError):
            raise ValueError("All feature values must be numeric.")
    return normalized


def _load_model(model_name: str):
    if not MODEL_NAME_RE.match(model_name):
        raise ValueError("Invalid model name.")

    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    model_path = MODELS_ROOT / model_name / "model"
    if not model_path.exists():
        raise FileNotFoundError(f"Model '{model_name}' not found.")

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    MODEL_CACHE[model_name] = model
    return model


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict/<model_name>", methods=["POST"])
def predict(model_name):
    if not request.is_json:
        return _error("Content-Type must be application/json.", 415)

    payload = request.get_json(silent=True)
    if payload is None:
        return _error("Invalid JSON payload.", 400)

    try:
        features = _normalize_features(payload)
        model = _load_model(model_name)
        predictions = model.predict(features).tolist()
        return jsonify(
            {
                "model": model_name,
                "count": len(predictions),
                "predictions": predictions,
            }
        )
    except ValueError as exc:
        return _error(str(exc), 400)
    except FileNotFoundError as exc:
        return _error(str(exc), 404)
    except Exception as exc:
        return _error(f"Prediction failed: {str(exc)}", 500)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
