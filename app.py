"""YOLO Animal Detection web application.

Two trained weights are served by this app and must not be swapped:
  * Rectangle (object detection) -> models/box_stage2_best.pt
  * Polygon (instance segmentation) -> models/poly_stage1_best.pt

Uploaded images are processed entirely in memory. No user image is written to
disk, so there are no temporary upload files to clean up after a prediction.
"""

from __future__ import annotations

import base64
import io
import os
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError

# Ultralytics monkey-patches PIL.Image.open to lazily install HEIC support when
# a file cannot be identified. Keep the original opener for upload validation so
# an invalid file is rejected immediately instead of triggering a runtime pip
# install (which may be unavailable on a deployed host). This must be captured
# before importing ultralytics.
_PIL_IMAGE_OPEN = Image.open
from ultralytics import YOLO  # noqa: E402  (must follow the opener capture)

BASE_DIR = Path(__file__).resolve().parent

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB per uploaded image
MAX_PIXELS = 25_000_000  # Guards against decompression bombs
CONFIDENCE = 0.25
IMAGE_SIZE = 640

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}
# The four classes present in the training dataset (classes.txt: 0 cat, 1 cow,
# 2 dog, 3 lion). Names are read from each model and checked against this set;
# IDs are never remapped here.
EXPECTED_CLASSES = {"cat", "cow", "dog", "lion"}

MODEL_SPECS = {
    "box": {
        "name": "Rectangle",
        "task": "detect",
        "task_label": "Object Detection",
        "stage": "Stage 2",
        "weight": "box_stage2_best.pt",
        "path": BASE_DIR / "models" / "box_stage2_best.pt",
        "description": "Draws a bounding box around each detected animal.",
    },
    "poly": {
        "name": "Polygon",
        "task": "segment",
        "task_label": "Instance Segmentation",
        "stage": "Stage 1",
        "weight": "poly_stage1_best.pt",
        "path": BASE_DIR / "models" / "poly_stage1_best.pt",
        "description": "Marks the visible shape of each animal with a segmentation mask.",
    },
}

CLASS_INFO = {
    "cat": {"label": "Cat", "emoji": "🐱"},
    "cow": {"label": "Cow", "emoji": "🐮"},
    "dog": {"label": "Dog", "emoji": "🐶"},
    "lion": {"label": "Lion", "emoji": "🦁"},
}

SAMPLES = [
    {"name": "Cat", "class": "cat", "file": "randomc_11.jpg"},
    {"name": "Cow", "class": "cow", "file": "randomw_14.jpeg"},
    {"name": "Dog", "class": "dog", "file": "randomd_17.jpeg"},
    {"name": "Lion", "class": "lion", "file": "randoml_100.jpeg"},
]

app = Flask(__name__)
# Leave headroom for multipart overhead; the exact 10 MB limit is enforced on
# the uploaded file bytes inside predict().
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE + 1024 * 1024
app.json.ensure_ascii = False

models: dict[str, YOLO] = {}
_model_locks = {key: threading.Lock() for key in MODEL_SPECS}


def load_models() -> None:
    """Load both trained weights once, at application startup."""
    for key, spec in MODEL_SPECS.items():
        path: Path = spec["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Required model weight is missing: {spec['weight']}")

        model = YOLO(str(path))
        task = getattr(model, "task", None)
        if task != spec["task"]:
            raise RuntimeError(
                f"Wrong weight for {spec['name']}: expected a {spec['task']} model, "
                f"but {spec['weight']} is a {task} model."
            )

        names = {str(name).lower() for name in model.names.values()}
        if names != EXPECTED_CLASSES:
            raise RuntimeError(
                f"{spec['weight']} has classes {sorted(names)}, but this app requires "
                f"exactly {sorted(EXPECTED_CLASSES)}."
            )

        models[key] = model
        app.logger.info("Loaded %s model from %s", spec["name"], spec["weight"])


def _client_error(message: str, status: int = 400):
    return jsonify(error=message), status


def _validate_upload(file_storage):
    """Return (image, error_message). Validation happens before prediction."""
    filename = (file_storage.filename or "").strip()
    if not filename:
        return None, "Please choose an image first."

    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return None, "Unsupported file type. Please upload a JPG, JPEG or PNG image."

    data = file_storage.read(MAX_FILE_SIZE + 1)
    if len(data) > MAX_FILE_SIZE:
        return None, "The image is larger than 10 MB. Please upload a smaller file."
    if not data:
        return None, "The uploaded file is empty."

    try:
        with _PIL_IMAGE_OPEN(io.BytesIO(data)) as probe:
            image_format = (probe.format or "").upper()
            probe.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        return None, "This file is not a valid image. Please upload a JPG, JPEG or PNG image."

    if image_format not in ALLOWED_FORMATS:
        return None, "Unsupported image format. Please upload a JPG, JPEG or PNG image."

    try:
        image = _PIL_IMAGE_OPEN(io.BytesIO(data))
        if image.width * image.height > MAX_PIXELS:
            return None, "The image dimensions are too large. Please upload a smaller image."
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return None, "This image could not be opened. Please try another JPG, JPEG or PNG file."

    if image.width < 8 or image.height < 8:
        return None, "The image is too small. Please upload an image at least 8 x 8 pixels."

    return image, None


def _run_prediction(mode: str, image: Image.Image):
    spec = MODEL_SPECS[mode]
    model = models[mode]

    started = time.perf_counter()
    with _model_locks[mode]:
        result = model.predict(
            source=image,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            verbose=False,
        )[0]
    inference_ms = round((time.perf_counter() - started) * 1000)

    # Ultralytics plots on a BGR array; convert it back to RGB for PIL.
    plotted = result.plot()
    plotted_image = Image.fromarray(plotted[:, :, ::-1])
    buffer = io.BytesIO()
    plotted_image.save(buffer, format="JPEG", quality=90, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    processing_ms = round((time.perf_counter() - started) * 1000)

    names = result.names
    detections = []
    counts: dict[str, int] = {}
    for index, box in enumerate(result.boxes):
        class_id = int(box.cls[0].item())
        class_name = str(names.get(class_id, class_id))
        confidence = round(float(box.conf[0].item()) * 100, 1)
        detection = {
            "index": index + 1,
            "class": class_name,
            "confidence": confidence,
        }
        if mode == "poly":
            detection["has_mask"] = bool(
                result.masks is not None
                and result.masks.data is not None
                and index < len(result.masks.data)
            )
        detections.append(detection)
        counts[class_name] = counts.get(class_name, 0) + 1

    return {
        "image": "data:image/jpeg;base64," + encoded,
        "detections": detections,
        "counts": counts,
        "total": len(detections),
        "mode": mode,
        "model": spec["name"],
        "task": spec["task_label"],
        "stage": spec["stage"],
        "processing_time_ms": processing_ms,
        "inference_time_ms": inference_ms,
        "confidence_threshold": CONFIDENCE,
    }


@app.route("/")
def index():
    model_cards = [
        {
            "key": key,
            "name": spec["name"],
            "task": spec["task_label"],
            "stage": spec["stage"],
            "weight": spec["weight"],
            "description": spec["description"],
        }
        for key, spec in MODEL_SPECS.items()
    ]
    classes = [
        {"key": key, "label": info["label"], "emoji": info["emoji"]}
        for key, info in CLASS_INFO.items()
    ]
    return render_template(
        "index.html",
        samples=SAMPLES,
        classes=classes,
        model_cards=model_cards,
        model_count=len(models),
        confidence=CONFIDENCE,
        image_size=IMAGE_SIZE,
        max_file_mb=MAX_FILE_SIZE // (1024 * 1024),
    )


@app.route("/health")
def health():
    return jsonify(
        status="ok" if len(models) == len(MODEL_SPECS) else "degraded",
        models_loaded=len(models),
        models_expected=len(MODEL_SPECS),
        classes=sorted(EXPECTED_CLASSES),
    )


@app.route("/predict", methods=["POST"])
def predict():
    mode = (request.form.get("mode") or "box").lower()
    if mode not in MODEL_SPECS:
        return _client_error("Please select a valid prediction model.")

    upload = request.files.get("image")
    if upload is None:
        return _client_error("Please choose an image first.")

    image, error = _validate_upload(upload)
    if error:
        return _client_error(error)

    try:
        payload = _run_prediction(mode, image)
        return jsonify(payload)
    except Exception:
        # Log the full traceback server-side but never expose filesystem paths
        # or library details to the browser.
        app.logger.exception("Prediction failed for mode=%s", mode)
        return _client_error(
            "Prediction failed while processing this image. "
            "Please try again with a different JPG, JPEG or PNG file.",
            500,
        )


@app.errorhandler(413)
def request_too_large(_error):
    return _client_error("The image is larger than 10 MB. Please upload a smaller file.", 413)


@app.errorhandler(404)
def not_found(_error):
    if request.path.startswith(("/predict", "/health")):
        return _client_error("The requested endpoint was not found.", 404)
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(_error):
    return _client_error("An unexpected server error occurred. Please try again.", 500)


@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if request.path.startswith("/predict") or request.path == "/health":
        response.headers["Cache-Control"] = "no-store"
    return response


load_models()


if __name__ == "__main__":
    # Local development only. Production uses Gunicorn (see Dockerfile).
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
    )
