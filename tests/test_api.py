import sys
import types
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

# Replace the real predictor so the API can be tested without
# downloading CLIP or loading the ResNet weights.
FAKE_RESULT = {
    "binary_prediction": "Tumor",
    "binary_confidence": 97.5,
    "tumor_type": "Glioma",
    "tumor_confidence": 91.2,
    "final_prediction": "Glioma",
    "final_confidence": 91.2,
}

fake_predictor = types.ModuleType("backend.predictor")
fake_predictor.predict_image = lambda image: FAKE_RESULT  # type: ignore[attr-defined]
sys.modules["backend.predictor"] = fake_predictor

from backend.main import app  # noqa: E402

client = TestClient(app)


def make_png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (224, 224), "black").save(buffer, format="PNG")
    return buffer.getvalue()


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_returns_result():
    response = client.post(
        "/predict",
        files={"file": ("scan.png", make_png(), "image/png")},
    )
    assert response.status_code == 200
    assert response.json() == {"filename": "scan.png", **FAKE_RESULT}


def test_predict_rejects_non_image():
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_predict_rejects_corrupt_image():
    response = client.post(
        "/predict",
        files={"file": ("scan.png", b"not really a png", "image/png")},
    )
    assert response.status_code == 400
