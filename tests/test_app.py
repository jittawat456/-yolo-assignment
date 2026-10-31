"""Smoke tests for the YOLO Animal Detection web app.

Run from the project folder:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import app as webapp  # noqa: E402


class WebAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        webapp.app.config.update(TESTING=True)
        cls.client = webapp.app.test_client()
        cls.samples = sorted((PROJECT / "static" / "samples").iterdir())

    def _predict(self, mode, path, filename=None):
        data = path.read_bytes()
        return self.client.post(
            "/predict",
            data={"mode": mode, "image": (io.BytesIO(data), filename or path.name)},
            content_type="multipart/form-data",
        )

    def test_models_loaded_with_correct_tasks(self):
        self.assertEqual(sorted(webapp.models), ["box", "poly"])
        self.assertEqual(webapp.models["box"].task, "detect")
        self.assertEqual(webapp.models["poly"].task, "segment")
        for model in webapp.models.values():
            self.assertEqual(set(model.names.values()), {"cat", "cow", "dog", "lion"})

    def test_health(self):
        payload = self.client.get("/health").get_json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["models_loaded"], 2)
        self.assertEqual(payload["classes"], ["cat", "cow", "dog", "lion"])

    def test_dashboard(self):
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn("YOLO Animal Detection", html)
        self.assertIn("Supported classes", html)
        self.assertEqual(html.count('class="class-card"'), 4)
        self.assertIn("2 models ready", html)

    def test_rectangle_predictions(self):
        for sample in self.samples:
            with self.subTest(sample=sample.name):
                response = self._predict("box", sample)
                self.assertEqual(response.status_code, 200)
                payload = response.get_json()
                self.assertEqual(payload["mode"], "box")
                self.assertTrue(payload["image"].startswith("data:image/jpeg;base64,"))
                self.assertGreaterEqual(payload["total"], 1)
                self.assertIsInstance(payload["processing_time_ms"], int)

    def test_polygon_predictions(self):
        for sample in self.samples:
            with self.subTest(sample=sample.name):
                response = self._predict("poly", sample)
                self.assertEqual(response.status_code, 200)
                payload = response.get_json()
                self.assertEqual(payload["mode"], "poly")
                self.assertGreaterEqual(payload["total"], 1)
                self.assertTrue(all("has_mask" in item for item in payload["detections"]))

    def test_rejects_wrong_extension(self):
        response = self.client.post(
            "/predict",
            data={"mode": "box", "image": (io.BytesIO(b"hello"), "notes.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file type", response.get_json()["error"])

    def test_rejects_fake_image(self):
        response = self.client.post(
            "/predict",
            data={"mode": "box", "image": (io.BytesIO(b"not a png"), "fake.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("not a valid image", response.get_json()["error"])

    def test_rejects_missing_image(self):
        response = self.client.post(
            "/predict", data={"mode": "box"}, content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 400)

    def test_rejects_invalid_mode(self):
        sample = self.samples[0]
        response = self._predict("other", sample)
        self.assertEqual(response.status_code, 400)
        self.assertIn("valid prediction model", response.get_json()["error"])

    def test_rejects_file_over_10mb(self):
        oversized = b"\xff\xd8\xff" + b"0" * (10 * 1024 * 1024 + 10)
        response = self.client.post(
            "/predict",
            data={"mode": "box", "image": (io.BytesIO(oversized), "big.jpg")},
            content_type="multipart/form-data",
        )
        self.assertIn(response.status_code, (400, 413))
        self.assertIn("10 MB", response.get_json()["error"])

    def test_internal_error_does_not_leak_server_path(self):
        from unittest.mock import patch

        sample = self.samples[0]
        with patch.object(
            webapp.models["box"], "predict", side_effect=RuntimeError(r"C:\secret\weights\best.pt")
        ):
            response = self._predict("box", sample)
        self.assertEqual(response.status_code, 500)
        body = response.get_data(as_text=True)
        self.assertNotIn("secret", body)
        self.assertNotIn("best.pt", body)
        self.assertIn("Prediction failed", body)

    def test_predict_response_is_not_cached(self):
        response = self._predict("box", self.samples[0])
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")

    def test_404_page(self):
        response = self.client.get("/missing-page")
        self.assertEqual(response.status_code, 404)
        self.assertIn("Page not found", response.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
