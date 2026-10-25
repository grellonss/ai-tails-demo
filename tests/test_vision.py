import cv2
import numpy as np
import pytest

from conftest import FIXTURES
from vision import (LandmarksApiError, ThermalImageError, analyze_landmarks_detailed, analyze_thermal_image,
                    classify_temperature, detect_cat_face, generate_mock_landmarks, send_image_for_processing)
from vision import detector


# --- FGS from landmarks -----------------------------------------------------------
@pytest.mark.parametrize("profile, expected", [("healthy", 0), ("moderate", 1), ("pain", 2)])
def test_fgs_scores_from_landmarks(profile, expected):
    s = analyze_landmarks_detailed(generate_mock_landmarks(profile))["scores"]
    assert (s["ear"], s["eye"], s["muzzle"]) == (expected,) * 3
    assert s["total"] == 3 * expected


def test_fgs_is_scale_invariant():
    a = analyze_landmarks_detailed(generate_mock_landmarks("moderate"))
    b = analyze_landmarks_detailed(generate_mock_landmarks("moderate", offset=(400, 90), scale=2.7))
    assert a["scores"] == b["scores"]


# --- t4a API client ---------------------------------------------------------------
class FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


def test_landmarks_api_payload_and_choice_of_animal(monkeypatch):
    sent = {}
    far = generate_mock_landmarks("healthy", offset=(2000, 2000))
    near = generate_mock_landmarks("pain")

    def fake_post(url, json, timeout):
        sent.update(url=url, json=json)
        return FakeResponse([{"cat": {"landmarks": far}}, {"cat": {"landmarks": near}}])

    monkeypatch.setattr(detector.requests, "post", fake_post)
    pts = send_image_for_processing(b"\xff\xd8jpeg", "http://api/landmarks", name="oliver.jpg",
                                    face_box={"x1": 0, "y1": 0, "x2": 400, "y2": 400})
    assert sent["json"]["name"] == "oliver.jpg"
    assert sent["json"]["image"].startswith("data:image/jpeg;base64,")
    assert pts == near                                   # the animal inside the YOLO box


def test_landmarks_api_not_configured():
    with pytest.raises(LandmarksApiError):
        send_image_for_processing(b"x", "")


def test_landmarks_api_wrong_number_of_points(monkeypatch):
    monkeypatch.setattr(detector.requests, "post", lambda *a, **k: FakeResponse([{"cat": {"landmarks": [{"x": 1, "y": 1}]}}]))
    with pytest.raises(LandmarksApiError):
        send_image_for_processing(b"x", "http://api/landmarks")


# --- YOLO ---------------------------------------------------------------------------
def test_yolo_finds_the_cat_face():
    face = detect_cat_face(cv2.imread(str(FIXTURES / "cat_face.jpg")))
    assert face is not None and face["confidence"] > 0.5


def test_yolo_no_face_on_blank_image():
    assert detect_cat_face(np.full((480, 640, 3), 127, np.uint8)) is None


# --- thermal ------------------------------------------------------------------------
@pytest.mark.parametrize("temp, status, score", [
    (37.9, "Hypothermia", 1), (38.0, "Normothermia", 0), (39.2, "Normothermia", 0),
    (39.3, "MildFever", 1), (39.99, "MildFever", 1), (40.0, "HighFever", 2),
])
def test_classify_temperature(temp, status, score):
    r = classify_temperature(temp)
    assert (r["thermal_status"], r["thermal_score"]) == (status, score)


@pytest.mark.parametrize("fixture, status", [("thermal_fever.jpg", "HighFever"), ("thermal_cold.jpg", "Hypothermia")])
def test_thermal_photo_is_read(fixture, status):
    r = analyze_thermal_image(cv2.imread(str(FIXTURES / fixture)))
    assert r["thermal_status"] == status
    assert r["scale"]["max"] > r["scale"]["min"]
    assert len(r["hotspots"]) == 3
    assert r["temperature"] == pytest.approx(r["surface_max_c"] + r["body_offset_c"], abs=0.11)


def test_thermal_rejects_a_normal_photo():
    with pytest.raises(ThermalImageError):
        analyze_thermal_image(cv2.imread(str(FIXTURES / "cat_face.jpg")))
