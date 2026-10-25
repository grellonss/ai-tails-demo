import io

import cv2
import numpy as np
import pytest

import app as app_module
from conftest import FIXTURES
from vision import generate_mock_landmarks


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def _file(name):
    return (io.BytesIO((FIXTURES / name).read_bytes()), name)


def _blank_jpg():
    ok, buf = cv2.imencode(".jpg", np.full((480, 640, 3), 127, np.uint8))
    return (io.BytesIO(buf.tobytes()), "blank.jpg")


def test_health(client):
    h = client.get("/api/health").get_json()
    assert h["status"] == "online" and "fgs_fever_acute" in h["rules"]


def test_both_photos_are_required(client):
    r = client.post("/api/analyze", data={"image": _file("cat_face.jpg")})
    assert r.status_code == 400 and r.get_json()["step"] == "input"


def test_no_face(client):
    r = client.post("/api/analyze", data={"image": _blank_jpg(), "thermal_image": _file("thermal_fever.jpg")})
    assert r.status_code == 422 and r.get_json()["step"] == "face"


def test_landmarks_api_missing(client, monkeypatch):
    monkeypatch.setattr(app_module.config, "LANDMARKS_API_URL", "")
    r = client.post("/api/analyze", data={"image": _file("cat_face.jpg"), "thermal_image": _file("thermal_fever.jpg")})
    assert r.status_code == 502 and r.get_json()["step"] == "landmarks"


def test_full_pipeline_pain_and_fever(client, monkeypatch):
    """Simulated API landmarks (pain profile) + thermal photo with fever -> multimodal rule."""
    monkeypatch.setattr(app_module, "send_image_for_processing", lambda *a, **k: generate_mock_landmarks("pain"))
    r = client.post("/api/analyze", data={"cat_name": "Luna", "image": _file("cat_face.jpg"),
                                          "thermal_image": _file("thermal_fever.jpg")})
    assert r.status_code == 200, r.get_json()
    d = r.get_json()
    assert d["scores"]["total"] == 6
    assert d["thermal"]["thermal_status"] == "HighFever"
    assert d["ontology"]["primary_status"] == "AcuteInflammatoryPain"
    assert "fgs_fever_acute" in [x["name"] for x in d["ontology"]["fired_rules"]]
    assert d["verdict"]["is_sick"] is True
    assert d["report"]["engine"] == "summary" and "AcuteInflammatoryPain" in d["report"]["text"]


def test_full_pipeline_healthy_face(client, monkeypatch):
    monkeypatch.setattr(app_module, "send_image_for_processing", lambda *a, **k: generate_mock_landmarks("healthy"))
    d = client.post("/api/analyze", data={"image": _file("cat_face.jpg"),
                                          "thermal_image": _file("thermal_cold.jpg")}).get_json()
    assert d["ontology"]["primary_status"] == "Good" and d["verdict"]["is_sick"] is False


def test_chat_without_azure(client):
    r = client.post("/api/chat", json={"question": "What should I do?", "context": {}})
    assert r.status_code == 200 and r.get_json()["engine"] == "none"
