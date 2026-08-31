from fastapi.testclient import TestClient

from autonoma import __version__
from autonoma.serving.app import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == __version__
    assert data["service"] == "autonoma"


def test_health_ready_returns_ready():
    assert client.get("/health/ready").json()["status"] == "ready"


def test_predict_stub_returns_valid_response():
    response = client.post(
        "/predict",
        json={"features": {"nswprice": 0.056, "nswdemand": 0.439}, "record_index": 12},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in (0, 1)
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["model_version"]


def test_metrics_label_on_route_template_not_raw_path():
    client.get("/health")
    body = client.get("/metrics").text
    assert 'endpoint="/health"' in body
