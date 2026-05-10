from fastapi.testclient import TestClient

from autonoma.api.main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"
    assert data["service"] == "autonoma"


def test_health_ready_returns_ready():
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_predict_stub_returns_valid_response():
    response = client.post("/predict", json={"features": {"close": 65000.0, "volume": 1234.5}})
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ("up", "down")
    assert 0.0 <= data["confidence"] <= 1.0
    assert "model_version" in data
