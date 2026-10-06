import csv

from autonoma.injection.replay import replay_csv


class FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class FakeClient:
    calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def post(self, url, json):
        self.calls.append((url, json))

        return FakeResponse(
            {
                "prediction": 1,
                "confidence": 0.5,
                "model_version": "test-v0",
            }
        )


def test_replay_skips_first_30_percent_and_preserves_order(
    tmp_path,
    monkeypatch,
):
    dataset = tmp_path / "sample.csv"

    with dataset.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["feature_a", "feature_b", "target"],
        )
        writer.writeheader()

        for index in range(10):
            writer.writerow(
                {
                    "feature_a": index,
                    "feature_b": index + 10,
                    "target": index % 2,
                }
            )

    FakeClient.calls = []

    monkeypatch.setattr(
        "autonoma.injection.replay.httpx.Client",
        lambda timeout: FakeClient(),
    )

    replayed = replay_csv(
        input_path=dataset,
        endpoint_url="http://test/predict",
        start_fraction=0.30,
        target_column="target",
    )

    assert replayed == 7
    assert len(FakeClient.calls) == 7

    record_indices = [call[1]["record_index"] for call in FakeClient.calls]

    assert record_indices == [3, 4, 5, 6, 7, 8, 9]

    first_payload = FakeClient.calls[0][1]

    assert first_payload["features"] == {
        "feature_a": 3,
        "feature_b": 13,
    }

    assert "target" not in first_payload["features"]


def test_replay_writes_prediction_log(tmp_path, monkeypatch):
    dataset = tmp_path / "sample.csv"
    output = tmp_path / "predictions.jsonl"

    with dataset.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["feature"],
        )
        writer.writeheader()

        for index in range(4):
            writer.writerow({"feature": index})

    FakeClient.calls = []

    monkeypatch.setattr(
        "autonoma.injection.replay.httpx.Client",
        lambda timeout: FakeClient(),
    )

    replayed = replay_csv(
        input_path=dataset,
        endpoint_url="http://test/predict",
        start_fraction=0.50,
        output_path=output,
    )

    assert replayed == 2

    lines = output.read_text(encoding="utf-8").strip().splitlines()

    assert len(lines) == 2
    assert '"record_index": 2' in lines[0]
    assert '"record_index": 3' in lines[1]
