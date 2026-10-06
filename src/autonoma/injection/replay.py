from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import httpx


def _parse_value(value: str) -> float | int | str | None:
    """Convert CSV values into types accepted by PredictRequest."""
    value = value.strip()

    if value == "":
        return None

    try:
        if "." not in value and "e" not in value.lower():
            return int(value)
        return float(value)
    except ValueError:
        return value


def _count_rows(path: Path) -> int:
    """Count data rows in a CSV without loading the dataset into memory."""
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return sum(1 for _ in reader)


def replay_csv(
    input_path: str | Path,
    endpoint_url: str = "http://localhost:8000/predict",
    start_fraction: float = 0.30,
    target_column: str | None = None,
    output_path: str | Path | None = None,
    timeout: float = 30.0,
) -> int:
    """
    Replay the final portion of a CSV dataset through the /predict endpoint.

    By default, the first 30% is skipped and the remaining 70% is
    replayed in the original temporal order.
    """
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if not 0.0 <= start_fraction < 1.0:
        raise ValueError("start_fraction must be between 0.0 and 1.0")

    total_rows = _count_rows(path)
    start_index = int(total_rows * start_fraction)

    output_file = None
    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_handle = output_file.open(
            "w",
            encoding="utf-8",
        )
    else:
        output_handle = None

    replayed = 0

    try:
        with (
            httpx.Client(timeout=timeout) as client,
            path.open("r", newline="", encoding="utf-8") as file,
        ):
                reader = csv.DictReader(file)

                if reader.fieldnames is None:
                    raise ValueError("CSV file must contain a header row")

                for record_index, row in enumerate(reader):
                    if record_index < start_index:
                        continue

                    features: dict[str, float | int | str | None] = {}

                    for column, value in row.items():
                        if column is None:
                            raise ValueError(
                                "CSV contains an unexpected extra column"
                            )

                        if column == target_column:
                            continue

                        features[column] = _parse_value(value or "")

                    payload = {
                        "features": features,
                        "record_index": record_index,
                    }

                    response = client.post(
                        endpoint_url,
                        json=payload,
                    )
                    response.raise_for_status()

                    result = response.json()

                    if output_handle is not None:
                        log_entry = {
                            "record_index": record_index,
                            **result,
                        }
                        output_handle.write(
                            json.dumps(log_entry) + "\n"
                        )

                    replayed += 1

    finally:
        if output_handle is not None:
            output_handle.close()

    return replayed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a CSV dataset through the AUTONOMA prediction API."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the CSV dataset.",
    )

    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000/predict",
        help="Prediction endpoint URL.",
    )

    parser.add_argument(
        "--start-fraction",
        type=float,
        default=0.30,
        help="Fraction of the dataset to skip before replay. Default: 0.30",
    )

    parser.add_argument(
        "--target-column",
        default=None,
        help="Optional target/label column to exclude from prediction features.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSONL file for storing prediction responses.",
    )

    args = parser.parse_args()

    count = replay_csv(
        input_path=args.input,
        endpoint_url=args.endpoint,
        start_fraction=args.start_fraction,
        target_column=args.target_column,
        output_path=args.output,
    )

    print(f"Replayed {count} records.")


if __name__ == "__main__":
    main()
