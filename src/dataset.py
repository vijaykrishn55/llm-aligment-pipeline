import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = ("prompt", "chosen", "rejected")


def validate_record(record: Any) -> dict[str, str]:
    """Validate and normalize one preference example."""
    if not isinstance(record, Mapping):
        raise ValueError("Each record must be a dictionary.")

    cleaned = {}

    for field in REQUIRED_FIELDS:
        value = record.get(field)

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"'{field}' must be a non-empty string.")

        cleaned[field] = value.strip()

    return cleaned


def prepare_dataset(
    records: Iterable[Any],
    max_length_ratio: float = 3.0,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Validate records, remove duplicates and filter extreme length gaps."""
    if max_length_ratio < 1:
        raise ValueError("max_length_ratio must be at least 1.")

    cleaned_records = []
    seen = set()

    stats = {
        "input": 0,
        "invalid": 0,
        "duplicates": 0,
        "length_filtered": 0,
        "kept": 0,
    }

    for record in records:
        stats["input"] += 1

        try:
            cleaned = validate_record(record)
        except ValueError:
            stats["invalid"] += 1
            continue

        key = tuple(cleaned[field] for field in REQUIRED_FIELDS)

        if key in seen:
            stats["duplicates"] += 1
            continue

        seen.add(key)

        chosen_length = len(cleaned["chosen"].split())
        rejected_length = len(cleaned["rejected"].split())

        shortest = min(chosen_length, rejected_length)
        longest = max(chosen_length, rejected_length)

        if longest / shortest > max_length_ratio:
            stats["length_filtered"] += 1
            continue

        cleaned_records.append(cleaned)

    stats["kept"] = len(cleaned_records)
    return cleaned_records, stats


def load_jsonl(path: str | Path) -> list[Any]:
    """Load one JSON record per line."""
    records = []

    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc.msg}"
                ) from exc

    return records


def save_jsonl(
    records: Iterable[Any],
    path: str | Path,
) -> None:
    """Validate and save records in JSONL format."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            cleaned = validate_record(record)
            file.write(
                json.dumps(cleaned, ensure_ascii=False) + "\n"
            )