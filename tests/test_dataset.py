import pytest

from src.dataset import (
    load_jsonl,
    prepare_dataset,
    save_jsonl,
    validate_record,
)


VALID_RECORD = {
    "prompt": "What is caching?",
    "chosen": "Caching stores frequently used data temporarily for faster future access.",
    "rejected": "Caching keeps data nearby so later requests can be faster.",
}


def test_validate_record():
    assert validate_record(VALID_RECORD) == VALID_RECORD


def test_invalid_record_is_rejected():
    with pytest.raises(ValueError):
        validate_record({"prompt": "Hello", "chosen": "Hi"})


def test_deduplication_and_length_filtering():
    long_record = {
        "prompt": "Explain something",
        "chosen": (
            "This answer has many more words than the rejected answer "
            "and is clearly much longer than the other response."
        ),
        "rejected": "No.",
    }

    records, stats = prepare_dataset(
        [VALID_RECORD, VALID_RECORD, long_record]
    )

    assert records == [VALID_RECORD]
    assert stats["duplicates"] == 1
    assert stats["length_filtered"] == 1
    assert stats["kept"] == 1


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "preferences.jsonl"

    save_jsonl([VALID_RECORD], path)

    assert load_jsonl(path) == [VALID_RECORD]