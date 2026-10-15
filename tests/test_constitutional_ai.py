import pytest

from src.constitutional_ai import create_preference_pair


def test_creates_preference_pair_and_trace():
    responses = iter([
        "An API is a way for software to communicate.",
        "The draft is clear, with no major safety issues.",
        "An API is a defined interface that lets software systems communicate.",
    ])

    pair, trace = create_preference_pair(
        "What is an API?",
        lambda instruction: next(responses),
    )

    assert pair["prompt"] == "What is an API?"
    assert pair["chosen"].startswith("An API is a defined")
    assert pair["rejected"].startswith("An API is a way")

    assert trace["draft"] == pair["rejected"]
    assert trace["critique"]
    assert trace["revised"] == pair["chosen"]


def test_empty_prompt_is_rejected():
    with pytest.raises(ValueError):
        create_preference_pair(" ", lambda _: "answer")


def test_empty_model_response_is_rejected():
    with pytest.raises(ValueError):
        create_preference_pair(
            "What is an API?",
            lambda _: " ",
        )