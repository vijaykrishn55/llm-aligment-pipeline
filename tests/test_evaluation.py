import pytest

from src.evaluate import (
    evaluate_outputs,
    summarize_results,
    summarize_reward_margins,
)


def test_aligned_win_rate_and_category_rates():
    rows = [
        {
            "category": "helpfulness",
            "winner": "aligned",
            "base_pass": False,
            "aligned_pass": True,
        },
        {
            "category": "safety",
            "winner": "tie",
            "base_pass": False,
            "aligned_pass": True,
        },
        {
            "category": "helpfulness",
            "winner": "base",
            "base_pass": True,
            "aligned_pass": False,
        },
    ]

    summary = summarize_results(rows)

    assert summary["aligned_win_rate"] == pytest.approx(0.5)
    assert summary["aligned_wins"] == 1
    assert summary["base_wins"] == 1
    assert summary["ties"] == 1
    assert summary["categories"]["safety"]["aligned_pass_rate"] == 1.0


def test_evaluate_outputs_uses_judge_results():
    records = [
        {"prompt": "Explain APIs.", "category": "helpfulness"}
    ]

    def fake_judge(prompt, base, aligned, category):
        return {
            "winner": "aligned",
            "base_pass": False,
            "aligned_pass": True,
            "reason": "The aligned answer is clearer.",
        }

    report = evaluate_outputs(
        records,
        ["An API is something."],
        ["An API lets software systems communicate."],
        fake_judge,
    )

    assert report["summary"]["aligned_win_rate"] == 1.0
    assert report["results"][0]["reason"] == (
        "The aligned answer is clearer."
    )


def test_reward_margin_summary():
    report = summarize_reward_margins({
        "log_history": [
            {"rewards/margins": 0.2},
            {"rewards/margins": 0.4},
        ]
    })

    assert report["available"] is True
    assert report["observations"] == 2
    assert report["mean"] == pytest.approx(0.3)


def test_missing_reward_margins_are_reported():
    report = summarize_reward_margins({})

    assert report["available"] is False