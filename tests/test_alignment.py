import pytest

from src.alignment import (
    freeze_reference_model,
    is_reference_frozen,
    validate_config,
    validate_dpo_loss,
)


class FakeParameter:
    def __init__(self):
        self.requires_grad = True


class FakeModel:
    def __init__(self):
        self.params = [FakeParameter(), FakeParameter()]
        self.training = True

    def parameters(self):
        return iter(self.params)

    def eval(self):
        self.training = False
        return self


@pytest.fixture
def valid_config():
    return {
        "model": {"id": "example-model"},
        "dataset": {"path": "data/preferences.jsonl"},
        "output": {
            "adapter_dir": "outputs/adapter",
            "metrics_path": "reports/metrics.json",
        },
        "training": {
            "beta": 0.1,
            "learning_rate": 1e-5,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "max_length": 512,
            "max_prompt_length": 256,
            "max_completion_length": 256,
            "fp16": True,
        },
        "lora": {
            "r": 8,
            "alpha": 16,
            "dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
        },
        "quantization": {
            "load_in_4bit": True,
        },
    }


def test_valid_config_passes(valid_config):
    validate_config(valid_config)


def test_invalid_beta_is_rejected(valid_config):
    valid_config["training"]["beta"] = 0

    with pytest.raises(ValueError):
        validate_config(valid_config)


def test_reference_model_is_frozen():
    model = FakeModel()

    freeze_reference_model(model)

    assert is_reference_frozen(model)
    assert model.training is False


def test_unfrozen_model_fails_verification():
    model = FakeModel()

    assert is_reference_frozen(model) is False


def test_dpo_loss_validation():
    # A valid numeric loss is accepted.
    assert validate_dpo_loss(0.42) == pytest.approx(0.42)

    # Invalid losses must be rejected.
    invalid_losses = [
        float("nan"),
        float("inf"),
        -0.1,
        "invalid",
        True,
    ]

    for loss in invalid_losses:
        with pytest.raises(ValueError):
            validate_dpo_loss(loss)