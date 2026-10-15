
import json
import logging
import math
from pathlib import Path
from typing import Any

from src.dataset import load_jsonl, prepare_dataset
from src.logging_config import configure_logging


logger = logging.getLogger(__name__)


def validate_config(config: dict[str, Any]) -> None:
    """Validate essential DPO training configuration."""
    required_sections = (
        "model",
        "dataset",
        "output",
        "training",
        "lora",
        "quantization",
    )

    for section in required_sections:
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing or invalid config section: {section}")

    required_values = (
        ("model", "id"),
        ("dataset", "path"),
        ("output", "adapter_dir"),
        ("output", "metrics_path"),
    )

    for section, key in required_values:
        value = config[section].get(key)

        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Missing or invalid configuration: {section}.{key}"
            )

    training = config["training"]

    positive_fields = (
        "beta",
        "learning_rate",
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_length",
        "max_prompt_length",
        "max_completion_length",
    )

    for key in positive_fields:
        value = training.get(key)

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(
                f"training.{key} must be a positive number."
            )

    for key in ("max_length", "max_prompt_length", "max_completion_length",
                "per_device_train_batch_size", "gradient_accumulation_steps"):
        if not isinstance(training[key], int):
            raise ValueError(f"training.{key} must be an integer.")

    if not isinstance(training.get("fp16"), bool):
        raise ValueError("training.fp16 must be true or false.")

    lora = config["lora"]

    for key in ("r", "alpha"):
        value = lora.get(key)

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(f"lora.{key} must be a positive integer.")

    dropout = lora.get("dropout")

    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not math.isfinite(dropout)
        or not 0 <= dropout < 1
    ):
        raise ValueError("lora.dropout must be between 0 and 1.")

    target_modules = lora.get("target_modules")

    if (
        not isinstance(target_modules, list)
        or not target_modules
        or not all(isinstance(item, str) and item for item in target_modules)
    ):
        raise ValueError(
            "lora.target_modules must be a non-empty list of strings."
        )

    if not isinstance(
        config["quantization"].get("load_in_4bit"), bool
    ):
        raise ValueError(
            "quantization.load_in_4bit must be true or false."
        )


def validate_dpo_loss(loss: Any) -> float:
    """Reject invalid DPO training-loss values."""
    if isinstance(loss, bool) or not isinstance(loss, (int, float)):
        raise ValueError("DPO loss must be a numeric value.")

    value = float(loss)

    if not math.isfinite(value) or value < 0:
        raise ValueError(
            "DPO loss must be finite and non-negative."
        )

    return value


def freeze_reference_model(model: Any) -> Any:
    """Disable training for every reference-model parameter."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model


def is_reference_frozen(model: Any) -> bool:
    """Return True only when all reference-model parameters are frozen."""
    return all(
        not parameter.requires_grad
        for parameter in model.parameters()
    )


def run_dpo_training(
    config_path: str = "configs/config.yaml",
) -> dict[str, Any]:
    """Run quantized DPO fine-tuning and save the LoRA adapter."""
    import torch
    import yaml
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import DPOConfig, DPOTrainer

    logger.info("Loading configuration from %s", config_path)

    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")

    validate_config(config)
    logger.info("Configuration validated successfully.")

    # Prepare and validate data before attempting GPU training.
    logger.info("Preparing preference dataset.")

    records, dataset_stats = prepare_dataset(
        load_jsonl(config["dataset"]["path"])
    )

    logger.info(
        "Prepared %d examples. Dataset statistics: %s",
        len(records),
        dataset_stats,
    )

    if len(records) < 2:
        raise ValueError(
            "At least two valid preference examples are required."
        )

    dataset = Dataset.from_list(records)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "DPO training requires a suitable CUDA GPU. "
            "Dataset and configuration checks completed, but "
            "model training must run in a GPU environment."
        )

    model_id = config["model"]["id"]
    training = config["training"]
    lora = config["lora"]
    output_dir = config["output"]["adapter_dir"]

    logger.info("Loading tokenizer for %s", model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.pad_token is None:
        raise ValueError(
            "The tokenizer does not define a usable padding token."
        )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=config["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    logger.info("Loading policy model: %s", model_id)

    policy_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
    )

    logger.info("Loading reference model: %s", model_id)

    reference_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
    )

    freeze_reference_model(reference_model)

    if not is_reference_frozen(reference_model):
        raise RuntimeError("Reference model is not fully frozen.")

    logger.info("Reference model frozen and verified.")

    policy_model.config.use_cache = False

    lora_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        task_type=TaskType.CAUSAL_LM,
    )

    training_args = DPOConfig(
        output_dir=output_dir,
        beta=training["beta"],
        learning_rate=training["learning_rate"],
        num_train_epochs=training["num_train_epochs"],
        per_device_train_batch_size=(
            training["per_device_train_batch_size"]
        ),
        gradient_accumulation_steps=(
            training["gradient_accumulation_steps"]
        ),
        max_length=training["max_length"],
        max_prompt_length=training["max_prompt_length"],
        max_completion_length=training["max_completion_length"],
        fp16=training["fp16"],
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=policy_model,
        ref_model=reference_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    logger.info(
        "Starting DPO training: beta=%s, epochs=%s",
        training["beta"],
        training["num_train_epochs"],
    )

    result = trainer.train()
    metrics = dict(result.metrics)

    # Validate the reported loss before saving the final artifacts.
    if "train_loss" not in metrics:
        raise RuntimeError("Training did not report train_loss.")

    metrics["train_loss"] = validate_dpo_loss(
        metrics["train_loss"]
    )

    metrics["dataset_stats"] = dataset_stats
    metrics["log_history"] = trainer.state.log_history

    logger.info(
        "DPO training completed. Training loss: %.4f",
        metrics["train_loss"],
    )

    # Save the trained adapter and tokenizer.
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("Saved LoRA adapter to %s", output_dir)

    # Save metrics and training history as JSON.
    metrics_path = Path(config["output"]["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, default=str)

    logger.info("Saved training metrics to %s", metrics_path)

    return metrics


if __name__ == "__main__":
    configure_logging()
    run_dpo_training()
