import gc
import json
import logging
from pathlib import Path
from typing import Any, Callable

from src.dataset import load_jsonl, prepare_dataset, save_jsonl
from src.logging_config import configure_logging


logger = logging.getLogger(__name__)

GenerateText = Callable[[str], str]

CONSTITUTION = (
    "Be helpful and address the user's actual request.",
    "Be honest and do not invent facts.",
    "Do not facilitate harmful or illegal actions.",
    "Protect sensitive personal information.",
    "Explain important limitations and uncertainty.",
)


def create_local_generator(
    model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    max_new_tokens: int = 512,
) -> GenerateText:
    """Load a quantized Hugging Face model for local generation."""
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Local Qwen generation requires a CUDA GPU. "
            "Run this pipeline on Kaggle."
        )

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive.")

    logger.info("Loading local generation model: %s", model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def generate(instruction: str) -> str:
        messages = [{"role": "user", "content": instruction}]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_length = inputs["input_ids"].shape[1]
        generated_tokens = output[0, prompt_length:]

        return tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

    return generate


def create_preference_pair(
    prompt: str,
    generate_text: GenerateText,
) -> tuple[dict[str, str], dict[str, str]]:
    """Create a draft, critique, revision and DPO preference pair."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string.")

    prompt = prompt.strip()
    principles = "\n".join(
        f"- {principle}" for principle in CONSTITUTION
    )

    draft = generate_text(
        f"""Answer the user's request clearly and directly.

User request:
{prompt}

Return only the answer."""
    )

    if not isinstance(draft, str) or not draft.strip():
        raise ValueError("The generator returned an empty draft.")

    draft = draft.strip()

    critique = generate_text(
        f"""Review the draft against these principles:
{principles}

User request:
{prompt}

Draft answer:
{draft}

Identify factual, safety, privacy or helpfulness issues.
If no meaningful issue exists, say so briefly."""
    )

    if not isinstance(critique, str) or not critique.strip():
        raise ValueError("The generator returned an empty critique.")

    critique = critique.strip()

    revised = generate_text(
        f"""Produce the best final answer to the user.

Principles:
{principles}

User request:
{prompt}

Original draft:
{draft}

Critique:
{critique}

Address valid issues. Return only the revised answer."""
    )

    if not isinstance(revised, str) or not revised.strip():
        raise ValueError("The generator returned an empty revision.")

    revised = revised.strip()

    pair = {
        "prompt": prompt,
        "chosen": revised,
        "rejected": draft,
    }

    trace = {
        "prompt": prompt,
        "draft": draft,
        "critique": critique,
        "revised": revised,
    }

    return pair, trace


def run_constitutional_ai(
    config_path: str = "configs/config.yaml",
) -> dict[str, Any]:
    """Generate synthetic preferences and save the workflow traces."""
    import torch
    import yaml

    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")

    settings = config["constitutional_ai"]
    max_prompts = settings["max_prompts"]

    if (
        isinstance(max_prompts, bool)
        or not isinstance(max_prompts, int)
        or max_prompts < 1
    ):
        raise ValueError("max_prompts must be a positive integer.")

    source_records = load_jsonl(settings["input_path"])

    prompts = [
        record["prompt"]
        for record in source_records
        if isinstance(record, dict)
        and isinstance(record.get("prompt"), str)
        and record["prompt"].strip()
    ][:max_prompts]

    if not prompts:
        raise ValueError("No valid prompts found in the input dataset.")

    model_id = config["model"]["id"]
    generate_text = create_local_generator(model_id)

    pairs = []
    traces = []

    try:
        for index, prompt in enumerate(prompts, start=1):
            logger.info(
                "Generating Constitutional AI example %d/%d",
                index,
                len(prompts),
            )

            pair, trace = create_preference_pair(
                prompt,
                generate_text,
            )

            if pair["chosen"].casefold() == pair["rejected"].casefold():
                trace["pair_usable"] = False
                trace["skip_reason"] = "Draft and revision were identical."
            else:
                trace["pair_usable"] = True
                pairs.append(pair)

            traces.append(trace)

        clean_pairs, stats = prepare_dataset(pairs)

        output_path = Path(settings["output_path"])
        save_jsonl(clean_pairs, output_path)

        trace_path = Path(settings["trace_path"])
        trace_path.parent.mkdir(parents=True, exist_ok=True)

        with trace_path.open("w", encoding="utf-8") as file:
            for trace in traces:
                file.write(
                    json.dumps(trace, ensure_ascii=False) + "\n"
                )

        logger.info("Saved preference dataset to %s", output_path)
        logger.info("Saved Constitutional AI traces to %s", trace_path)
        logger.info("Synthetic dataset statistics: %s", stats)

        if len(clean_pairs) < 2:
            logger.warning(
                "Only %d usable examples remain. DPO training requires "
                "at least two valid preference records.",
                len(clean_pairs),
            )

        return stats

    finally:
        del generate_text
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    configure_logging()
    run_constitutional_ai()