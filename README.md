# LLM Alignment & Preference Tuning Pipeline

An end-to-end LLM alignment project covering preference dataset engineering, Direct Preference Optimization (DPO), Constitutional AI (RLAIF), runtime guardrails, and alignment evaluation.

## Objectives

- Build and validate preference datasets using prompt, chosen, and rejected responses.
- Fine-tune an instruction model using DPO and LoRA.
- Generate synthetic preference data through constitutional critique and revision.
- Implement input and output safety checks.
- Compare base and aligned model behavior.
- Test critical components and generate evaluation reports.

## Architecture

```text
Preference Dataset
       |
       v
Validation and Cleaning
       |
       v
Constitutional AI
       |
       v
DPO Fine-Tuning
       |
       v
LoRA Adapter
       |
       v
Runtime Guardrails
       |
       v
Alignment Evaluation
       |
       v
Reports and Test Results
```

## Project Structure

```text
llm-alignment-pipeline/
├── configs/
│   └── config.yaml
├── data/
│   ├── preferences.jsonl
│   ├── synthetic_preferences.jsonl
│   └── evaluation.jsonl
├── reports/
├── src/
│   ├── dataset.py
│   ├── alignment.py
│   ├── constitutional_ai.py
│   ├── guardrails.py
│   ├── evaluate.py
│   └── logging_config.py
├── tests/
│   ├── test_dataset.py
│   ├── test_alignment.py
│   ├── test_constitutional_ai.py
│   ├── test_guardrails.py
│   └── test_evaluation.py
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## Technology Stack

- Python
- PyTorch and Hugging Face Transformers
- TRL for DPO training
- PEFT and LoRA for parameter-efficient fine-tuning
- BitsAndBytes for 4-bit quantization
- Hugging Face Datasets
- Groq API for synthetic preference generation and LLM-as-a-Judge evaluation
- PyYAML for configuration
- Pytest for automated tests

## Setup

Use Python 3.10 or newer in a compatible environment.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Some training dependencies require a compatible Linux/CUDA environment. For local unit testing, install `pytest`, `PyYAML`, and the other packages imported by the tests. The actual DPO training environment will be configured on Kaggle.

## Configuration

Project settings are maintained in `configs/config.yaml`.

The configuration includes the base model, dataset paths, DPO hyperparameters, LoRA settings, quantization, Constitutional AI settings, guardrails, and evaluation outputs.

The default model is `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.

## 1. Preference Dataset Engineering

The dataset module validates the required fields:

- `prompt`: the user instruction.
- `chosen`: the preferred response.
- `rejected`: the less-preferred response.

It also removes duplicate records, filters extreme response-length imbalances, and supports JSONL loading and saving.

Run the dataset tests:

```powershell
python -m pytest tests/test_dataset.py -v
```

## 2. Direct Preference Optimization

The alignment module prepares the preference dataset, configures the policy and frozen reference models, applies LoRA configuration, and invokes TRL's `DPOTrainer`.

The pipeline is configured for 4-bit quantization and saves the trained adapter and metrics.

Training requires a suitable CUDA GPU. The actual training run and package compatibility verification must be completed in the GPU environment.

## 3. Constitutional AI

The Constitutional AI module:

1. Generates an initial response.
2. Critiques the response against defined principles.
3. Generates a revised answer.
4. Creates a chosen/rejected preference pair.
5. Saves synthetic preference data and generation traces.

Configure `GROQ_API_KEY` and `GROQ_MODEL` as environment variables before running:

```powershell
python -m src.constitutional_ai
```

The generated preference pairs should be inspected before being used for fine-tuning.

## 4. Runtime Guardrails

The guardrail module provides:

- Input screening for selected prompt-injection patterns.
- Detection of selected malicious-request patterns.
- Output checks for selected unsafe content and sensitive-data patterns.
- Safe fallback responses when checks reject a request or generation fails.

The baseline checks are intentionally lightweight. They are not comprehensive protection against prompt injection, sensitive-data disclosure, or all harmful content.

Run the guardrail tests:

```powershell
python -m pytest tests/test_guardrails.py -v
```

## 5. Alignment Evaluation

The evaluation module compares base-model and adapter outputs using a configured LLM judge.

The report includes:

- Base-versus-aligned wins, losses, and ties.
- Aligned win rate.
- Helpfulness and safety pass rates.
- Available DPO reward-margin observations.
- Individual prompts, responses, and judge explanations.

Run evaluation after training and saving the adapter:

```powershell
python -m src.evaluate
```

Evaluation requires a suitable GPU and configured Groq credentials. Reports are written to `reports/`.

The initial evaluation dataset is a small functional test set, not a statistically conclusive benchmark. LLM-as-a-Judge can also be inconsistent or biased.

## 6. Automated Testing

Run all unit tests:

```powershell
python -m pytest tests/ -v
```

The tests cover dataset validation, DPO configuration and loss validation, reference-model freezing, Constitutional AI generation, guardrails, and evaluation metrics.

Unit tests do not establish that fine-tuning improves model alignment. That requires actual training and evaluation results.

## 7. Logging and Artifacts

The logging module supports console logs and rotating log files.

Expected artifacts include:

- `outputs/dpo-adapter/`: trained LoRA adapter.
- `reports/dpo_train_metrics.json`: DPO training metrics and available training history.
- `reports/constitutional_traces.jsonl`: generated drafts, critiques, and revisions.
- `reports/alignment_evaluation.json`: detailed evaluation results.
- `reports/alignment_evaluation.md`: evaluation summary.
- `logs/pipeline.log`: application logs.

These artifacts are produced by their respective workflows; their presence depends on successfully completing the relevant execution stages.

## Limitations

- The initial preference dataset is too small for meaningful fine-tuning.
- Synthetic preference labels may be incorrect and require review.
- Pattern-based guardrails can miss attacks and produce false positives.
- Evaluation results depend on the selected examples and judge model.
- Reward margins alone do not demonstrate real-world safety or alignment.
- Training and model evaluation have not been considered complete until the GPU runs and their outputs are verified.

## Future Improvements

- Expand and manually review the preference dataset.
- Add stronger prompt-injection and output-safety evaluation.
- Use larger, diverse evaluation datasets.
- Compare results across multiple seeds and checkpoints.
- Improve judge calibration and reproducibility.
- Add deployment monitoring and model-version tracking.
