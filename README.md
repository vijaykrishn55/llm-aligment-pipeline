
# LLM Alignment & Preference Tuning Pipeline

An LLM alignment project implementing Constitutional AI, Direct Preference Optimization (DPO), LoRA fine-tuning, runtime guardrails, and base-versus-aligned model evaluation.

## Features

- Preference dataset validation and cleaning
- Constitutional AI critique and revision for synthetic preference generation
- 4-bit quantization and LoRA-based DPO fine-tuning
- Input and output safety guardrails
- LLM-as-a-Judge evaluation with helpfulness and safety checks
- Training metrics, evaluation reports, logging, and automated tests

## Architecture

```text
Preference Data → Constitutional AI → DPO + LoRA
                                      ↓
                                Aligned Adapter
                                      ↓
                         Guardrails & Evaluation
                                      ↓
                                   Reports
```

## Tech Stack

Python, PyTorch, Hugging Face Transformers, TRL, PEFT, BitsAndBytes, Datasets, PyYAML, and Pytest.

**Model:** `Qwen/Qwen2.5-7B-Instruct`

## Project Structure

```text
configs/       Configuration and training settings
data/          Source, synthetic, and evaluation datasets
src/           Dataset, alignment, guardrails, evaluation
               and Constitutional AI modules
tests/         Automated tests
outputs/       Trained LoRA adapter
reports/       Training metrics and evaluation results
```

## Run the Pipeline

Install the dependencies:

```bash
pip install -r requirements.txt
```

Generate synthetic preference data:

```bash
python -m src.constitutional_ai
```

Run DPO fine-tuning:

```bash
python -m src.alignment
```

Evaluate the trained adapter:

```bash
python -m src.evaluate
```

Run the test suite:

```bash
python -m pytest tests/ -v
```

DPO training and model evaluation require a compatible GPU environment. The current implementation uses a local Hugging Face model for generation and judging.

## Recorded Results

- **DPO training:** Completed a five-epoch run using four preference pairs.
- **Training loss:** `0.6835`
- **Final DPO reward margin:** `0.0370`
- **Automated tests:** 24 passed.
- **Evaluation:** Six prompts evaluated; zero aligned wins, zero base wins, and six ties.
- **Tie-adjusted aligned score:** 50%.
- **Helpfulness and safety:** Both models passed all six evaluation checks after identical responses were handled consistently.

A later Constitutional AI generation run produced 16 traces and 11 retained preference pairs. The expanded dataset was not used for a subsequent training and evaluation run before the Kaggle session restarted.

The recorded evaluation did **not demonstrate measurable alignment improvement**. The 50% score reflects ties, not wins by the aligned model. The evaluation set is small and is not statistically conclusive.

## Limitations

- The preference datasets are small and synthetically generated.
- Generated preference pairs require quality review.
- Pattern-based guardrails are not comprehensive security controls.
- LLM-as-a-Judge can be inconsistent or biased.
- Reward margins and successful training do not, by themselves, establish alignment improvement.
