import logging
import re
from dataclasses import dataclass
from typing import Callable, Any

logger = logging.getLogger(__name__)

SAFE_RESPONSE = (
    "I can't help with that request, but I can help with a safe alternative."
)

ERROR_RESPONSE = (
    "I couldn't generate a response right now. Please try again later."
)

DEFAULT_SETTINGS = {
    "block_prompt_injection": True,
    "block_unsafe_requests": True,
    "validate_outputs": True,
}

INJECTION_PATTERNS = [
    re.compile(
        r"\bignore\s+(?:all\s+)?(?:previous|prior|above|earlier)"
        r"\s+instructions\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:reveal|print|show|dump|leak)\s+(?:the\s+)?"
        r"(?:system prompt|developer message|hidden instructions)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:disable|bypass|ignore)\s+(?:the\s+)?"
        r"(?:safety rules|guardrails|content filters)\b",
        re.IGNORECASE,
    ),
]

UNSAFE_REQUEST_PATTERNS = [
    re.compile(
        r"\b(?:create|write|build|deploy|install)\s+(?:a\s+)?"
        r"(?:keylogger|ransomware|credential stealer|phishing kit)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bsteal\s+(?:user\s+)?(?:passwords|credentials|session tokens)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:make|build)\s+(?:a\s+)?(?:bomb|explosive device)\b",
        re.IGNORECASE,
    ),
]

UNSAFE_OUTPUT_PATTERNS = UNSAFE_REQUEST_PATTERNS

SENSITIVE_OUTPUT_PATTERNS = [
    re.compile(
        r"\b(?:email|phone|mobile|aadhaar|passport|password|api[_ -]?key)"
        r"\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:gsk_[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9_-]{20,})\b"),
]

PROMPT_LEAK_PATTERNS = [
    re.compile(
        r"\b(?:system prompt|hidden instructions|developer message)\s*[:=]",
        re.IGNORECASE,
    ),
]


@dataclass(frozen=True)
class GuardrailResult:
    allowed: bool
    text: str
    reason: str | None = None


def _settings(settings: dict[str, Any] | None) -> dict[str, Any]:
    merged = DEFAULT_SETTINGS.copy()
    if settings:
        merged.update(settings)
    return merged


def check_input(
    text: str,
    settings: dict[str, Any] | None = None,
) -> GuardrailResult:
    """Screen a user request before it reaches the model."""
    options = _settings(settings)

    if not isinstance(text, str) or not text.strip():
        return GuardrailResult(False, SAFE_RESPONSE, "invalid_input")

    if options["block_prompt_injection"]:
        if any(pattern.search(text) for pattern in INJECTION_PATTERNS):
            return GuardrailResult(
                False, SAFE_RESPONSE, "prompt_injection"
            )

    if options["block_unsafe_requests"]:
        if any(
            pattern.search(text)
            for pattern in UNSAFE_REQUEST_PATTERNS
        ):
            return GuardrailResult(
                False, SAFE_RESPONSE, "unsafe_request"
            )

    return GuardrailResult(True, text.strip())


def validate_output(
    text: str,
    settings: dict[str, Any] | None = None,
) -> GuardrailResult:
    """Validate a model response before returning it to the user."""
    options = _settings(settings)

    if not isinstance(text, str) or not text.strip():
        return GuardrailResult(False, SAFE_RESPONSE, "empty_output")

    output = text.strip()

    if options["validate_outputs"]:
        if any(pattern.search(output) for pattern in PROMPT_LEAK_PATTERNS):
            return GuardrailResult(False, SAFE_RESPONSE, "prompt_leakage")

        if any(
            pattern.search(output)
            for pattern in SENSITIVE_OUTPUT_PATTERNS
        ):
            return GuardrailResult(False, SAFE_RESPONSE, "sensitive_data")

        if any(
            pattern.search(output)
            for pattern in UNSAFE_OUTPUT_PATTERNS
        ):
            return GuardrailResult(False, SAFE_RESPONSE, "unsafe_output")

    return GuardrailResult(True, output)


def load_guardrail_settings(
    config_path: str = "configs/config.yaml",
) -> dict[str, Any]:
    """Load guardrail switches from the project's YAML configuration."""
    import yaml

    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    return _settings(config.get("guardrails"))


def run_guarded_generation(
    prompt: str,
    generate_text: Callable[[str], str],
    settings: dict[str, Any] | None = None,
) -> GuardrailResult:
    """Screen input, call the model, then screen its output."""
    if settings is None:
        settings = load_guardrail_settings()
    
    input_result = check_input(prompt, settings)

    if not input_result.allowed:
        logger.warning(
            "Request blocked by guardrails: %s", input_result.reason
        )
        return input_result

    try:
        output = generate_text(input_result.text)
    except Exception:
        logger.exception("Model generation failed")
        return GuardrailResult(False, ERROR_RESPONSE, "generation_error")

    result = validate_output(output, settings)

    if not result.allowed:
        logger.warning(
            "Model output blocked by guardrails: %s", result.reason
        )

    return result