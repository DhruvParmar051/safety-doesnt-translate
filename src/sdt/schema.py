"""Record types passed between phases. These are the contracts in IMPLEMENTATION_PLAN.md."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Kind(StrEnum):
    """What a prompt is for."""

    HARMFUL = "harmful"  # should be refused: Aya Red-teaming, MultiJail, XSTest unsafe
    XSTEST_SAFE = "xstest_safe"  # sounds dangerous, should be answered: over-refusal
    HARMLESS = "harmless"  # plain instruction: calibration for refusal directions only


class Split(StrEnum):
    CALIB = "calib"  # builds repairs: directions, saliency, components, steering
    TEST = "test"  # every reported number


class Label(StrEnum):
    """Judge verdict on one reply."""

    REFUSAL = "refusal"  # declines, fully or partly
    HARMFUL = "harmful"  # complies and the content could cause harm
    SAFE = "safe"  # answers without harmful content
    INVALID = "invalid"  # judge output could not be parsed


@dataclass
class PromptRecord:
    id: str
    lang: str
    prompt: str
    kind: Kind
    source: str
    split: Split = Split.TEST
    category: str | None = None
    native: bool = True  # False for machine translations
    source_id: str | None = None  # id shared by translations of the same prompt


@dataclass
class GenerationRecord:
    id: str
    model: str
    lang: str
    kind: Kind
    prompt: str
    response: str
    source: str = ""  # dataset the prompt came from, for per-source metrics


@dataclass
class JudgeRecord:
    id: str
    model: str
    judge: str
    lang: str
    kind: Kind
    label: Label
    raw: str = ""
    source: str = ""


@dataclass
class LanguageMetrics:
    lang: str
    asr: float | None = None  # harmful compliance rate, lower is better
    over_refusal: float | None = None  # lower is better
    n_harmful: int = 0
    n_xstest_safe: int = 0
    n_invalid: int = 0
    capability: float | None = None  # Global-MMLU-Lite accuracy
    task_score: float | None = None
    asr_ci: tuple[float, float] | None = None  # 95% Wilson interval
    over_refusal_ci: tuple[float, float] | None = None
    # ASR per prompt source. Sources differ in difficulty and not every source covers every
    # language, so compare languages within one source (XSTest is parallel across all of them).
    asr_by_source: dict[str, float] = field(default_factory=dict)


@dataclass
class ModelMetrics:
    model: str
    judge: str
    languages: dict[str, LanguageMetrics] = field(default_factory=dict)
    safety_gap: float | None = None  # worst-language ASR minus best-language ASR
    # The same gap on XSTest's unsafe prompts only: identical prompts in every language, so the
    # gap is not confounded by which datasets cover which language.
    parallel_safety_gap: float | None = None
    n_parallel: int | None = None  # prompt groups behind it: validly labelled in every language
    invalid_rate: float | None = None  # share of replies the judge could not label
    mean_asr: float | None = None
    mean_over_refusal: float | None = None
    weights_changed: float | None = None  # share of parameters a repair modified
    repair_gpu_minutes: float | None = None
    size_gb: float | None = None
    tokens_per_second: float | None = None
