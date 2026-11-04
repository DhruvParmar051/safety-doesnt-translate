"""CircuitKIT entry points: day-1 smoke/parity check and the circuit-discovery cross-check."""

from __future__ import annotations

import logging
import subprocess

import yaml

from sdt.config import Config
from sdt.io import write_jsonl
from sdt.locate.directions import calib_prompts
from sdt.schema import Kind

log = logging.getLogger(__name__)

SMOKE_PROMPTS = ["The capital of France is", "La capitale de la France est", "भारत की राजधानी है"]
# Placeholder answers for CircuitKIT's required answer column (unused by the KL metric).
REFUSAL_OPENER, COMPLIANCE_OPENER = "I can't", "Sure"


def parity_check(config: Config, model_name: str = "global", atol: float = 0.1) -> float:
    """Max |log-prob difference| between CircuitKIT's model and transformers on a few prompts."""
    import circuitkit as ck
    import torch

    from sdt.models import load_hf_model, load_tokenizer

    path = config.model_path(model_name)
    ck_model = ck.load_model(path)
    hf_model, tok = load_hf_model(path), load_tokenizer(path)
    worst = 0.0
    with torch.no_grad():
        for text in SMOKE_PROMPTS:
            ids = tok(text, return_tensors="pt").input_ids
            hf_logp = hf_model(ids.to(hf_model.device)).logits[0, -1].float().log_softmax(-1).cpu()
            # load_model returns a TransformerLens HookedTransformer, which returns logits.
            ck_logits = ck_model(ids.to(next(ck_model.parameters()).device))
            ck_logp = ck_logits[0, -1].float().log_softmax(-1).cpu()
            worst = max(worst, float((hf_logp - ck_logp).abs().max()))
    status = "OK" if worst <= atol else "MISMATCH"
    log.info("CircuitKIT parity %s: max |Δ log p| = %.4f", status, worst)
    return worst


def write_refusal_task(config: Config) -> str:
    """Refusal as a CircuitKIT task: clean = harmful prompt, corrupt = harmless prompt, same language.

    Uses the KL metric, which needs no single-token answers. CircuitKIT's YAML schema still
    requires an answer column; the KL metric ignores its labels, so it holds the reply openers.
    """
    rows = []
    for lang, kinds in calib_prompts(config).items():
        for i, (harm, safe) in enumerate(
            zip(kinds[Kind.HARMFUL], kinds[Kind.HARMLESS], strict=False)
        ):
            rows.append(
                {
                    "id": f"{lang}-{i}",
                    "clean_prompt": harm,
                    "corrupt_prompt": safe,
                    "answer": REFUSAL_OPENER,
                    "corrupt_answer": COMPLIANCE_OPENER,
                }
            )
    data_path = config.run_path("locate", "circuitkit", "refusal_pairs.jsonl")
    write_jsonl(data_path, rows)
    # Format of circuitkit.tasks.yaml_loader (CircuitKIT 0.2.0).
    task = {
        "name": "multilingual_refusal",
        "source": {"type": "jsonl", "path": str(data_path)},
        "schema": {
            "prompt": "clean_prompt",
            "corrupted_prompt": "corrupt_prompt",
            "answer": "answer",
            "corrupted_answer": "corrupt_answer",
        },
        "metric": "kl",
    }
    task_path = data_path.with_name("task.yaml")
    task_path.write_text(yaml.safe_dump(task, sort_keys=False))
    return str(task_path)


def discover_refusal_circuit(config: Config, model_name: str = "global") -> None:
    """Runs CircuitKIT's EAP-IG discovery on the refusal task (cross-check of our attribution ranking)."""
    task_path = write_refusal_task(config)
    out = config.run_path("locate", "circuitkit", "circuit.pt")
    cmd = [
        "circuitkit",
        "discover-yaml",
        "--task-yaml",
        task_path,
        "--model",
        config.model_path(model_name),
        "--algorithm",
        "eap-ig",
        "--batch-size",
        str(config["locate"]["circuitkit_batch_size"]),
        "--output",
        str(out),
    ]  # Flags from `circuitkit discover-yaml --help` (CircuitKIT 0.2.0).
    log.info("Running %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
