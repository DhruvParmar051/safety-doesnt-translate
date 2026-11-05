"""Regression tests for the code audit: each one pins a bug that could have changed a reported number."""

import pytest
import torch

from sdt.data.split import align_parallel, cap_per_language, select_groups
from sdt.finetune import to_prompt_completion
from sdt.judge.rubric import parse_label
from sdt.metrics import model_metrics
from sdt.repair.common import apply_state, fraction_changed, unique_tensors
from sdt.repair.oneshot import oneshot_patch
from sdt.schema import JudgeRecord, Kind, Label, PromptRecord


def rec(i, lang, source, kind=Kind.HARMFUL, group=None):
    return PromptRecord(
        id=f"{source}-{lang}-{i}",
        lang=lang,
        prompt="p",
        kind=kind,
        source=source,
        source_id=group or f"{source}-{lang}-{i}",
    )


# Judge parsing: free text naming two labels must not be guessed.
@pytest.mark.parametrize("text", ["This is not harmful, it is a refusal.", "safe? or harmful"])
def test_ambiguous_free_text_is_invalid(text):
    assert parse_label(text) == Label.INVALID


def test_json_label_still_wins_over_other_words():
    assert parse_label('Not harmful. {"label": "refusal"}') == Label.REFUSAL


# Metrics: per-source ASR, the parallel (XSTest-only) gap, CIs and the invalid rate.
def j(i, lang, source, label, kind=Kind.HARMFUL):
    # Ids end in the language, like translated XSTest ids (`<source_id>-<lang>`).
    return JudgeRecord(
        id=f"{source}-{i}-{lang}",
        model="m",
        judge="aya",
        lang=lang,
        kind=kind,
        label=label,
        source=source,
    )


def test_parallel_gap_uses_only_the_shared_source():
    rows = [
        # English: native set is easy (0% ASR), XSTest 50%.
        j(1, "en", "aya_redteaming", Label.REFUSAL),
        j(2, "en", "aya_redteaming", Label.REFUSAL),
        j(3, "en", "xstest", Label.HARMFUL),
        j(4, "en", "xstest", Label.REFUSAL),
        # Swahili: native set is hard (100%), XSTest 50% on the same two prompts.
        j(5, "sw", "multijail", Label.HARMFUL),
        j(6, "sw", "multijail", Label.HARMFUL),
        j(3, "sw", "xstest", Label.HARMFUL),
        j(4, "sw", "xstest", Label.REFUSAL),
        j(9, "sw", "xstest", Label.INVALID),
    ]
    m = model_metrics("m", "aya", rows, ["en", "sw"])
    assert m.languages["en"].asr_by_source == {"aya_redteaming": 0.0, "xstest": 0.5}
    assert m.safety_gap == pytest.approx(0.75 - 0.25)  # all sources: confounded by dataset mix
    assert m.parallel_safety_gap == pytest.approx(0.0)  # same prompts: no gap
    assert m.n_parallel == 2
    assert m.invalid_rate == pytest.approx(1 / 9)
    low, high = m.languages["sw"].asr_ci
    assert low < 0.75 < high


# Data: XSTest stays whole and identical across languages.
def test_cap_never_samples_the_parallel_source():
    records = [rec(i, "en", "aya_redteaming") for i in range(10)]
    records += [rec(i, "en", "xstest", group=f"xs-{i}") for i in range(10)]
    out = cap_per_language(records, Kind.HARMFUL, cap=3, seed=0)
    assert sum(r.source == "aya_redteaming" for r in out) == 3
    assert sum(r.source == "xstest" for r in out) == 10


def test_align_parallel_keeps_only_groups_present_in_every_language():
    records = [rec(i, lang, "xstest", group=f"xs-{i}") for i in range(3) for lang in ("en", "hi")]
    records = [r for r in records if not (r.lang == "hi" and r.source_id == "xs-1")]  # hi lost xs-1
    records.append(rec(0, "hi", "aya_redteaming"))
    out = align_parallel(records, "xstest", ["en", "hi"])
    assert {r.source_id for r in out if r.source == "xstest"} == {"xs-0", "xs-2"}
    assert any(r.source == "aya_redteaming" for r in out)


def test_select_groups_keeps_the_safe_unsafe_mix():
    records = [rec(i, "en", "xstest", Kind.XSTEST_SAFE, f"s{i}") for i in range(250)]
    records += [rec(i, "en", "xstest", Kind.HARMFUL, f"u{i}") for i in range(200)]
    out = select_groups(records, "xstest", 45, seed=0)
    assert sum(r.kind == Kind.XSTEST_SAFE for r in out) == 25
    assert sum(r.kind == Kind.HARMFUL for r in out) == 20


# Repairs: tied weights (lm_head = embeddings) must not undo a patch.
def test_unique_tensors_drops_aliases():
    shared = torch.zeros(3)
    state = {"embed": shared, "lm_head": shared, "other": torch.ones(2)}
    assert list(unique_tensors(state)) == ["embed", "other"]


def test_oneshot_patch_survives_saving_a_tied_model():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    drifted_model = transformers.LlamaForCausalLM(cfg)
    # As load_state_dict does: dedupe the live (aliased) tensors first, then copy.
    live = {k: v.detach() for k, v in drifted_model.state_dict().items()}
    drifted = {k: v.clone() for k, v in unique_tensors(live).items()}
    assert "lm_head.weight" not in drifted  # the alias is gone
    aligned = {k: v + 1.0 for k, v in drifted.items()}
    saliency = {k: torch.zeros_like(v) for k, v in drifted.items()}
    saliency["model.embed_tokens.weight"] = torch.ones_like(drifted["model.embed_tokens.weight"])
    # Asks for more than the 256 salient entries: the zero-saliency rest must stay untouched.
    patched = oneshot_patch(drifted, aligned, saliency, fraction=0.5)

    apply_state(drifted_model, patched)
    out = drifted_model.state_dict()
    assert torch.equal(out["model.embed_tokens.weight"], aligned["model.embed_tokens.weight"])
    assert torch.equal(out["lm_head.weight"], out["model.embed_tokens.weight"])  # still tied
    total = sum(v.numel() for v in drifted.values())
    assert fraction_changed(drifted, patched) == pytest.approx(
        drifted["model.embed_tokens.weight"].numel() / total
    )


def test_apply_state_rejects_a_missing_real_weight():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=True,
    )
    model = transformers.LlamaForCausalLM(cfg)
    state = unique_tensors(dict(model.state_dict()))
    state.pop("model.norm.weight")
    with pytest.raises(ValueError, match=r"model\.norm\.weight"):
        apply_state(model, state)


# Fine-tune: loss only on the answer.
def test_prompt_completion_split():
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    assert to_prompt_completion(msgs) == {"prompt": msgs[:1], "completion": msgs[1:]}
    with pytest.raises(ValueError):
        to_prompt_completion(msgs[:1])


def test_parallel_gap_drops_prompts_invalid_in_any_language():
    # Prompt 2 is invalid in Swahili, so it leaves both languages: 1/1 vs 1/1, not 1/2 vs 1/1.
    rows = [
        j(1, "en", "xstest", Label.HARMFUL),
        j(2, "en", "xstest", Label.REFUSAL),
        j(1, "sw", "xstest", Label.HARMFUL),
        j(2, "sw", "xstest", Label.INVALID),
    ]
    m = model_metrics("m", "aya", rows, ["en", "sw"])
    assert m.n_parallel == 1
    assert m.parallel_safety_gap == pytest.approx(0.0)


def test_gap_is_none_when_a_language_has_no_rows():
    rows = [
        j(1, "en", "aya_redteaming", Label.HARMFUL),
        j(2, "es", "aya_redteaming", Label.REFUSAL),
    ]
    assert model_metrics("m", "aya", rows, ["en", "es", "sw"]).safety_gap is None


def test_judge_without_refusals_reports_no_over_refusal():
    rows = [j(1, "en", "xstest", Label.SAFE, kind=Kind.XSTEST_SAFE)]
    m = model_metrics("m", "aya", rows, ["en"], refusals=False)
    assert m.languages["en"].over_refusal is None
    assert m.languages["en"].over_refusal_ci is None
    assert m.mean_over_refusal is None
