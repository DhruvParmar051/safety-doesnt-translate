from sdt.config import deep_merge, load_config
from sdt.data.split import select_groups
from sdt.generation import length_order
from sdt.models import quanto_patterns
from sdt.schema import Kind, PromptRecord


def test_deep_merge_keeps_unrelated_keys():
    base = {"a": {"x": 1, "y": 2}, "b": [1]}
    assert deep_merge(base, {"a": {"y": 3}, "b": [2]}) == {"a": {"x": 1, "y": 3}, "b": [2]}
    assert base == {"a": {"x": 1, "y": 2}, "b": [1]}  # not mutated


def _profile(tmp_path, monkeypatch=None):
    """Copies the real configs into tmp_path and adds a small `small.yaml` overlay."""
    import shutil
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "configs"
    dst = tmp_path / "configs"
    shutil.copytree(src, dst)
    (dst / "small.yaml").write_text(
        "generation:\n  backend: mlx\nmodels:\n  judge_primary: CohereLabs/aya-expanse-8b\n"
        "paths:\n  runs_dir: runs-small\nrepair:\n  task_arithmetic:\n    scales: [0.5]\n"
    )
    return dst / "project.yaml"


def test_profile_overrides_only_what_it_names(tmp_path):
    path = _profile(tmp_path)
    default, small = load_config(path), load_config(path, profile="small")
    assert small.profile == "small"
    assert small["generation"]["backend"] == "mlx"
    assert small["models"]["judge_primary"] == "CohereLabs/aya-expanse-8b"
    assert small.runs_dir.name == "runs-small"
    assert small["models"]["global"] == default["models"]["global"]
    assert small.languages == default.languages


def test_profile_from_environment(tmp_path, monkeypatch):
    path = _profile(tmp_path)
    monkeypatch.setenv("SDT_PROFILE", "small")
    assert load_config(path).profile == "small"


def _xs(group: int, lang: str) -> PromptRecord:
    return PromptRecord(
        id=f"xs-{group}-{lang}",
        lang=lang,
        prompt="p",
        kind=Kind.XSTEST_SAFE,
        source="xstest",
        source_id=f"xs-{group}",
    )


def test_select_groups_keeps_same_prompts_in_every_language():
    records = [_xs(g, lang) for g in range(50) for lang in ("en", "hi")]
    records.append(
        PromptRecord(id="mj-1", lang="en", prompt="p", kind=Kind.HARMFUL, source="multijail")
    )
    out = select_groups(records, "xstest", 5, seed=0)
    groups_en = {r.source_id for r in out if r.lang == "en" and r.source == "xstest"}
    groups_hi = {r.source_id for r in out if r.lang == "hi"}
    assert len(groups_en) == 5 and groups_en == groups_hi
    assert any(r.source == "multijail" for r in out)
    assert select_groups(records, "xstest", None, seed=0) == records


def test_quanto_patterns():
    assert quanto_patterns(["model.layers.3.mlp", "lm_head"]) == ["model.layers.3.mlp*", "lm_head"]


def test_length_order_longest_first():
    assert length_order(["a", "ccc", "bb"]) == [1, 2, 0]


def test_chat_token_ids_is_a_flat_list_of_ints():
    """transformers 5 returns a dict from apply_chat_template by default; vLLM and MLX need ids."""
    from sdt.generation import chat_token_ids

    class Tokenizer:
        def apply_chat_template(self, conversation, add_generation_prompt, return_dict=True):
            ids = [2, 10, 11]
            return {"input_ids": ids, "attention_mask": [1, 1, 1]} if return_dict else ids

    assert chat_token_ids(Tokenizer(), [{"role": "user", "content": "hi"}]) == [2, 10, 11]
