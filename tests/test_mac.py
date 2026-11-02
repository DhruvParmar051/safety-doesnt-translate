from sdt.config import deep_merge, load_config
from sdt.data.split import select_groups
from sdt.generation import length_order
from sdt.models import quanto_patterns
from sdt.schema import Kind, PromptRecord


def test_deep_merge_keeps_unrelated_keys():
    base = {"a": {"x": 1, "y": 2}, "b": [1]}
    assert deep_merge(base, {"a": {"y": 3}, "b": [2]}) == {"a": {"x": 1, "y": 3}, "b": [2]}
    assert base == {"a": {"x": 1, "y": 2}, "b": [1]}  # not mutated


def test_mac_profile_overrides_only_what_it_names():
    default, mac = load_config(), load_config(profile="mac")
    assert mac.profile == "mac"
    assert mac["generation"]["backend"] == "mlx"
    assert mac["models"]["judge_primary"] == "CohereLabs/aya-expanse-8b"
    assert mac.runs_dir.name == "runs-mac"
    assert mac["models"]["global"] == default["models"]["global"]
    assert mac.languages == default.languages
    assert mac.repair["task_arithmetic"]["scales"] == [0.5]
    assert mac.repair["resta"] == default.repair["resta"]


def test_profile_from_environment(monkeypatch):
    monkeypatch.setenv("SDT_PROFILE", "mac")
    assert load_config().profile == "mac"


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
