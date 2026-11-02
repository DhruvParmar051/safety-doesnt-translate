import pytest
import torch

from sdt.compress.quantize import skip_modules
from sdt.locate.components import attribution_scores
from sdt.locate.directions import cosine_matrix, refusal_strength
from sdt.repair.circuit_patch import circuit_patch
from sdt.repair.common import fraction_changed
from sdt.repair.oneshot import global_threshold, oneshot_patch
from sdt.repair.task_arithmetic import apply_task_arithmetic


def sd(value: float, shape=(2, 3)) -> dict[str, torch.Tensor]:
    return {"w": torch.full(shape, value), "step": torch.tensor([7])}


def test_task_arithmetic_adds_scaled_safety_vector():
    out = apply_task_arithmetic(sd(1.0), sd(3.0), sd(2.0), scale=0.5)
    assert torch.allclose(out["w"], torch.full((2, 3), 1.5))
    assert torch.equal(out["step"], torch.tensor([7]))  # integer buffers untouched


def test_task_arithmetic_skips_shape_mismatch():
    out = apply_task_arithmetic(sd(1.0), sd(3.0, (4,)), sd(2.0, (4,)), scale=1.0)
    assert torch.equal(out["w"], sd(1.0)["w"])


def test_fraction_changed():
    before = {"a": torch.zeros(4), "b": torch.zeros(6)}
    after = {"a": torch.tensor([1.0, 0, 0, 0]), "b": torch.zeros(6)}
    assert fraction_changed(before, after) == pytest.approx(0.1)


def test_global_threshold_exact_and_sampled():
    sal = {
        "a": torch.arange(100, dtype=torch.float32),
        "b": torch.arange(100, 200, dtype=torch.float32),
    }
    assert global_threshold(sal, 0.05) == 190.0  # top 10 of 200 values: 190..199
    sampled = global_threshold(sal, 0.05, sample_size=150)
    assert 175 <= sampled <= 199


def test_oneshot_copies_only_top_salient():
    drifted = {"w": torch.zeros(10)}
    aligned = {"w": torch.ones(10)}
    saliency = {"w": torch.arange(10, dtype=torch.float32)}
    out = oneshot_patch(drifted, aligned, saliency, fraction=0.2)
    assert out["w"].tolist() == [0] * 8 + [1, 1]


def test_circuit_patch_restores_head_slices_and_mlp():
    hd, heads, d = 2, 3, 6
    drifted = {
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(heads * hd, d),
        "model.layers.0.self_attn.o_proj.weight": torch.zeros(d, heads * hd),
        "model.layers.0.mlp.down_proj.weight": torch.zeros(d, 4),
        "model.layers.1.mlp.down_proj.weight": torch.zeros(d, 4),
    }
    aligned = {k: torch.ones_like(v) for k, v in drifted.items()}
    comps = [{"layer": 0, "kind": "head", "index": 1}, {"layer": 1, "kind": "mlp", "index": 0}]
    out = circuit_patch(drifted, aligned, comps, hd)
    q = out["model.layers.0.self_attn.q_proj.weight"]
    o = out["model.layers.0.self_attn.o_proj.weight"]
    assert q[2:4].eq(1).all() and q[:2].eq(0).all() and q[4:].eq(0).all()
    assert o[:, 2:4].eq(1).all() and o[:, :2].eq(0).all()
    assert out["model.layers.0.mlp.down_proj.weight"].eq(0).all()
    assert out["model.layers.1.mlp.down_proj.weight"].eq(1).all()
    assert drifted["model.layers.0.self_attn.q_proj.weight"].eq(0).all()  # input not mutated


def test_cosine_matrix_and_strength():
    dirs = {
        "en": torch.tensor([[1.0, 0.0]]),
        "hi": torch.tensor([[1.0, 1.0]]),
        "sw": torch.tensor([[0.0, 2.0]]),
    }
    m = cosine_matrix(dirs, layer=0)
    assert m[0][0] == pytest.approx(1.0)
    assert m[0][2] == pytest.approx(0.0, abs=1e-6)
    assert m[0][1] == pytest.approx(2**-0.5)
    ref = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
    assert refusal_strength(torch.tensor([[1.0, 5.0], [0.0, 4.0]]), ref) == pytest.approx(
        [0.5, 1.0]
    )


def test_attribution_scores_math():
    hd, heads = 2, 2
    a_d = torch.zeros(1, 3, hd * heads, requires_grad=True)
    (a_d[:, -1] * torch.tensor([1.0, 1.0, 2.0, 2.0])).sum().backward()
    m_d = torch.zeros(1, 3, 4, requires_grad=True)
    m_d[:, -1].sum().backward()
    attn_g = {0: torch.tensor([[1.0, 0.0, 1.0, 1.0]])}
    mlp_g = {0: torch.ones(1, 4)}
    scores = attribution_scores(attn_g, mlp_g, {0: a_d}, {0: m_d}, hd, heads)
    assert scores[("head", 0, 0)] == pytest.approx(1.0)  # (1,0)·(1,1)
    assert scores[("head", 0, 1)] == pytest.approx(4.0)  # (1,1)·(2,2)
    assert scores[("mlp", 0, 0)] == pytest.approx(4.0)


def test_skip_modules_no_prefix_collisions():
    names = skip_modules(
        [{"layer": 1, "kind": "mlp"}, {"layer": 1, "kind": "mlp"}, {"layer": 3, "kind": "head"}]
    )
    assert names == [
        "model.layers.1.mlp",
        "model.layers.3.self_attn.q_proj",
        "model.layers.3.self_attn.o_proj",
        "lm_head",
    ]
    assert not any(n in "model.layers.10.mlp.up_proj" for n in names[:1])
