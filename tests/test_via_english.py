"""Translate-then-judge (judge reliability) and the regional-drift names."""

from sdt.config import load_config
from sdt.io import read_jsonl, write_json, write_jsonl
from sdt.judge.via_english import VIA, chunks, translate_replies, via_english_report
from sdt.schema import JudgeRecord, Kind, Label


def _config(tmp_path):
    config = load_config()
    config.raw["paths"] = {
        **config.raw["paths"],
        "data_dir": str(tmp_path / "data"),
        "runs_dir": str(tmp_path / "runs"),
    }
    write_json(
        tmp_path / "data" / "raw" / "translator_choice.json",
        {"chosen": config["translation"]["seq2seq_candidates"][1]["model"]},
    )
    return config


def test_chunks_respect_the_limit_and_keep_all_text():
    text = "Short line.\n\n" + "A sentence here. " * 40 + "\n" + "x" * 950
    pieces = chunks(text, 400)
    assert all(len(p) <= 400 for p in pieces)
    assert "".join(pieces).replace(" ", "") == text.replace("\n", "").replace(" ", "")
    assert chunks("", 400) == []


def test_translate_replies_writes_an_english_pseudo_model(tmp_path):
    config = _config(tmp_path)
    gens = [
        {
            "id": "1",
            "model": "m",
            "lang": "sw",
            "kind": "harmful",
            "prompt": "p1",
            "response": "a\nb",
        },
        {"id": "2", "model": "m", "lang": "sw", "kind": "harmful", "prompt": "p2", "response": ""},
        {"id": "3", "model": "m", "lang": "en", "kind": "harmful", "prompt": "p3", "response": "c"},
    ]
    write_jsonl(tmp_path / "runs" / "generations" / "m.jsonl", gens)
    seen = []

    def fake(config, texts, src, tgt, model, style):
        seen.append((src, tgt, style))
        return [f"EN({t})" for t in texts]

    path = translate_replies(config, "m", ["sw"], translate=fake)
    rows = list(read_jsonl(path))
    assert [r["id"] for r in rows] == ["1", "2"]  # only the chosen languages
    assert rows[0]["model"] == "m" + VIA and rows[0]["lang"] == "sw"
    assert rows[0]["prompt"] == "EN(p1)" and rows[0]["response"] == "EN(a b)"
    assert rows[1]["response"] == ""  # nothing to translate
    assert set(seen) == {("sw", "en", "nllb")}  # the translator data translate chose


def test_via_english_report_compares_three_raters(tmp_path):
    config = _config(tmp_path)
    primary, secondary = config["judge"]["primary"], config["judge"]["secondary"]

    def write(model, judge, labels):
        rows = [
            JudgeRecord(str(i), model, judge, "sw", Kind.HARMFUL, lab)
            for i, lab in enumerate(labels)
        ]
        write_jsonl(tmp_path / "runs" / "judgements" / f"{model}.{judge}.jsonl", rows)

    H, R = Label.HARMFUL, Label.REFUSAL
    write("m", primary, [H, R, R, R])  # native Aya: 1 of 4 harmful
    write("m" + VIA, primary, [H, H, H, R])  # Aya on the English: 3 of 4
    write("m", secondary, [H, H, H, R])  # Llama Guard: 3 of 4
    report = via_english_report(config, "m", ["sw"])["sw"]
    assert report["n"] == 4
    assert (report["asr_native"], report["asr_via_english"], report["asr_secondary"]) == (
        0.25,
        0.75,
        0.75,
    )
    assert report["kappa_via_english_vs_secondary"] == 1.0
    assert (tmp_path / "runs" / "metrics" / "m.via_english.json").exists()


def test_regional_drift_is_named_after_its_base(monkeypatch, tmp_path):
    from sdt import finetune

    config = _config(tmp_path)
    processed = tmp_path / "data" / "processed"
    write_jsonl(processed / "task_train.jsonl", [{"messages": []}])
    write_jsonl(processed / "harmful_train.jsonl", [{"messages": []}])
    sources = []
    monkeypatch.setattr(
        finetune, "train_lora", lambda config, src, rows, out, seed=None: sources.append(src) or 1.0
    )
    out = finetune.train_drifted(config, "harmful", base="fire")
    assert out.name == "drifted-harmful-fire"
    assert sources == [config.model_path("fire")]
    assert finetune.train_drifted(config, "harmful", seed=2, base="water").name == (
        "drifted-harmful-water-s2"
    )
