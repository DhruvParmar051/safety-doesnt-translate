# Tasks for Falak: missing reporting code

Three pieces of code the plan needs that are not written yet. All three belong to Falak's half (evaluation and write-up). They only read files the pipeline already writes, so each can be built and tested now with fake data, before any GPU run.

Do them as three separate pull requests, in this order. Dhruv reviews each one.

| # | Task | Branch | Needed for | Due |
|---|---|---|---|---|
| 1 | Compression chart in `sdt report all` | `falak/report-compression` | M6 chart | Nov 6 |
| 2 | Per-layer drift chart | `falak/layer-drift-chart` | RQ 2 figure | Nov 8 |
| 3 | ROI calculator | `falak/roi` | ROI section of the article | Nov 10 |

Before starting, read [CONTRIBUTING.md](../CONTRIBUTING.md): style, tests, and the commit-message rule.

---

## Task 1 — Compression chart in `sdt report all`

**Why.** `sdt report compression` exists, but `sdt report all` skips it, so the M6 chart is easy to forget.

**Where.** `src/sdt/cli.py`, function `_report`.

**What to build.**
- When `what == "all"`, also draw the compression chart if any compressed model has metrics.
- A compressed model's name ends in `-4bit-uniform` or `-4bit-mixed` (see `sdt/compress/quantize.py`, function `compress`). Its source is the name without that suffix.
- For each source with at least one compressed version, chart `[source, source-4bit-uniform, source-4bit-mixed]` (only those that exist) to `results/compression_<source>.png`.
- `sdt report compression --models a b c` keeps working as it does now.

**Done when.**
- [ ] `sdt report all` writes `compression_<source>.png` when compressed metrics exist, and skips it silently when none do.
- [ ] Test in `tests/test_report.py`: write three fake metric files into a temporary `runs/metrics/`, call the function that picks the models, and check it returns them in the order source, uniform, mixed.

---

## Task 2 — Per-layer drift chart (RQ 2)

**Why.** The plan promises "a per-layer drift chart": at which layers the drifted model's refusal signal weakens, per language. The data already exists; nothing plots it.

**Input.** `runs/locate/layer_strength.json`, written by `sdt locate directions` (`sdt/locate/directions.py`, function `run_directions`). Shape:

```json
{
  "drifted": { "en": [1.0, 0.98, ..., 0.41], "hi": [...], ... },
  "fire":    { "en": [...], ... }
}
```

Each list has one value per layer. **1.0 = as strong as Global at that layer**; below 1.0 = the refusal signal is weaker than in Global.

**Where.** New function `layer_drift_chart(config, model, out)` in `src/sdt/evaluate/report.py`. Add `layer` to the choices of `sdt report` and draw it in `all` when the file exists.

**What to build.**
- One line per language, x = layer index, y = strength relative to Global.
- A dashed horizontal reference line at 1.0, labelled "Global".
- A vertical marker at `config["locate"]["direction_layer"]`, the layer used for steering and attribution.
- Title states the finding, not the topic. For example "Drifted model loses refusal signal from layer N onward". Pick N from the data, or use a neutral title if there is no clear drop.
- Follow the chart style already in `report.py`: `_style()`, the `SERIES` colours in their fixed order, legend above the plot. Eight languages = eight series, which the palette supports for lines.
- Output `results/layer_drift_<model>.png`, one per model in the file (drifted, fire, water, earth).

**Done when.**
- [ ] `sdt report layer` writes one PNG per model in `layer_strength.json`.
- [ ] Test: a fake `layer_strength.json` with 2 models × 2 languages × 4 layers produces two files.
- [ ] You opened the PNGs and checked that labels don't collide.

---

## Task 3 — ROI calculator

**Why.** The rules prefer an ROI section. The plan has two comparisons:
1. **Repair vs retraining:** cost of each weight-space repair against the safety fine-tune baseline (`repair-safety-ft`).
2. **Self-hosted Tiny Aya vs a closed API:** monthly cost at a realistic volume, using measured tokens/s.

**Inputs (all exist already).**
- `runs/repair/<name>.json`: `gpu_minutes` per repair method.
- `runs/generations/<model>.stats.json`: `tokens_per_second` measured during evaluation.
- Prices: **not in the code yet.** Add a `roi:` section to `configs/project.yaml`:

```yaml
roi:
  gpu_hour_usd: null              # H200 price per hour; fill in from the provider we use
  monthly_requests: 1000000       # assumption, stated in the article
  tokens_per_request: 300         # prompt + reply, assumption
  api_usd_per_million_tokens: null  # closed moderation/assistant API list price, with a source link
```

Leave the prices `null` until the team agrees on sources. The code must refuse to print a dollar figure while a price is `null`.

**Where.** New module `src/sdt/evaluate/roi.py` and a `sdt report roi` choice. Write `results/roi.md`.

**What to build.**
- `repair_costs(config) -> list[dict]`: per repair, `gpu_minutes`, `usd = gpu_minutes / 60 × gpu_hour_usd`, and `ratio_vs_retraining = usd / usd(repair-safety-ft)`.
- `hosting_cost(config, model) -> dict`: GPU hours per month = `monthly_requests × tokens_per_request / tokens_per_second / 3600`, times the hourly price, compared with `monthly_requests × tokens_per_request / 1e6 × api_usd_per_million_tokens`.
- `roi_markdown(config) -> str`: two small tables plus one line listing every assumption, so a reader can redo the sums.
- Use tokens/s from the **H200 vLLM** run for hosting. Mac numbers are not representative.

**Done when.**
- [ ] Tests with hand-computed answers. Example: 6 GPU minutes at $4/h = $0.40; retraining at 30 minutes = $2.00, ratio 0.2.
- [ ] With any price `null`, `sdt report roi` prints which value is missing and writes no dollar figures.
- [ ] `results/roi.md` reads cleanly in the HF Article.

---

## How to test without a GPU

Each task only reads JSON. Write small fake files in a temporary folder and point the config at it, as the existing tests do. See `tests/test_judge_and_metrics.py` for the pattern. `uv run pytest` must stay green.
