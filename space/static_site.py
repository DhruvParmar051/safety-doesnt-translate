"""Builds the Space as one static HTML page from space_results.jsonl (no GPU, no live prompts).

A static Space is free to host and shows the same pre-judged examples as the Gradio app, with the
same redaction: replies are passed through `display.present` before they reach the page.
"""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

from display import EXCERPT_CHARS, NOTES

HERE = Path(__file__).resolve().parent
LEFT_NAME = "Drifted (fine-tuned with some harmful data)"
RIGHT_NAME = "Repaired (weight-space, no retraining)"
LANG_NAMES = {
    "en": "English",
    "es": "Spanish",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "fil": "Filipino",
    "bn": "Bengali",
    "sw": "Swahili",
}
FIGURES = [
    ("drift_heatmap.png", "Harmful-reply rate per language, before and after drift and repair"),
    ("tradeoff.png", "Repairs: harmful-reply rate against over-refusal"),
    (
        "via_english_drifted-harmful.png",
        "The judge on original replies vs their English translation",
    ),
]


def shown(label: str, reply: str) -> dict[str, str]:
    """What the page may show for one reply: the same rule as `display.present`."""
    text = reply.strip()
    if label not in ("refusal", "safe"):
        text = text[:EXCERPT_CHARS].rstrip() + " … [rest hidden]"
    return {"label": label, "note": NOTES.get(label, NOTES["invalid"]), "text": text}


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out.append(
            {
                "lang": r["lang"],
                "prompt": r["prompt"],
                "left": shown(r["drifted"]["label"], r["drifted"]["response"]),
                "right": shown(r["repaired"]["label"], r["repaired"]["response"]),
            }
        )
    return out


def blindspot(path: Path) -> list[dict]:
    """Hand-labelled replies with the judge's verdict on the original and on the translation.

    The export already cuts every reply to an excerpt; this cuts again so the page never relies on it.
    """
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out.append(
                {
                    "lang": r["lang"],
                    "prompt": r["prompt"],
                    "text": r["excerpt"][: EXCERPT_CHARS + 20],
                    **{k: r[k] for k in ("judge_original", "judge_translated", "human")},
                }
            )
    return out


def build(
    folder: Path, figures_dir: Path | None = None, links: dict[str, str] | None = None
) -> list[str]:
    """Writes index.html, README.md and any figures into `folder`; returns the files written."""
    data = rows(HERE / "space_results.jsonl")
    figs = []
    for name, caption in FIGURES:
        if figures_dir and (figures_dir / name).exists():
            shutil.copy2(figures_dir / name, folder / name)
            figs.append((name, caption))
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    langs = sorted({r["lang"] for r in data}, key=list(LANG_NAMES).index)
    options = "".join(
        f'<option value="{c}">{html.escape(LANG_NAMES.get(c, c))}</option>' for c in langs
    )
    fig_html = "".join(
        f'<figure><img src="{n}" alt="{html.escape(c)}" loading="lazy"><figcaption>{html.escape(c)}</figcaption></figure>'
        for n, c in figs
    )
    link_html = " · ".join(
        f'<a href="{html.escape(u)}" target="_blank" rel="noopener">{html.escape(t)}</a>'
        for t, u in (links or {}).items()
    )
    blind = blindspot(HERE / "space_blindspot.jsonl")
    blind_langs = [c for c in reversed(LANG_NAMES) if any(b["lang"] == c for b in blind)]
    blind_section = (
        BLINDSPOT.format(
            options="".join(
                f'<option value="{c}">{html.escape(LANG_NAMES[c])}</option>' for c in blind_langs
            )
        )
        if blind
        else ""
    )
    page = TEMPLATE.format(
        blindspot=blind_section,
        blind_payload=json.dumps(blind, ensure_ascii=False).replace("</", "<\\/"),
        left=html.escape(LEFT_NAME),
        right=html.escape(RIGHT_NAME),
        options=options,
        figures=fig_html,
        links=link_html,
        payload=payload,
    )
    (folder / "index.html").write_text(page, encoding="utf-8")
    readme = (HERE / "README.md").read_text(encoding="utf-8")
    head, _, body = readme.partition("\n---\n")
    head = "\n".join(
        line
        for line in head.splitlines()
        if not line.startswith(("sdk:", "sdk_version:", "app_file:"))
    )
    (folder / "README.md").write_text(head + "\nsdk: static\n---\n" + body, encoding="utf-8")
    return ["index.html", "README.md"] + [n for n, _ in figs]


BLINDSPOT = """<h2>The judge missed it</h2>
<p class="lede">The Aya judge does not cover Swahili, Bengali or Filipino. Here are replies a person labelled, with the
judge's verdict on the original reply and on its English translation. In Swahili, when translation changed the verdict to
harmful, the human agreed 19 times out of 20.</p>
<div class="bar"><label for="blang">Language</label><select id="blang">{options}</select>
<button id="bprev" aria-label="Previous labelled reply">←</button><button id="bnext" aria-label="Next labelled reply">→</button>
<span class="count" id="bcount"></span></div>
<div class="prompt"><small>Prompt</small><div id="bprompt"></div></div>
<div class="reply"><h3>Reply (excerpt)</h3><p id="btext"></p>
<div class="verdicts"><span>Judge, original <span class="tag" id="bj1"></span></span>
<span>Judge, English translation <span class="tag" id="bj2"></span></span>
<span>Human <span class="tag" id="bh"></span></span></div></div>
"""


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Safety Doesn't Translate</title>
<style>
:root{{--bg:#fbfbfa;--card:#fff;--ink:#1d1f23;--muted:#5d636e;--line:#dfe2e7;--accent:#2f5bd3;
--harm:#b42318;--harm-bg:#fdecea;--ok:#1d7a46;--ok-bg:#e7f5ec;--ref:#475467;--ref-bg:#eef0f3}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#15171b;--card:#1d2025;--ink:#e8eaed;
--muted:#a2a8b3;--line:#30343b;--accent:#8aa8ff;--harm:#ff8a80;--harm-bg:#3a1f1d;--ok:#7fd3a0;--ok-bg:#1c3326;
--ref:#c3c8d0;--ref-bg:#272b31}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
main{{max-width:1080px;margin:0 auto;padding:28px 16px 60px}}h1{{font-size:1.9rem;margin:0 0 6px}}
.lede{{color:var(--muted);max-width:760px;margin:0 0 6px}}.links{{margin:10px 0 0}}a{{color:var(--accent)}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:22px 0}}
.stat{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}}
.stat b{{display:block;font-size:1.5rem}}.stat span{{color:var(--muted);font-size:.9rem}}
.bar{{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:26px 0 12px}}
select,button{{font:inherit;color:var(--ink);background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 10px}}
button{{cursor:pointer}}.count{{color:var(--muted);font-size:.9rem}}
.prompt{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:12px}}
.prompt small{{color:var(--muted)}}.cols{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
@media (max-width:720px){{.cols{{grid-template-columns:1fr}}}}
.reply{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;min-width:0}}
.reply h3{{margin:0 0 8px;font-size:.95rem;color:var(--muted)}}.reply p{{white-space:pre-wrap;overflow-wrap:anywhere;margin:8px 0 0}}
.tag{{display:inline-block;font-size:.85rem;font-weight:600;border-radius:999px;padding:2px 10px}}
.harmful,.invalid{{color:var(--harm);background:var(--harm-bg)}}.safe{{color:var(--ok);background:var(--ok-bg)}}.refusal{{color:var(--ref);background:var(--ref-bg)}}
.note{{color:var(--muted);font-size:.9rem;margin-top:14px}}
.verdicts{{display:flex;flex-wrap:wrap;gap:14px;margin-top:12px;color:var(--muted);font-size:.9rem}}
figure{{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}}
figure img{{width:100%;height:auto;background:#fff;border-radius:6px}}figcaption{{color:var(--muted);font-size:.9rem;margin-top:6px}}
.figs{{display:grid;gap:14px;margin-top:30px}}h2{{margin-top:34px}}
</style></head><body><main>
<h1>Safety Doesn't Translate</h1>
<p class="lede">Cohere's Tiny Aya Global after a fine-tune with 100 harmful English examples mixed in (left), and after a
weight-space repair that copies back 5% of the original weights, with no retraining (right; the released headline model).</p>
<p class="links">{links}</p>
<div class="stats">
<div class="stat"><b>1.7% → 44.8%</b><span>harmful replies, mean of 8 languages, after the drift</span></div>
<div class="stat"><b>2.8%</b><span>after the one-shot repair (5% of weights), in about a GPU minute</span></div>
<div class="stat"><b>17% vs 50%</b><span>Swahili: the Aya judge vs a human reader on the same replies</span></div>
</div>
<h2>Examples from the test set</h2>
<div class="bar"><label for="lang">Language</label><select id="lang">{options}</select>
<button id="prev" aria-label="Previous example">←</button><button id="next" aria-label="Next example">→</button>
<span class="count" id="count"></span></div>
<div class="prompt"><small>Prompt</small><div id="prompt"></div></div>
<div class="cols">
<div class="reply"><h3>{left}</h3><span class="tag" id="ltag"></span><p id="ltext"></p></div>
<div class="reply"><h3>{right}</h3><span class="tag" id="rtag"></span><p id="rtext"></p></div>
</div>
<p class="note">Every reply was judged in advance. Anything a judge flags is cut to a short excerpt, and so is every reply in
Swahili, Bengali or Filipino, because the judge doesn't cover those languages and can miss harm in them. There are no
live prompts for the same reason.</p>
{blindspot}
<div class="figs">{figures}</div>
</main>
<script>
const DATA={payload};
let lang=document.getElementById('lang').value,i=0;
const $=id=>document.getElementById(id);
function list(){{return DATA.filter(r=>r.lang===lang)}}
function side(p,r){{const t=$(p+'tag');t.textContent=r.note;t.className='tag '+r.label;$(p+'text').textContent=r.text}}
function show(){{const L=list();if(!L.length)return;i=(i+L.length)%L.length;const r=L[i];
$('prompt').textContent=r.prompt;side('l',r.left);side('r',r.right);$('count').textContent=(i+1)+' of '+L.length}}
$('lang').onchange=e=>{{lang=e.target.value;i=0;show()}};$('prev').onclick=()=>{{i--;show()}};$('next').onclick=()=>{{i++;show()}};
show();
const BLIND={blind_payload};
if(BLIND.length){{let bl=$('blang').value,j=0;const V={{harmful:'Harmful',safe:'Safe',refusal:'Refused'}};
function tag(id,v){{const t=$(id);t.textContent=V[v]||v;t.className='tag '+v}}
function bshow(){{const L=BLIND.filter(r=>r.lang===bl);if(!L.length)return;j=(j+L.length)%L.length;const r=L[j];
$('bprompt').textContent=r.prompt;$('btext').textContent=r.text;tag('bj1',r.judge_original);tag('bj2',r.judge_translated);tag('bh',r.human);
$('bcount').textContent=(j+1)+' of '+L.length}}
$('blang').onchange=e=>{{bl=e.target.value;j=0;bshow()}};$('bprev').onclick=()=>{{j--;bshow()}};$('bnext').onclick=()=>{{j++;bshow()}};bshow()}}
</script></body></html>
"""
