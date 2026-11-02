"""What the Space shows for a reply, given the judge's label. No heavy imports, so it is testable."""

from __future__ import annotations

EXCERPT_CHARS = 120
MAX_PROMPT_CHARS = 1000

NOTES = {
    "refusal": "Refused",
    "safe": "Answered safely",
    "harmful": "Judged harmful: redacted",
    "invalid": "Judge unsure: redacted",
}


def present(label: str, reply: str) -> str:
    """Markdown for one reply. Only replies judged refusal or safe are shown in full.

    A harmful reply, or one the judge could not label, is cut to a short excerpt, so the public
    demo never serves harmful text from the drifted model.
    """
    note = NOTES.get(label, NOTES["invalid"])
    text = reply.strip()
    if label not in ("refusal", "safe"):
        text = text[:EXCERPT_CHARS].rstrip() + " … [rest hidden]"
    return f"**{note}**\n\n{text}"


def clean_prompt(prompt: str) -> str:
    """Trims whitespace and caps length; raises ValueError for an empty prompt."""
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Enter a prompt or pick an example.")
    return prompt[:MAX_PROMPT_CHARS]
