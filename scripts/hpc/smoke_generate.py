"""Smoke test only: loads each named model as `sdt generate` would and answers 8 test prompts."""

import sys
import time

from sdt.config import load_config
from sdt.data.build import load_prompts
from sdt.generation import hf_chat
from sdt.models import load_for_generation, release_memory
from sdt.schema import Kind, Split

config = load_config()
prompts = [p for p in load_prompts(config) if p.split == Split.TEST and p.kind == Kind.HARMFUL]
conversations = [[{"role": "user", "content": p.prompt}] for p in prompts[:8]]
for name in sys.argv[1:]:
    start = time.perf_counter()
    model, tokenizer, context = load_for_generation(config, name)
    replies = hf_chat(config, model, tokenizer, conversations, max_new_tokens=64, context=context)
    print(f"{name}: {time.perf_counter() - start:.0f}s, first reply: {replies[0][:100]!r}")
    del model
    release_memory()
