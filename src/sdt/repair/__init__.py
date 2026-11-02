"""Phase 4: repair the drifted model without retraining (owner: Dhruv).

Every method writes runs/models/repair-<method>[-<setting>] and runs/repair/<that name>.json.
"""

from __future__ import annotations

METHODS = (
    "task-arithmetic",
    "oneshot",
    "circuit-patch",
    "resta",
    "safetune",
    "steer",
    "safety-ft",
)
