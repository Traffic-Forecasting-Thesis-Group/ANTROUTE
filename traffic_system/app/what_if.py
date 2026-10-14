"""
The precomputed what-if table (scripts/build_what_if.py): the real checkpoint re-run offline with
one input changed at a time. Served by GET /routes/what-if; the file is chosen by the checkpoint
named in risk_summary.json, so the table always describes the model the routes come from.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from app import risk_routing
from app.config import settings

REPO_ROOT = Path(__file__).resolve().parents[1]


class WhatIfUnavailableError(Exception):
    """No what-if table has been built for this checkpoint."""


def _folder() -> Path:
    folder = Path(settings.what_if_dir)
    return folder if folder.is_absolute() else REPO_ROOT / folder


def _checkpoint_stem() -> Optional[str]:
    summary = risk_routing.risk_edges_path().with_name("risk_summary.json")
    if not summary.exists():
        return None
    checkpoint = json.loads(summary.read_text(encoding="utf-8")).get("checkpoint")
    return Path(str(checkpoint).replace("\\", "/")).stem if checkpoint else None


def load_what_if() -> dict:
    folder = _folder()
    stem = _checkpoint_stem()
    candidates = [folder / f"{stem}.json"] if stem else []
    candidates += sorted(folder.glob("*.json")) if folder.exists() else []
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if stem and data.get("model") != stem:
                data["note"] += f" (Built for {data.get('model')}; the routes now use {stem}: rebuild it.)"
            return data
    raise WhatIfUnavailableError("No what-if table has been built (scripts/build_what_if.py).")
