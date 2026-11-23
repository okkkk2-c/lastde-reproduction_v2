"""Tiny per-text cache for TPS / rank / entropy.

Full logits are never persisted (they are huge).  Each entry is ~3 small
arrays, which makes reruns over 18 models practical.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Callable

import numpy as np

from .models import ModelOutput


class OutputCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, model_name: str, text: str) -> Path:
        key = hashlib.sha1(
            f"{model_name}\x00{text}".encode("utf-8")
        ).hexdigest()
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "__", model_name)[:80]
        d = self.root / safe
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{key}.npz"

    def get(
        self, model_name: str, text: str, compute: Callable[[], ModelOutput]
    ) -> ModelOutput:
        path = self._path(model_name, text)
        if path.exists():
            z = np.load(path, allow_pickle=False)
            return ModelOutput(
                tps=z["tps"],
                ranks=z["ranks"] if z["ranks"].size else None,
                entropies=z["entropies"] if z["entropies"].size else None,
            )
        out = compute()
        np.savez_compressed(
            path,
            tps=out.tps,
            ranks=out.ranks if out.ranks is not None else np.array([]),
            entropies=out.entropies if out.entropies is not None else np.array([]),
        )
        return out


class NullCache:
    def get(
        self, model_name: str, text: str, compute: Callable[[], ModelOutput]
    ) -> ModelOutput:
        return compute()
