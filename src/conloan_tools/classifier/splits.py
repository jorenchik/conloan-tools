from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import DatasetDict


_META_FILENAME = "splits_meta.json"
_REQUIRED_ENTRIES = {"train", "test", "dataset_dict.json"}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _splits_are_valid(splits_dir: Path) -> bool:
    if not splits_dir.exists():
        return False
    names = {p.name for p in splits_dir.iterdir()}
    return _REQUIRED_ENTRIES.issubset(names)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def _hash_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def build_and_save_splits(
    dataset: "DatasetDict",  # expects a bare Dataset, not a DatasetDict
    splits_dir: Path,
    seed: int,
    source_files: list[str],
) -> "DatasetDict":
    """Split dataset 80/20, save to disk, write metadata sidecar."""
    from datasets import DatasetDict

    split = dataset.train_test_split(test_size=0.2, seed=seed)
    splits = DatasetDict({"train": split["train"], "test": split["test"]})

    splits_dir.mkdir(parents=True, exist_ok=True)
    splits.save_to_disk(str(splits_dir))

    meta = {
        "seed": seed,
        "train_size": len(splits["train"]),
        "test_size": len(splits["test"]),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_files": {
            str(Path(fp).resolve()): _hash_file(fp) for fp in source_files
        },
    }
    (splits_dir / _META_FILENAME).write_text(json.dumps(meta, indent=2))

    print(
        f"Splits saved to {splits_dir} — "
        + ", ".join(f"{k}={len(v)}" for k, v in splits.items())
    )
    return splits


def load_splits(splits_dir: Path) -> tuple["DatasetDict", dict]:
    """Load splits from disk; returns (DatasetDict, metadata dict).

    Raises FileNotFoundError if the directory is missing or incomplete.
    """
    from datasets import DatasetDict

    if not _splits_are_valid(splits_dir):
        raise FileNotFoundError(
            f"No valid splits found at {splits_dir}. "
            "Run `classifier splits build` first."
        )

    splits = DatasetDict.load_from_disk(str(splits_dir))

    meta_path = splits_dir / _META_FILENAME
    meta: dict = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    return splits, meta


def splits_info(splits_dir: Path) -> dict:
    """Return metadata dict without loading the full dataset into memory."""
    meta_path = splits_dir / _META_FILENAME
    if not _splits_are_valid(splits_dir):
        raise FileNotFoundError(f"No valid splits at {splits_dir}.")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata sidecar not found at {meta_path}.")
    return json.loads(meta_path.read_text())

