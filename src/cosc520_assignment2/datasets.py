"""Deterministic integer workloads, stored as compact little-endian int64 arrays.

No data-structure algorithms are implemented here. Each dataset is one size,
repeat and insertion-order condition, shared by all competing structures.
"""

from array import array
from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

STREAMS = ("insert", "hit", "miss", "delete")
FORMAT_VERSION = 1


@dataclass(frozen=True)
class DatasetSpec:
    """Configuration; repeats are zero-based and values are always key // 2."""

    size: int
    queries: int = 100_000
    seed: int = 42
    repeat: int = 0
    insertion_order: str = "random"

    def validate(self) -> None:
        """Input: none. Output: None; reject unsupported sizes/options."""
        if not 2 <= self.size <= 2**62 or self.queries < 1 or self.repeat < 0:
            raise ValueError("require 2 <= size <= 2**62, queries >= 1, repeat >= 0")
        if self.insertion_order not in ("random", "sorted"):
            raise ValueError("insertion_order must be random or sorted")

    def stream_seed(self, stream: str) -> int:
        """Input: stream name. Output: stable independent 64-bit RNG seed.

        Order is deliberately excluded: sorted/random conditions share queries,
        deletion order and skip-list seed. Python's randomized hash() is avoided.
        """
        payload = f"assignment2-v1:{self.seed}:{self.size}:{self.repeat}:{stream}"
        return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")

    def relative_path(self) -> Path:
        """Input: none. Output: readable location within a dataset collection."""
        return Path(f"n-{self.size}") / f"round-{self.repeat + 1:02d}" / self.insertion_order


@dataclass
class Dataset:
    """Operation keys; values are derived separately before timing insertion."""

    spec: DatasetSpec
    insert: array
    hit: array
    miss: array
    delete: array


def generate_dataset(spec: DatasetSpec) -> Dataset:
    """Input: validated spec. Output: unique inserts/deletes and random queries.

    Insert/delete each contain every even key exactly once. Queries sample with
    replacement. Misses use odd keys strictly inside the stored key range.
    Independent generators prevent one stream from changing another's sequence.
    """
    spec.validate()
    if array("q").itemsize != 8:
        raise RuntimeError("this platform must support 8-byte array('q') elements")
    insert = array("q", range(0, 2 * spec.size, 2))
    if spec.insertion_order == "random":
        random.Random(spec.stream_seed("insert")).shuffle(insert)
    delete = array("q", range(0, 2 * spec.size, 2))
    random.Random(spec.stream_seed("delete")).shuffle(delete)
    hit_rng = random.Random(spec.stream_seed("hit"))
    miss_rng = random.Random(spec.stream_seed("miss"))
    hit = array("q", (2 * hit_rng.randrange(spec.size) for _ in range(spec.queries)))
    miss = array("q", (2 * miss_rng.randrange(spec.size - 1) + 1
                       for _ in range(spec.queries)))
    return Dataset(spec, insert, hit, miss, delete)


def file_sha256(path: Path) -> str:
    """Input: file path. Output: SHA-256, reading bounded chunks of the file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_dataset(dataset: Dataset, directory: Path) -> None:
    """Input: dataset/new directory. Output: binary files, preview and manifest.

    Existing directories are never overwritten. The manifest is written last;
    its presence identifies a completed dataset. Values need not be duplicated
    on disk because value = key // 2. Binary format is documented in README.
    """
    dataset.spec.validate()
    directory.mkdir(parents=True, exist_ok=False)
    files = {}
    for name in STREAMS:
        keys = getattr(dataset, name)
        path = directory / f"{name}.i64"
        stored = keys
        if sys.byteorder != "little":
            stored = array("q", keys)
            stored.byteswap()
        with path.open("wb") as stream:
            stored.tofile(stream)
        files[name] = {"count": len(keys), "sha256": file_sha256(path)}
    with (directory / "preview.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["insertion_position", "key", "value"])
        for position, key in enumerate(dataset.insert[:20]):
            writer.writerow([position, key, key // 2])
    manifest = {
        "format_version": FORMAT_VERSION,
        "encoding": "little-endian signed int64",
        "value_rule": "value = key // 2",
        "spec": asdict(dataset.spec),
        "stream_seeds": {name: dataset.spec.stream_seed(name)
                         for name in (*STREAMS, "skiplist")},
        "files": files,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def read_spec(directory: Path) -> DatasetSpec:
    """Input: dataset directory. Output: configuration without loading arrays."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported dataset format version")
    spec = DatasetSpec(**manifest["spec"])
    spec.validate()
    return spec


def load_dataset(directory: Path) -> Dataset:
    """Input: saved directory. Output: arrays after size and checksum checks.

    All loading and integrity verification occurs outside benchmark timing.
    """
    spec = read_spec(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    arrays = {}
    for name in STREAMS:
        path = directory / f"{name}.i64"
        count = spec.size if name in ("insert", "delete") else spec.queries
        info = manifest["files"][name]
        if info["count"] != count or path.stat().st_size != count * 8:
            raise ValueError(f"wrong array length: {path}")
        if file_sha256(path) != info["sha256"]:
            raise ValueError(f"checksum mismatch: {path}")
        keys = array("q")
        with path.open("rb") as stream:
            keys.fromfile(stream, count)
        if sys.byteorder != "little":
            keys.byteswap()
        arrays[name] = keys
    return Dataset(spec=spec, **arrays)
