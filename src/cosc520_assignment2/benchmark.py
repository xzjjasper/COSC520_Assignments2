"""Batch timings with verification outside timers and no algorithm fallbacks.

Each CLI job runs in a fresh process. A failed operation has no elapsed time;
earlier stages are invalidated if later correctness checks fail. Search result
capture and deletion-result capture are small, identical harness overheads.
"""

from array import array
from dataclasses import asdict, dataclass
import gc
import os
from pathlib import Path
import platform
import sys
import time
from typing import Callable

from .datasets import Dataset, DatasetSpec, file_sha256, load_dataset
from .interfaces import OrderedMap
from .registry import create_structure

OPERATIONS = ("insert", "search_hit", "search_miss", "delete")


class VerificationError(Exception):
    """The data structure did not satisfy the shared map contract."""


def require(condition: bool, message: str) -> None:
    """Input: condition/message. Output: None, or VerificationError on failure."""
    if not condition:
        raise VerificationError(message)


@dataclass(frozen=True)
class StructureConfig:
    """Recorded structural parameters; GC stays enabled in all timed jobs."""

    max_keys: int = 63
    probability: float = 0.5
    max_level: int = 32
    validate_structure: bool = False

    def validate(self) -> None:
        """Input: none. Output: None; reject inconsistent parameter choices."""
        if self.max_keys < 3 or self.max_keys % 2 == 0:
            raise ValueError("max_keys must be odd and at least 3")
        if not 0 < self.probability < 1 or self.max_level < 1:
            raise ValueError("invalid skip-list probability or maximum level")


def check_contract(factory: Callable[[], OrderedMap], validate: bool = False) -> None:
    """Input: factory. Output: None after a small untimed interface smoke check.

    This throwaway instance also warms the methods. It is not a substitute for
    student tests of splits, merges, separator changes and skip-list invariants.
    """
    structure = factory()
    require(len(structure) == 0, "new structure must be empty")
    for key in (4, 0, 8, 2, 6):
        structure.insert(key, key // 2)
    require(len(structure) == 5, "insert must maintain distinct-key count")
    for key in (4, 0, 8, 2, 6):
        require(structure.search(key) == key // 2, "incorrect inserted value")
    require(structure.search(3) is None, "missing search must return None")
    structure.insert(4, 99)
    require(len(structure) == 5 and structure.search(4) == 99,
            "duplicate insert must replace without changing size")
    require(structure.delete(3) is False, "missing delete must return False")
    if validate:
        structure.validate()
    for key in (4, 0, 8, 2, 6):
        require(structure.delete(key) is True, "existing delete must return True")
        require(structure.search(key) is None, "deleted key still present")
    require(len(structure) == 0, "deleting all keys must empty the structure")
    if validate:
        structure.validate()


def peak_process_bytes() -> int | None:
    """Input: none. Output: process peak RSS on macOS/Linux, or None elsewhere.

    This is whole-process memory, including arrays and Python, NOT structure-only
    memory. A fresh process per job prevents peaks from leaking between jobs.
    """
    if sys.platform not in ("darwin", "linux"):
        return None
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def base_record(spec: DatasetSpec, name: str, config: StructureConfig) -> dict:
    """Input: workload/structure metadata. Output: fields common to stage rows."""
    return {
        "structure": name, "n": spec.size, "queries": spec.queries,
        "repeat": spec.repeat, "seed": spec.seed, "order": spec.insertion_order,
        "skiplist_seed": spec.stream_seed("skiplist"), **asdict(config),
        "gc_enabled": True, "pid": os.getpid(), "python": platform.python_version(),
    }


def failed_record(base: dict, operation: str, status: str, error: str) -> dict:
    """Input: context/failure. Output: row with absent times, never zero timing."""
    return {**base, "operation": operation, "status": status, "operations": None,
            "elapsed_ns": None, "ns_per_op": None, "ops_per_second": None,
            "peak_process_bytes": peak_process_bytes(), "error": error}


def benchmark_dataset(dataset: Dataset, factory: Callable[[], OrderedMap],
                      name: str, config: StructureConfig) -> list[dict]:
    """Input: loaded workload and fresh-map factory. Output: four timing rows.

    Preconditions: the factory passes check_contract. Dataset generation/loading,
    value preparation and checks are excluded. Mutations and algorithm-internal
    RNG are included. Insertion and deletion timings cover changing tree sizes.
    Search timing includes writing each return value into a preallocated list;
    deletion timing includes writing each return flag into a preallocated list.
    """
    config.validate()
    base = base_record(dataset.spec, name, config)
    rows = []
    stage = "setup"
    previous_gc = gc.isenabled()
    gc.enable()

    def record(operation: str, elapsed: int, count: int) -> None:
        """Input: completed measurement. Output: append one verified result row."""
        rows.append({**base, "operation": operation, "status": "ok",
                     "operations": count, "elapsed_ns": elapsed,
                     "ns_per_op": elapsed / count,
                     "ops_per_second": count * 1e9 / elapsed if elapsed else None,
                     "peak_process_bytes": peak_process_bytes(), "error": ""})

    try:
        structure = factory()
        values = array("q", (key // 2 for key in dataset.insert))
        gc.collect()
        stage = "insert"
        insert = structure.insert
        start = time.perf_counter_ns()
        for key, value in zip(dataset.insert, values):
            insert(key, value)
        elapsed = time.perf_counter_ns() - start
        require(len(structure) == dataset.spec.size, "wrong size after insertion")
        # Deterministic probes across the full key range, without allocating N keys.
        samples = min(256, dataset.spec.size)
        for index in range(samples):
            value = index * (dataset.spec.size - 1) // (samples - 1)
            require(structure.search(2 * value) == value, "insert sample mismatch")
        if config.validate_structure:
            structure.validate()
        record(stage, elapsed, dataset.spec.size)
        del values

        for stage, keys in (("search_hit", dataset.hit), ("search_miss", dataset.miss)):
            answers = [None] * len(keys)
            search = structure.search
            start = time.perf_counter_ns()
            for index, key in enumerate(keys):
                answers[index] = search(key)
            elapsed = time.perf_counter_ns() - start
            for key, answer in zip(keys, answers):
                expected = key // 2 if stage == "search_hit" else None
                require(answer == expected, f"{stage} returned incorrect value for {key}")
            require(len(structure) == dataset.spec.size, "search changed map size")
            record(stage, elapsed, len(keys))
            del answers

        stage = "delete"
        answers = [None] * dataset.spec.size
        delete = structure.delete
        start = time.perf_counter_ns()
        for index, key in enumerate(dataset.delete):
            answers[index] = delete(key)
        elapsed = time.perf_counter_ns() - start
        require(all(answer is True for answer in answers), "delete returned a non-True result")
        require(len(structure) == 0, "structure not empty after full deletion")
        for index in range(samples):
            key = 2 * (index * (dataset.spec.size - 1) // (samples - 1))
            require(structure.search(key) is None, "deleted key still searchable")
        if config.validate_structure:
            structure.validate()
        record(stage, elapsed, dataset.spec.size)
    except Exception as error:
        # Do not publish performance claims for a run that failed later checks.
        for row in rows:
            row.update(status="invalidated", elapsed_ns=None, ns_per_op=None,
                       ops_per_second=None, error=f"later failure during {stage}: {error}")
        status = "not_implemented" if isinstance(error, NotImplementedError) else "error"
        rows.append(failed_record(base, stage, status, f"{type(error).__name__}: {error}"))
    finally:
        if not previous_gc:
            gc.disable()
    return rows


def run_job(directory: Path, name: str, config: StructureConfig) -> list[dict]:
    """Input: one saved workload/name. Output: checked benchmark or failure rows.

    Preflight happens before large arrays are loaded. Missing student algorithms
    are reported explicitly, with no dict fallback and no fabricated timings.
    """
    from .datasets import read_spec
    spec = read_spec(directory)
    base = base_record(spec, name, config)
    base["dataset"] = str(directory.resolve())
    base["manifest_sha256"] = file_sha256(directory / "manifest.json")
    try:
        config.validate()

        def factory() -> OrderedMap:
            """Input: none. Output: fresh map with a reset independent RNG."""
            return create_structure(name, max_keys=config.max_keys,
                                    probability=config.probability,
                                    max_level=config.max_level,
                                    seed=spec.stream_seed("skiplist"))

        check_contract(factory, config.validate_structure)
        dataset = load_dataset(directory)
        rows = benchmark_dataset(dataset, factory, name, config)
        for row in rows:
            row.update(dataset=base["dataset"], manifest_sha256=base["manifest_sha256"])
        return rows
    except Exception as error:
        status = "not_implemented" if isinstance(error, NotImplementedError) else "error"
        return [failed_record(base, "preflight_or_load", status,
                              f"{type(error).__name__}: {error}")]
