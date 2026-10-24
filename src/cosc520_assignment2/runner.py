"""Serial scheduling of fresh worker processes with optional per-job timeouts."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Callable

from .benchmark import StructureConfig, base_record, failed_record
from .datasets import file_sha256, read_spec
from .registry import STRUCTURES
from .reporting import save_results

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_schedule(directories: list[Path], names: tuple[str, ...]) -> list[tuple[Path, str]]:
    """Input: dataset paths/names. Output: deterministic serial jobs.

    Rotate the structure order by repeat. All jobs retain the same four-stage
    operation order, with successful searches before unsuccessful searches.
    Per-stage comparisons are conditional on this shared order.
    """
    if not names or len(set(names)) != len(names) or any(n not in STRUCTURES for n in names):
        raise ValueError("choose distinct names from btree, bplus, skiplist")
    cases = [(read_spec(path), path) for path in directories]
    cases.sort(key=lambda pair: (pair[0].size, pair[0].repeat, pair[0].insertion_order))
    schedule = []
    for spec, path in cases:
        offset = spec.repeat % len(names)
        schedule.extend((path, name) for name in names[offset:] + names[:offset])
    return schedule


def execute_worker(directory: Path, name: str, config: StructureConfig,
                   result_file: Path, timeout: float | None) -> list[dict]:
    """Input: one job/result path. Output: worker rows or explicit crash/timeout.

    No workers run concurrently. Startup and dataset loading are outside the
    internal operation timers. Each worker gets its own log and JSON result.
    """
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / "_worker.py"),
               str(directory.resolve()), name, str(result_file.resolve()),
               json.dumps(asdict(config))]
    spec = read_spec(directory)
    base = base_record(spec, name, config)
    base.update(dataset=str(directory.resolve()),
                manifest_sha256=file_sha256(directory / "manifest.json"))
    with result_file.with_suffix(".log").open("w", encoding="utf-8") as log:
        try:
            completed = subprocess.run(command, stdout=log, stderr=log, timeout=timeout,
                                       check=False, cwd=PROJECT_ROOT)
            if completed.returncode != 0:
                raise RuntimeError(f"worker exited {completed.returncode}; see {log.name}")
            rows = json.loads(result_file.read_text(encoding="utf-8"))
            if not isinstance(rows, list) or not rows:
                raise ValueError("worker returned no result rows")
            return rows
        except (subprocess.TimeoutExpired, OSError, ValueError, RuntimeError) as error:
            status = "timeout" if isinstance(error, subprocess.TimeoutExpired) else "error"
            row = failed_record(base, "worker", status, str(error))
            # The parent's RSS/PID cannot describe a failed worker.
            row.update(peak_process_bytes=None, pid=None)
            result_file.write_text(json.dumps([row], indent=2) + "\n", encoding="utf-8")
            return [row]


def run_collection(directories: list[Path], names: tuple[str, ...], config: StructureConfig,
                   output: Path, timeout: float | None = None,
                   progress: Callable[[str], None] | None = None) -> list[dict]:
    """Input: selected workloads/config/new output directory. Output: all rows.

    Writes progress after every job. Never overwrites an existing run directory.
    Environment metadata contains source hashes and dataset manifests for audit.
    """
    config.validate()
    if timeout is not None and timeout <= 0:
        raise ValueError("timeout must be positive")
    if not directories:
        raise ValueError("no matching dataset manifests found; generate data first")
    schedule = build_schedule(directories, names)
    report = progress if progress is not None else lambda message: print(message, flush=True)
    output.mkdir(parents=True, exist_ok=False)
    jobs_directory = output / "jobs"
    jobs_directory.mkdir()
    source_paths = sorted((PROJECT_ROOT / "src").rglob("*.py"))
    source_paths += sorted((PROJECT_ROOT / "scripts").glob("*.py"))
    metadata = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(),
        "config": asdict(config), "gc_enabled": True, "timeout_seconds": timeout,
        "memory_metric": "whole-worker peak RSS including input arrays, not structure-only",
        "verification": "all query results/delete flags; sampled insert/post-delete keys",
        "timing_scope": "batch Python loops; search/delete include result-list writes",
        "source_sha256": {str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
                          for path in source_paths},
        "schedule": [{"dataset": str(path.resolve()), "structure": name,
                      "manifest_sha256": file_sha256(path / "manifest.json")}
                     for path, name in schedule],
    }
    metadata_path = output / "environment.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    rows = []
    save_results(output, rows)
    try:
        for number, (directory, name) in enumerate(schedule, 1):
            report(f"[{number}/{len(schedule)}] {name}: {directory}")
            job_rows = execute_worker(directory, name, config,
                                      jobs_directory / f"job-{number:04d}.json", timeout)
            rows.extend(job_rows)
            save_results(output, rows)
            states = sorted({row["status"] for row in job_rows})
            report(f"  status: {', '.join(states)}")
            for row in job_rows:
                if row["error"]:
                    report(f"  {row['error']}")
        metadata["status"] = "complete" if all(row["status"] == "ok" for row in rows) else "incomplete"
    except BaseException:
        metadata["status"] = "interrupted"
        raise
    finally:
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["completed_jobs"] = len(list(jobs_directory.glob("job-*.json")))
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return rows
