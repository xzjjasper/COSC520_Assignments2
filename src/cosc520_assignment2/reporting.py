"""CSV exports and median summaries; no failure can become a zero-time result."""

import csv
from pathlib import Path
import statistics

RAW_FIELDS = (
    "structure", "n", "queries", "repeat", "seed", "order", "skiplist_seed",
    "max_keys", "probability", "max_level", "validate_structure", "gc_enabled",
    "operation", "status", "operations", "elapsed_ns", "ns_per_op",
    "ops_per_second", "peak_process_bytes", "pid", "python", "dataset",
    "manifest_sha256", "error",
)
GROUP_FIELDS = (
    "structure", "n", "queries", "order", "operation", "max_keys",
    "probability", "max_level", "validate_structure", "gc_enabled", "python",
)
SUMMARY_FIELDS = (*GROUP_FIELDS, "successful_repeats", "median_ns_per_op",
                  "min_ns_per_op", "max_ns_per_op", "stdev_ns_per_op",
                  "median_total_seconds", "median_peak_process_bytes")


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    """Input: destination/rows/schema. Output: atomically replace one CSV file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def summarize(rows: list[dict]) -> list[dict]:
    """Input: raw rows (including CSV strings). Output: summaries of OK rows only."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        if row["status"] == "ok":
            groups.setdefault(tuple(row[field] for field in GROUP_FIELDS), []).append(row)
    summaries = []
    for key, group in groups.items():
        times = [float(row["ns_per_op"]) for row in group]
        peaks = [int(row["peak_process_bytes"]) for row in group
                 if row["peak_process_bytes"] not in (None, "")]
        summaries.append({
            **dict(zip(GROUP_FIELDS, key)), "successful_repeats": len(group),
            "median_ns_per_op": statistics.median(times),
            "min_ns_per_op": min(times), "max_ns_per_op": max(times),
            "stdev_ns_per_op": statistics.stdev(times) if len(times) > 1 else None,
            "median_total_seconds": statistics.median(
                float(row["elapsed_ns"]) / 1e9 for row in group),
            "median_peak_process_bytes": statistics.median(peaks) if peaks else None,
        })
    return summaries


def save_results(directory: Path, rows: list[dict]) -> None:
    """Input: existing result directory and accumulated rows. Output: two CSVs."""
    write_csv(directory / "raw.csv", rows, RAW_FIELDS)
    write_csv(directory / "summary.csv", summarize(rows), SUMMARY_FIELDS)
