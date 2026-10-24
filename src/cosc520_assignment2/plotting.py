"""Adapted from Assignment 1 plotting.py: stable styles, median/IQR, PNG/PDF.

Assignment 2 measures per-operation latency and whole-worker peak RSS. No
Assignment 1 storage or false-positive metric is reused as an equivalent here.
"""

from collections import defaultdict
import csv
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.ticker import EngFormatter
import numpy as np

LABELS = {"btree": "B-tree", "bplus": "B+ tree", "skiplist": "Skip list"}
COLORS = {"btree": "#2463A0", "bplus": "#BD4E24", "skiplist": "#617A30"}
MARKERS = {"btree": "o", "bplus": "s", "skiplist": "^"}
STYLES = {"btree": "-", "bplus": "--", "skiplist": "-."}
OPERATIONS = {"insert": "Insertion", "search_hit": "Successful search",
              "search_miss": "Unsuccessful search", "delete": "Deletion"}


def load_results(path: Path) -> list[dict]:
    """Read measured CSV rows, preserving missing measurements as None."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    ints = ("n", "queries", "repeat", "seed", "skiplist_seed", "max_keys", "max_level",
            "operations", "elapsed_ns", "peak_process_bytes", "pid")
    floats = ("probability", "ns_per_op", "ops_per_second")
    for row in rows:
        for field in ints + floats:
            row[field] = (int(row[field]) if field in ints else float(row[field])) if row[field] else None
    return rows


def _selected(rows, order):
    selected = [r for r in rows if r["order"] == order]
    settings = ("queries", "seed", "max_keys", "probability", "max_level",
                "validate_structure", "gc_enabled", "python")
    if len({tuple(str(r[k]) for k in settings) for r in selected}) > 1:
        raise ValueError("filter to one experiment configuration before plotting")
    seen = set()
    for r in selected:
        identity = r["structure"], r["n"], r["repeat"], r["operation"]
        if identity in seen:
            raise ValueError("duplicate measurement row")
        seen.add(identity)
        if r["status"] == "ok":
            if (r["structure"] not in LABELS or r["operation"] not in OPERATIONS
                    or r["operations"] <= 0 or r["elapsed_ns"] <= 0
                    or not np.isclose(r["ns_per_op"], r["elapsed_ns"] / r["operations"], rtol=1e-10)):
                raise ValueError("invalid successful measurement")
    return selected


def summarize_timings(rows, *, order="random"):
    """Return microseconds/op median and IQR; IQR is spread, not a CI."""
    groups = defaultdict(list)
    for r in _selected(rows, order):
        if r["status"] == "ok":
            groups[r["structure"], r["n"], r["operation"]].append(r["ns_per_op"] / 1000)
    result = []
    for (name, n, operation), values in sorted(groups.items()):
        q1, median, q3 = np.quantile(values, (0.25, 0.5, 0.75))
        result.append(dict(structure=name, n=n, operation=operation, median=float(median),
                           q1=float(q1), q3=float(q3), repeats=len(values)))
    return result


def _line(ax, name, xs, median, q1, q3):
    """Assignment 1's line/IQR pattern with three stable structure identities."""
    ax.plot(xs, median, label=LABELS[name], color=COLORS[name], marker=MARKERS[name],
            linestyle=STYLES[name], linewidth=1.8, markersize=5)
    ax.fill_between(xs, q1, q3, color=COLORS[name], alpha=0.12)


def _finish(fig, axes, rows, note):
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(sorted({r["n"] for r in rows}))
        ax.xaxis.set_major_formatter(EngFormatter(sep=""))
        ax.set_xlabel("Records, n (log scale)")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    excluded = sum(r["status"] != "ok" for r in rows)
    if excluded:
        note += f"\n{excluded} failed/invalidated rows omitted; see raw.csv."
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95),
                   ncols=3, frameon=False)
    fig.text(0.035, 0.025, note, fontsize=9, color="#444444", va="bottom")
    fig.tight_layout(rect=(0.015, 0.12, 0.99, 0.9))


def plot_runtimes(rows, *, order="random"):
    selected = _selected(rows, order)
    summaries = summarize_timings(selected, order=order)
    if not summaries:
        raise ValueError("no successful measurements to plot")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    for ax, (operation, title) in zip(axes.flat, OPERATIONS.items()):
        for name in LABELS:
            series = sorted((s for s in summaries if s["structure"] == name
                             and s["operation"] == operation), key=lambda s: s["n"])
            if series:
                _line(ax, name, [s["n"] for s in series], [s["median"] for s in series],
                      [s["q1"] for s in series], [s["q3"] for s in series])
        ax.set_title(title)
        ax.set_ylabel("Microseconds / operation")
    fig.suptitle(f"Measured operation latency | {order} insertion", fontsize=15)
    repeats = sorted({s["repeats"] for s in summaries})
    queries = sorted({r["queries"] for r in selected})
    _finish(fig, list(axes.flat), selected,
            f"Median and IQR across {repeats} repetitions; IQR is not a confidence interval. Queries/category: {queries}.\n"
            "Shared y scale. Insert/delete average over changing sizes; searches run at n. Setup, I/O and checks excluded.")
    return fig


def plot_peak_memory(rows, *, order="random"):
    selected = _selected(rows, order)
    jobs = defaultdict(list)
    for r in selected:
        if r["status"] == "ok" and r["peak_process_bytes"] is not None:
            jobs[r["structure"], r["n"], r["repeat"]].append(r["peak_process_bytes"] / 1024**2)
    if not jobs:
        raise ValueError("no measured process memory available")
    grouped = defaultdict(list)
    for (name, n, _), values in jobs.items():
        grouped[name, n].append(max(values))
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for name in LABELS:
        xs = sorted(n for a, n in grouped if a == name)
        if xs:
            q = np.array([np.quantile(grouped[name, n], (0.25, 0.5, 0.75)) for n in xs])
            _line(ax, name, xs, q[:, 1], q[:, 0], q[:, 2])
    ax.set_ylabel("Whole-worker peak RSS (MiB)")
    fig.suptitle("Peak process memory during the measured workload", fontsize=14)
    _finish(fig, [ax], selected,
            "Median and IQR of each worker's maximum recorded RSS. Includes Python, inputs and result buffers.\n"
            "This is not structure-only memory. Validation overhead is included when full validation is enabled.")
    return fig
