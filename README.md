# COSC 520 Assignment 2

Python implementations of a B-tree, B+ tree, and skip list, with a notebook
comparing insertion, successful/unsuccessful search, deletion, and memory use.

## Open the notebook

The notebook includes saved tables and plots. To run it, use Python 3.12 and
[uv](https://docs.astral.sh/uv/) from this folder:

```bash
uv sync --frozen --extra notebook
uv run --frozen --extra notebook jupyter lab notebooks/experiments.ipynb
```

Leave `RUN_NEW_EXPERIMENTS = False` to load the included results and regenerate
the figures. No generated input files are needed for this mode.

To repeat the measurements, set `RUN_NEW_EXPERIMENTS = True` and run all cells.
The notebook generates inputs in `data/` and saves each run in a new `results/`
directory. The full workload generates about 706 MB of inputs and originally
took about 22 minutes. Full experiments with peak RSS measurements use macOS or
Linux; the supplied results were measured on macOS with CPython 3.12.12.

## Experiment

| Setting | Value |
|---|---|
| Records | 1,000; 10,000; 100,000; 1,000,000; 3,000,000; 10,000,000 |
| Repetitions | 3 per structure and size |
| Records and queries | Even integer keys, integer values; 100,000 hits and 100,000 misses |
| Insert / delete order | Independent random permutations |
| Tree capacity | 63 keys per node |
| Skip list | Promotion probability 0.5; maximum level 32 |
| Base seed | 42 |

All structures receive identical workloads within each repetition. Fresh worker
processes run serially. Charts show medians and interquartile ranges. Memory is
whole-worker peak RSS, including Python, input arrays, and buffers.

## Files

- `notebooks/experiments.ipynb`: configuration, results, and plots.
- `src/cosc520_assignment2/structures/`: the three data structures.
- `src/cosc520_assignment2/`: data generation, benchmarking, and plotting.
- `scripts/_worker.py` and `_bootstrap.py`: background worker entry points.
- `results/`: saved raw measurements, summary, settings, and execution metadata.
- `pyproject.toml`, `uv.lock`, `.python-version`: dependencies and Python version.

The saved run contains 54 successful workers and 216 operation measurements.
It predates the B-tree search refactor; rerunning measures the current source.
Historical absolute paths have been made project-relative for sharing. Measured
values and recorded source hashes are unchanged. Historical dataset paths and
source hashes describe the original run, not files required to load its results.

Generated inputs, new runs, plot exports, environments, and caches are ignored
by Git. The supplied results and the notebook's saved outputs are included.
