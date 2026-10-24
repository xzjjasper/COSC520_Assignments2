"""Notebook-friendly preparation and execution using the existing serial runner."""

import json
from pathlib import Path

from .benchmark import StructureConfig
from .datasets import DatasetSpec, generate_dataset, read_spec, save_dataset
from .registry import STRUCTURES
from .runner import run_collection


def run_experiments(*, sizes, queries, repeats, seed, orders, data_dir: Path,
                    output_dir: Path, config: StructureConfig, progress=print):
    """Generate/reuse exact workloads, then run each structure in a fresh process.

    Existing dataset parameters must match exactly; existing results are never
    overwritten. Workers verify input hashes. Generation and validation are
    outside the operation timers. The notebook can replace progress with a
    display-handle update to keep cell output bounded.
    """
    if repeats < 1 or not sizes or not orders or len(set(sizes)) != len(sizes) or len(set(orders)) != len(orders):
        raise ValueError("require positive repeats and distinct sizes/orders")
    specs = [DatasetSpec(n, queries, seed, repeat, order)
             for n in sizes for repeat in range(repeats) for order in orders]
    config.validate()
    for spec in specs:
        spec.validate()
    if output_dir.exists():
        raise FileExistsError(f"choose a new result directory: {output_dir}")
    directories = []
    for spec in specs:
        directory = data_dir / spec.relative_path()
        if directory.exists():
            if read_spec(directory) != spec:
                raise ValueError(f"existing workload has different parameters: {directory}")
        else:
            progress(f"Generating n={spec.size:,}, repeat={spec.repeat + 1}, {spec.insertion_order}")
            save_dataset(generate_dataset(spec), directory)
        directories.append(directory)
    rows = run_collection(directories, STRUCTURES, config, output_dir, progress=progress)
    parameters = {"sizes": list(sizes), "queries": queries, "repeats": repeats,
                  "seed": seed, "orders": list(orders), "data_dir": str(data_dir.resolve())}
    (output_dir / "experiment.json").write_text(json.dumps(parameters, indent=2) + "\n")
    return rows
