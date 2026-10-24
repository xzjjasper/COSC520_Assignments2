"""Internal fresh-process entry point used by the notebook runner."""
import _bootstrap  # noqa: F401
import json
from pathlib import Path
import sys

from cosc520_assignment2.benchmark import StructureConfig, run_job

if __name__ == "__main__":
    directory, name, destination, config_json = sys.argv[1:]
    config = StructureConfig(**json.loads(config_json))
    rows = run_job(Path(directory), name, config)
    output = Path(destination)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
