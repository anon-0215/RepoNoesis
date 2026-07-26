from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "backend"))

from app.m5.contracts import (
    AdaptiveSequence,
    DatasetManifest,
    RepositorySpec,
    Scenario,
)


def main(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    models = {
        "manifest.schema.json": DatasetManifest,
        "repository.schema.json": RepositorySpec,
        "scenario.schema.json": Scenario,
        "sequence.schema.json": AdaptiveSequence,
    }
    for name, model in models.items():
        (output / name).write_text(
            json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output.resolve())
