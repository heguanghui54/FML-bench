"""Deterministically merge trajectory JSONL buffers by episode identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def merge_episode_files(inputs: list[Path], out_path: Path) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    for path in inputs:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            episode_id = str(row.get("episode_id") or "")
            if not episode_id:
                raise ValueError(f"Missing episode_id in {path}:{line_number}")
            if episode_id in rows:
                if rows[episode_id] != row:
                    raise ValueError(f"Conflicting duplicate episode_id: {episode_id}")
                duplicate_count += 1
                continue
            rows[episode_id] = row
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [rows[key] for key in sorted(rows)]
    out_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ordered),
        encoding="utf-8",
    )
    return {
        "schema_version": "ml-scientist-episode-merge-v1",
        "inputs": [str(path) for path in inputs],
        "out": str(out_path),
        "episode_count": len(ordered),
        "duplicate_count": duplicate_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge_episode_files(args.input, args.out), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
