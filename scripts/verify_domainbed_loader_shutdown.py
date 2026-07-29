#!/usr/bin/env python3
"""Exercise DomainBed's explicit DataLoader worker shutdown on a real host."""

import argparse
import importlib.util
import json
import multiprocessing
from pathlib import Path
import time

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("loader_path", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("shutdown_probe_loader", args.loader_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dataset = torch.utils.data.TensorDataset(
        torch.arange(128, dtype=torch.float32).reshape(64, 2),
        torch.arange(64),
    )
    loader = module.InfiniteDataLoader(
        dataset=dataset,
        weights=None,
        batch_size=8,
        num_workers=args.workers,
    )
    next(iter(loader))
    workers_before = len(multiprocessing.active_children())
    loader.close()

    deadline = time.time() + 10
    while multiprocessing.active_children() and time.time() < deadline:
        time.sleep(0.1)
    workers_after = len(multiprocessing.active_children())
    result = {
        "workers_requested": args.workers,
        "workers_before_close": workers_before,
        "workers_after_close": workers_after,
        "passed": workers_before > 0 and workers_after == 0,
    }
    print(json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
