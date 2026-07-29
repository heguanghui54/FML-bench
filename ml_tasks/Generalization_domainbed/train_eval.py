"""
Wrapper script for DomainBed: runs training then extracts val or test metrics.

Val/test split strategy for DomainBed (ColoredMNIST with test_env=2):
  - val:  Out-split accuracy on held-out test domain (env2_out, 30%)
  - test: In-split accuracy on held-out test domain (env2_in, 70%)

Both val and test measure generalization to the same unseen domain (env2).
The holdout_fraction=0.3 splits each domain into 30% out-split and 70% in-split.
"""
import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
import time


def _write_teardown_record(output_dir, *, mode, child_returncode, grace_seconds):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "teardown_info.json")
    with open(path, "w") as f:
        json.dump({
            "schema_version": "domainbed-training-teardown-v1",
            "mode": mode,
            "child_returncode": child_returncode,
            "grace_seconds": grace_seconds,
            "required_artifacts_complete": all(os.path.isfile(os.path.join(output_dir, name)) for name in (
                "done", "results.jsonl", "model.pkl"
            )),
        }, f, indent=2)


def wait_for_training_process(process, output_dir, grace_seconds=10.0):
    """Accept a controlled teardown only after every training artifact exists."""
    required = [os.path.join(output_dir, name) for name in ("done", "results.jsonl", "model.pkl")]
    while True:
        returncode = process.poll()
        if returncode is not None:
            _write_teardown_record(
                output_dir,
                mode="NATURAL_EXIT",
                child_returncode=returncode,
                grace_seconds=grace_seconds,
            )
            return returncode
        if all(os.path.isfile(path) for path in required):
            try:
                returncode = process.wait(timeout=grace_seconds)
                mode = "NATURAL_EXIT_AFTER_DONE"
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                returncode = 0
                mode = "CONTROLLED_TERMINATION_AFTER_COMPLETE_ARTIFACTS"
            _write_teardown_record(
                output_dir,
                mode=mode,
                child_returncode=returncode,
                grace_seconds=grace_seconds,
            )
            return returncode
        time.sleep(0.25)


def run_training(output_dir="./results_tmp"):
    """Run DomainBed training. Returns the process exit code."""
    cmd = [
        "python", "-m", "domainbed.scripts.train",
        "--data_dir=../data/",
        "--algorithm", "ERM",
        "--dataset", "ColoredMNIST",
        "--test_env", "2",
        "--holdout_fraction", "0.3",
        "--output_dir", output_dir,
    ]
    print(f"Running: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, start_new_session=True)
    try:
        return wait_for_training_process(process, output_dir)
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        raise


def extract_metrics(output_dir, split):
    """Extract metrics from results.jsonl based on split type."""
    results_file = os.path.join(output_dir, "results.jsonl")

    with open(results_file, 'r') as f:
        lines = f.readlines()

    # Parse the last result line
    last_line = lines[-1].strip()
    if not last_line:
        last_line = lines[-2].strip()
    last_result = json.loads(last_line)

    if split == 'val':
        # Val: out-split accuracy on held-out test domain (env2, 30%)
        accuracy = last_result.get('env2_out_acc', 0.0)
        description = f"Val (held-out domain env2 out_split 30%): {accuracy:.4f}"
    else:
        # Test: in-split accuracy on held-out test domain (env2, 70%)
        accuracy = last_result.get('env2_in_acc', 0.0)
        description = f"Test (held-out domain env2 in_split 70%): {accuracy:.4f}"

    print(f"\n{description}")
    print(f"in_acc_mean = {accuracy:.6f}")

    return {
        "ColoredMNIST_test_env2": {
            "means": {
                "in_acc_mean": accuracy,
            },
            "stderrs": {
                "in_acc_stderr": 0.0,
            },
            "final_info_dict": {
                "in_acc": [accuracy],
            }
        }
    }


def main():
    parser = argparse.ArgumentParser(description='DomainBed train + eval with val/test split')
    parser.add_argument('--split', choices=['val', 'test'], required=True,
                        help='Which evaluation split to report: val or test')
    args = parser.parse_args()

    output_dir = "./results_tmp"
    checkpoint_dir = "./model_checkpoint"

    if args.split == 'val':
        # Run training (expensive)
        returncode = run_training(output_dir)
        if returncode != 0:
            print(f"Training failed with return code {returncode}", file=sys.stderr)
            sys.exit(returncode)

        # Save results.jsonl for test reuse
        os.makedirs(checkpoint_dir, exist_ok=True)
        shutil.copy2(os.path.join(output_dir, "results.jsonl"),
                      os.path.join(checkpoint_dir, "results.jsonl"))
        print(f"Saved results.jsonl to {checkpoint_dir}/ for test reuse")
    else:
        # Load saved results.jsonl (no training)
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy2(os.path.join(checkpoint_dir, "results.jsonl"),
                      os.path.join(output_dir, "results.jsonl"))
        print(f"Loaded results.jsonl from {checkpoint_dir}/ (skipping training)")

    # Extract metrics for the requested split
    results = extract_metrics(output_dir, args.split)

    # Save results
    output_path = os.path.join(output_dir, f"{args.split}_info.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
