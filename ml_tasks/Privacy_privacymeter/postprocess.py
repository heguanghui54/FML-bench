"""
Postprocess MIA results with val/test split and constraint checks.

Splits target model results into val (30%) / test (70%):
- Shuffled with fixed seed=42 for unbiased assignment
- val: 30% of target models
- test: 70% of target models
"""
import re
import json
import argparse
import numpy as np
import os


CONSTRAINT_CONTRACT_VERSION = "privacy-privacymeter-confirmatory-v2"
BASELINE_TEST_ACCURACY = 0.61393
ACCURACY_TOLERANCE = 0.01
MIN_TEST_ACCURACY = BASELINE_TEST_ACCURACY - ACCURACY_TOLERANCE
MAX_TPR_AT_0_1_FPR = 0.02
MAX_TPR_AT_0_0_FPR = 0.005


def main(exp_dir: str, split: str):
    log_file = os.path.join(exp_dir, "report", "log_time_analysis.log")

    auc_list = []
    tpr01_list = []
    tpr0_list = []
    test_acc_list = []

    auc_pattern = re.compile(
        r"Target Model \d+: AUC ([0-9.]+), TPR@0.1%FPR of ([0-9.]+), TPR@0.0%FPR of ([0-9.]+)"
    )
    test_acc_pattern = re.compile(r"Test accuracy (\d+\.\d+)")

    with open(log_file, "r") as f:
        for line in f:
            auc_match = auc_pattern.search(line)
            if auc_match:
                auc_list.append(float(auc_match.group(1)))
                tpr01_list.append(float(auc_match.group(2)))
                tpr0_list.append(float(auc_match.group(3)))
            acc_match = test_acc_pattern.search(line)
            if acc_match:
                test_acc_list.append(float(acc_match.group(1)))

    if not auc_list:
        raise ValueError("No AUC values found in log file.")
    if not test_acc_list:
        raise ValueError("No test accuracy values found in log file.")

    # Split by target model index (shuffled for unbiased 30/70 split)
    all_indices = list(range(len(auc_list)))
    rng = np.random.RandomState(42)
    rng.shuffle(all_indices)
    split_point = int(0.3 * len(auc_list))
    if split_point < 1:
        split_point = 1
    if split == 'val':
        indices = sorted(all_indices[:split_point])
    else:
        indices = sorted(all_indices[split_point:])

    if len(auc_list) <= 1:
        indices = list(range(len(auc_list)))

    auc_split = [auc_list[i] for i in indices]
    tpr01_split = [tpr01_list[i] for i in indices]
    tpr0_split = [tpr0_list[i] for i in indices]
    # Use all test_acc values (shared across splits since it's model accuracy)
    test_acc_split = test_acc_list

    auc_mean = sum(auc_split) / len(auc_split) if auc_split else 0.0
    tpr01_mean = sum(tpr01_split) / len(tpr01_split) if tpr01_split else 0.0
    tpr0_mean = sum(tpr0_split) / len(tpr0_split) if tpr0_split else 0.0
    test_acc_mean = sum(test_acc_split) / len(test_acc_split) if test_acc_split else 0.0

    auc_gap = abs(auc_mean - 0.5)
    auc_gap_list = [abs(auc - 0.5) for auc in auc_split]

    # One source-grounded constraint contract applies to both validation and
    # protected test.  Earlier pilot code used different thresholds by split;
    # confirmatory-v2 removes that selection/evaluation mismatch.
    violations = []

    # Save original metrics before any replacement
    original_metrics = {
        "AUC_gap": auc_gap,
        "AUC_gap_list": auc_gap_list,
        "TPR@0.1%FPR": tpr01_mean,
        "TPR@0.0%FPR": tpr0_mean,
        "test_acc": test_acc_mean,
    }

    if tpr01_mean > MAX_TPR_AT_0_1_FPR or tpr0_mean > MAX_TPR_AT_0_0_FPR:
        violations.append(
            "TPR constraint violated: "
            f"TPR@0.1%={tpr01_mean} > {MAX_TPR_AT_0_1_FPR} or "
            f"TPR@0.0%={tpr0_mean} > {MAX_TPR_AT_0_0_FPR}"
        )
    if test_acc_mean < MIN_TEST_ACCURACY:
        violations.append(
            f"Accuracy constraint violated: {test_acc_mean} < {MIN_TEST_ACCURACY}"
        )

    if violations and split == 'val':
        auc_gap = None
    elif violations:
        # Preserve FML's constraint-adjusted test credit while keeping the raw
        # candidate measurements in original_metrics below.
        auc_gap = 0.3213666666666667
        auc_gap_list = [0.32030000000000003, 0.31799999999999995, 0.3258]

    constraint_violated = bool(violations)

    final_info = {
        "constraint_contract": {
            "version": CONSTRAINT_CONTRACT_VERSION,
            "baseline_test_accuracy": BASELINE_TEST_ACCURACY,
            "accuracy_tolerance": ACCURACY_TOLERANCE,
            "minimum_test_accuracy": MIN_TEST_ACCURACY,
            "maximum_tpr_at_0_1_fpr": MAX_TPR_AT_0_1_FPR,
            "maximum_tpr_at_0_0_fpr": MAX_TPR_AT_0_0_FPR,
            "same_thresholds_for_validation_and_test": True,
        },
        "cifar10": {
            "means": {
                "AUC_gap_mean": auc_gap,
                "TPR@0.1%FPR_mean": tpr01_mean,
                "TPR@0.0%FPR_mean": tpr0_mean,
                "test_acc_mean": test_acc_mean,
                "constraint_violated": constraint_violated,
            },
            "stderrs": {
                "AUC_gap_stderr": 0.0,
                "TPR@0.1%FPR_stderr": 0.0,
                "TPR@0.0%FPR_stderr": 0.0,
                "test_acc_stderr": 0.0,
            },
            "final_info_dict": {
                "AUC_gap": auc_gap_list,
                "TPR@0.1%FPR": tpr01_split,
                "TPR@0.0%FPR": tpr0_split,
                "test_acc": test_acc_split,
                "constraint_violated": constraint_violated,
                "constraint_violations": violations,
            },
        }
    }

    # Preserve the uncredited measurements on either split so failure remains a
    # scientific outcome rather than an opaque crash or a metric-free fallback.
    if constraint_violated:
        final_info["cifar10"]["original_metrics"] = original_metrics

    output_file = os.path.join(exp_dir, f"{split}_info.json")
    with open(output_file, "w") as f:
        json.dump(final_info, f, indent=2)

    if split == 'val' and violations:
        violation_reason = " AND ".join(violations)
        print(f"[WARNING] Invalid results: {violation_reason}")
        raise ValueError(f"Invalid results: {violation_reason}")
    elif constraint_violated:
        violation_reason = " AND ".join(violations)
        print(f"[WARNING] Test constraint violated: {violation_reason}. "
              f"Reporting baseline AUC_gap. Original preserved in 'original_metrics'.")
    print(f"Saved {split} results to {output_file}")
    print(f"AUC_gap_mean: {auc_gap}, test_acc_mean: {test_acc_mean}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    args = parser.parse_args()
    main(args.exp_dir, args.split)
