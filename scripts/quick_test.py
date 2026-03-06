"""Quick sanity test: run each strategy once on each dataset.

Runs Z3 and Opt-NIA (the two strategies compared in the paper) once on each
benchmark set. Takes ~30 minutes total vs ~5 hours for the full multi-run eval.

Usage:
    python scripts/quick_test.py [--workers 4] [--skip-external]

    # Run a single dataset + strategy combination:
    python scripts/quick_test.py --dataset Test --strategy Z3
    python scripts/quick_test.py --dataset Validation --strategy Opt-NIA
    python scripts/quick_test.py --dataset AProVE --strategy Z3
    python scripts/quick_test.py --dataset MathProblems --strategy Opt-NIA
    python scripts/quick_test.py --dataset Dartagnan --strategy Z3

    # Run one dataset with both strategies:
    python scripts/quick_test.py --dataset Test
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from evaluator import evaluate_candidate
from multirun_eval import STRATEGIES, STRATEGY_ORDER, load_datasets

ALL_DATASETS = ["Test", "Validation", "AProVE", "MathProblems", "Dartagnan"]
ALL_STRATEGIES = ["Z3", "Opt-NIA"]


def load_all_datasets(skip_external=False):
    """Load all datasets including external ones."""
    datasets = load_datasets(PROJECT_ROOT, skip_aprove=skip_external)
    if not skip_external:
        from eval_external import load_filelist, DATASET_CONFIG
        for ds_name in DATASET_CONFIG:
            files = load_filelist(PROJECT_ROOT, ds_name)
            datasets[ds_name] = {"files": files, "n_runs": 1}
    return datasets


def run_single(ds_name, files, strategy_name, workers):
    """Run a single dataset + strategy and return the result."""
    t0 = time.perf_counter()
    result = evaluate_candidate(
        name=strategy_name,
        tactic_expr=STRATEGIES[strategy_name],
        benchmark_files=files,
        timeout_ms=10000,
        max_workers=workers,
        par_factor=2,
    )
    elapsed = time.perf_counter() - t0
    pct = result.solved_count / result.total_count * 100
    print(f"  {strategy_name:>10}: {result.solved_count}/{result.total_count} "
          f"({pct:.1f}%) [{elapsed:.0f}s]")
    return result


def main():
    parser = argparse.ArgumentParser(description="Quick single-run sanity test")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-external", action="store_true",
                        help="Only run Test + Validation (skip AProVE/MathProblems/Dartagnan)")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=ALL_DATASETS,
                        help="Run only this dataset (default: all)")
    parser.add_argument("--strategy", type=str, default=None,
                        choices=ALL_STRATEGIES,
                        help="Run only this strategy (default: both Z3 and Opt-NIA)")
    args = parser.parse_args()

    # Force UTF-8 on Windows
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    # Determine if we need external datasets
    need_external = not args.skip_external
    if args.dataset and args.dataset in ["Test", "Validation"]:
        need_external = False  # no need to load external for core datasets
    if args.dataset and args.dataset in ["MathProblems", "Dartagnan", "AProVE"]:
        need_external = True

    datasets = load_all_datasets(skip_external=not need_external)

    # Filter to requested dataset
    if args.dataset:
        if args.dataset not in datasets:
            print(f"ERROR: Dataset '{args.dataset}' not available. "
                  f"Available: {list(datasets.keys())}")
            return 1
        datasets = {args.dataset: datasets[args.dataset]}

    # Filter to requested strategies
    strategies_to_test = [args.strategy] if args.strategy else ALL_STRATEGIES

    print(f"\n{'='*60}")
    print(f"  Quick Sanity Test")
    print(f"  Strategies: {', '.join(strategies_to_test)}")
    print(f"  Datasets: {', '.join(datasets.keys())}")
    print(f"  Workers: {args.workers}")
    print(f"{'='*60}\n")

    all_pass = True
    results_table = []

    for ds_name, ds in datasets.items():
        files = ds["files"]
        print(f"--- {ds_name} ({len(files)} files) ---")

        ds_results = {}
        for sn in strategies_to_test:
            ds_results[sn] = run_single(ds_name, files, sn, args.workers)

        # Check delta only if both strategies were run
        if "Z3" in ds_results and "Opt-NIA" in ds_results:
            opt = ds_results["Opt-NIA"].solved_count
            z3 = ds_results["Z3"].solved_count
            delta = opt - z3
            status = "PASS" if delta > 0 else ("TIE" if delta == 0 else "FAIL")
            if delta <= 0:
                all_pass = False
            results_table.append((ds_name, len(files), z3, opt, delta, status))
            print(f"  Delta: Opt-NIA - Z3 = {delta:+d} [{status}]\n")
        else:
            # Single strategy mode — just report
            sn = strategies_to_test[0]
            r = ds_results[sn]
            results_table.append((ds_name, len(files), r.solved_count if sn == "Z3" else None,
                                  r.solved_count if sn == "Opt-NIA" else None, None, "—"))
            print()

    # Summary
    print(f"{'='*60}")
    if any(delta is not None for _, _, _, _, delta, _ in results_table):
        print(f"  {'Dataset':<16} {'N':>5} {'Z3':>5} {'Opt-NIA':>8} {'Delta':>6} {'Status'}")
        print(f"  {'-'*16} {'-'*5} {'-'*5} {'-'*8} {'-'*6} {'-'*6}")
        for ds_name, n, z3, opt, delta, status in results_table:
            z3_s = f"{z3:>5}" if z3 is not None else "    —"
            opt_s = f"{opt:>8}" if opt is not None else "       —"
            delta_s = f"{delta:>+6}" if delta is not None else "     —"
            print(f"  {ds_name:<16} {n:>5} {z3_s} {opt_s} {delta_s} {status}")
    else:
        sn = strategies_to_test[0]
        print(f"  {'Dataset':<16} {'N':>5} {sn:>8}")
        print(f"  {'-'*16} {'-'*5} {'-'*8}")
        for ds_name, n, z3, opt, delta, status in results_table:
            val = z3 if z3 is not None else opt
            print(f"  {ds_name:<16} {n:>5} {val:>8}")

    print(f"{'='*60}")
    if all(delta is not None for _, _, _, _, delta, _ in results_table):
        print(f"\n  OVERALL: {'ALL PASS' if all_pass else 'SOME CHECKS FAILED'}")
    print(f"  Note: single-run results may vary by +/-3 instances due to solver non-determinism.")
    print()

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
