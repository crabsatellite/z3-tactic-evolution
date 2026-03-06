"""Standalone evaluation script for a single generation of tactic candidates."""

import argparse
import json
import os
import sys
import time

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from evaluator import evaluate_generation, save_results
from utils import load_config, get_project_root, load_split


def main():
    parser = argparse.ArgumentParser(description="Evaluate Z3 tactic candidates")
    parser.add_argument("--logic", required=True, help="SMT-LIB logic (e.g., QF_NIA)")
    parser.add_argument("--generation", required=True, type=int, help="Generation number")
    parser.add_argument("--config", default=None, help="Config file path")
    parser.add_argument("--use-test", action="store_true", help="Evaluate on test set (final, run once)")
    parser.add_argument("--use-validation", action="store_true", help="Evaluate on validation-locked set")
    parser.add_argument("--use-train-inner", action="store_true", help="Evaluate on train-inner (1000) instead of full train (1400)")
    parser.add_argument("--timeout", type=int, default=None, help="Override timeout (ms)")
    parser.add_argument("--workers", type=int, default=None, help="Override max workers")
    args = parser.parse_args()

    config = load_config(args.config)
    root = get_project_root()
    logic = args.logic
    gen = args.generation

    # Load candidates
    candidates_path = os.path.join(root, config["paths"]["candidates_dir"], logic, f"gen_{gen:03d}.json")
    if not os.path.exists(candidates_path):
        print(f"ERROR: Candidates file not found: {candidates_path}")
        sys.exit(1)

    with open(candidates_path) as f:
        data = json.load(f)

    candidates = data.get("candidates", {})
    if not candidates:
        print("ERROR: No candidates found in file")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Z3 Tactic Evaluation — {logic} — Generation {gen}")
    print(f"  Candidates: {len(candidates)}")
    print(f"{'='*60}\n")

    # Validate all candidates first
    from tactic_builder import validate_tactic
    valid_candidates = {}
    for name, expr in candidates.items():
        ok, err = validate_tactic(expr)
        if ok:
            valid_candidates[name] = expr
        else:
            print(f"  SKIP {name}: {err}")

    if not valid_candidates:
        print("ERROR: No valid candidates")
        sys.exit(1)

    print(f"  Valid candidates: {len(valid_candidates)}/{len(candidates)}\n")

    # Load benchmark files
    bench_dir = os.path.join(root, config["paths"]["benchmarks_dir"])

    # Try three-tier split first, fall back to original
    three_tier_path = os.path.join(bench_dir, logic, "split_three_tier.json")
    if os.path.exists(three_tier_path) and (args.use_train_inner or args.use_validation):
        with open(three_tier_path) as f:
            split = json.load(f)
    else:
        split = load_split(bench_dir, logic)

    if args.use_test:
        files = split["test_files"]
        split_name = "test"
    elif args.use_validation:
        files = split.get("validation_locked_files", split["test_files"])
        split_name = "validation-locked"
    elif args.use_train_inner:
        files = split.get("train_inner_files", split["train_files"])
        split_name = "train-inner"
    else:
        files = split.get("train_inner_files", split["train_files"]) if os.path.exists(three_tier_path) else split["train_files"]
        split_name = "train-inner" if os.path.exists(three_tier_path) else "train"
    print(f"  Benchmark files: {len(files)} ({split_name})")

    # Evaluate
    timeout = args.timeout or config["evaluation"]["timeout_ms"]
    workers = args.workers or config["evaluation"]["max_workers"]

    print(f"  Timeout: {timeout}ms | Workers: {workers}")
    print(f"  Started: {time.strftime('%H:%M:%S')}\n")

    start = time.perf_counter()
    results = evaluate_generation(
        valid_candidates, files,
        timeout_ms=timeout,
        max_workers=workers,
        par_factor=config["evaluation"]["par_factor"]
    )
    elapsed = time.perf_counter() - start

    # Save results (use separate dirs for validation/test)
    if split_name == "validation-locked":
        results_path = os.path.join(root, config["paths"]["results_dir"], logic, "validation", f"gen_{gen:03d}.json")
    elif split_name == "test":
        results_path = os.path.join(root, config["paths"]["results_dir"], logic, "test", f"gen_{gen:03d}.json")
    else:
        results_path = os.path.join(root, config["paths"]["results_dir"], logic, f"gen_{gen:03d}.json")
    save_results(results, results_path, gen)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Generation {gen} Results ({elapsed:.1f}s)")
    print(f"{'='*60}")
    for i, r in enumerate(results):
        print(f"  {i+1}. {r.name}: solved={r.solved_count}/{r.total_count} "
              f"PAR2={r.par2_score:.0f} avg={r.avg_time_ms:.1f}ms")

    print(f"  Results saved: {results_path}")

    return results_path


if __name__ == "__main__":
    main()
