"""Evaluate strategies on AProVE QF_NIA benchmark (external validation)."""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from evaluator import evaluate_generation, save_results
from tactic_builder import validate_tactic


def find_aprove_files(base_dir):
    """Find all .smt2 files under the AProVE subdirectory."""
    aprove_dir = os.path.join(base_dir, "non-incremental", "QF_NIA", "AProVE")
    if not os.path.exists(aprove_dir):
        # Try flat structure
        aprove_dir = os.path.join(base_dir, "AProVE")
    if not os.path.exists(aprove_dir):
        print(f"ERROR: AProVE directory not found at {aprove_dir}")
        sys.exit(1)

    files = []
    for root, dirs, fnames in os.walk(aprove_dir):
        for f in fnames:
            if f.endswith(".smt2"):
                files.append(os.path.join(root, f))
    files.sort()
    return files


def main():
    parser = argparse.ArgumentParser(description="Evaluate on AProVE QF_NIA benchmark")
    parser.add_argument("--benchmark-dir", required=True, help="Path to extracted QF_NIA benchmark")
    parser.add_argument("--candidates", required=True, help="Path to candidates JSON file")
    parser.add_argument("--timeout", type=int, default=10000, help="Timeout in ms (default 10000)")
    parser.add_argument("--workers", type=int, default=8, help="Max parallel workers")
    parser.add_argument("--output", default=None, help="Output results JSON path")
    args = parser.parse_args()

    # Load candidates
    with open(args.candidates) as f:
        data = json.load(f)
    candidates = data.get("candidates", {})

    # Validate candidates
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

    # Find AProVE files
    files = find_aprove_files(args.benchmark_dir)
    print(f"\n{'='*60}")
    print(f"  AProVE QF_NIA External Validation")
    print(f"  Candidates: {len(valid_candidates)}")
    print(f"  Benchmark files: {len(files)}")
    print(f"  Timeout: {args.timeout}ms | Workers: {args.workers}")
    print(f"  Started: {time.strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")

    start = time.perf_counter()
    results = evaluate_generation(
        valid_candidates, files,
        timeout_ms=args.timeout,
        max_workers=args.workers,
        par_factor=2
    )
    elapsed = time.perf_counter() - start

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        save_results(results, args.output, 0)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  AProVE QF_NIA Results ({elapsed:.1f}s)")
    print(f"{'='*60}")
    for i, r in enumerate(results):
        print(f"  {i+1}. {r.name}: solved={r.solved_count}/{r.total_count} "
              f"({r.solved_count/r.total_count*100:.1f}%) PAR2={r.par2_score:.0f} "
              f"avg={r.avg_time_ms:.1f}ms")

    print(f"\n  Total wall time: {elapsed:.1f}s")
    if args.output:
        print(f"  Results saved: {args.output}")


if __name__ == "__main__":
    main()
