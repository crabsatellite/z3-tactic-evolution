"""McNemar's test: compare Opt-NIA_final vs z3_default on test set."""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))

from evaluator import evaluate_single_instance
from utils import get_project_root


def evaluate_instance_pair(args):
    """Evaluate one instance with both strategies, return (file, solved_A, solved_B)."""
    smt2_path, tactic_A, tactic_B, timeout_ms = args
    res_A = evaluate_single_instance(smt2_path, tactic_A, timeout_ms)
    res_B = evaluate_single_instance(smt2_path, tactic_B, timeout_ms)
    solved_A = res_A.status in ("sat", "unsat")
    solved_B = res_B.status in ("sat", "unsat")
    return (smt2_path, solved_A, solved_B)


def main():
    root = get_project_root()

    # Load test files
    split_path = os.path.join(root, "data", "benchmarks", "QF_NIA", "split_three_tier.json")
    with open(split_path) as f:
        split = json.load(f)
    test_files = split["test_files"]
    print(f"Test files: {len(test_files)}")

    # Define the two strategies
    tactic_A = {  # Opt-NIA_final
        "type": "Then",
        "children": [
            {"type": "With", "tactic": "simplify", "params": {"som": True, "pull_cheap_ite": True}},
            {"type": "tactic", "name": "propagate-values"},
            {"type": "tactic", "name": "ctx-simplify"},
            {"type": "tactic", "name": "solve-eqs"},
            {"type": "tactic", "name": "elim-uncnstr"},
            {"type": "tactic", "name": "propagate-ineqs"},
            {"type": "OrElse", "children": [
                {"type": "TryFor", "timeout": 7000, "child": {"type": "With", "tactic": "smt", "params": {"random_seed": 7}}},
                {"type": "TryFor", "timeout": 2000, "child": {"type": "With", "tactic": "smt", "params": {"random_seed": 31415}}},
                {"type": "With", "tactic": "smt", "params": {"random_seed": 271}}
            ]}
        ]
    }
    tactic_B = {"type": "tactic", "name": "smt"}  # z3_default

    timeout_ms = 10000
    workers = 4

    print(f"Running McNemar test: Opt-NIA_final vs z3_default")
    print(f"Workers: {workers}, Timeout: {timeout_ms}ms")
    print(f"Started: {time.strftime('%H:%M:%S')}")
    sys.stdout.flush()

    # Evaluate all instances with both strategies
    tasks = [(f, tactic_A, tactic_B, timeout_ms) for f in test_files]

    # Contingency table counters
    both_solved = 0      # a: both solve
    only_A_solved = 0    # b: only Opt-NIA solves
    only_B_solved = 0    # c: only z3_default solves
    neither_solved = 0   # d: neither solves

    start = time.perf_counter()
    done = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(evaluate_instance_pair, t): t for t in tasks}
        for future in as_completed(futures):
            _, solved_A, solved_B = future.result()
            if solved_A and solved_B:
                both_solved += 1
            elif solved_A and not solved_B:
                only_A_solved += 1
            elif not solved_A and solved_B:
                only_B_solved += 1
            else:
                neither_solved += 1
            done += 1
            if done % 100 == 0:
                print(f"  Progress: {done}/{len(test_files)}", flush=True)

    elapsed = time.perf_counter() - start

    print(f"\n{'='*60}")
    print(f"  McNemar's Test Results ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"\n  Contingency Table:")
    print(f"                    z3_default solved  z3_default failed")
    print(f"  Opt-NIA solved      {both_solved:>4d}              {only_A_solved:>4d}")
    print(f"  Opt-NIA failed      {only_B_solved:>4d}              {neither_solved:>4d}")
    print(f"\n  Opt-NIA total solved: {both_solved + only_A_solved}")
    print(f"  z3_default total solved: {both_solved + only_B_solved}")
    print(f"  Discordant pairs: b={only_A_solved}, c={only_B_solved}")

    # McNemar's test (with continuity correction)
    b = only_A_solved
    c = only_B_solved
    if b + c == 0:
        print("\n  No discordant pairs — cannot compute McNemar's test")
        return

    # Chi-squared with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)

    # Without continuity correction
    chi2_no_cc = (b - c) ** 2 / (b + c)

    # p-value from chi-squared distribution (1 df)
    # Use scipy if available, otherwise approximate
    try:
        from scipy.stats import chi2 as chi2_dist
        p_value = 1 - chi2_dist.cdf(chi2, df=1)
        p_value_no_cc = 1 - chi2_dist.cdf(chi2_no_cc, df=1)
        print(f"\n  McNemar chi2 (with continuity correction): {chi2:.4f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  McNemar chi2 (without correction): {chi2_no_cc:.4f}")
        print(f"  p-value: {p_value_no_cc:.6f}")

        if p_value < 0.001:
            sig = "p < 0.001 ***"
        elif p_value < 0.01:
            sig = "p < 0.01 **"
        elif p_value < 0.05:
            sig = "p < 0.05 *"
        else:
            sig = "not significant"
        print(f"\n  Significance: {sig}")
    except ImportError:
        print(f"\n  McNemar chi2 (with cc): {chi2:.4f}")
        print(f"  McNemar chi2 (no cc): {chi2_no_cc:.4f}")
        print(f"  (Install scipy for p-value computation)")

    # Also compute exact binomial test on discordant pairs
    try:
        from scipy.stats import binomtest
        result = binomtest(b, b + c, 0.5)
        print(f"\n  Exact binomial test (two-sided): p = {result.pvalue:.6f}")
    except ImportError:
        pass

    # Save results
    results = {
        "test": "McNemar",
        "strategy_A": "Opt-NIA_final",
        "strategy_B": "z3_default",
        "n_instances": len(test_files),
        "contingency": {
            "both_solved": both_solved,
            "only_A_solved": only_A_solved,
            "only_B_solved": only_B_solved,
            "neither_solved": neither_solved
        },
        "chi2_cc": chi2,
        "chi2_no_cc": chi2_no_cc
    }
    out_path = os.path.join(root, "data", "results", "QF_NIA", "mcnemar_test.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
