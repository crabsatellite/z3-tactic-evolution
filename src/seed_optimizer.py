"""Seed selection optimizer: find the most complementary set of K seeds.

Algorithm:
1. Sample N candidate seeds (e.g., 0-49)
2. Evaluate each seed on HARD instances (those that timeout with default)
3. Record which seed solves which instance → binary solve matrix
4. Greedy set cover: pick seeds that maximize marginal solved count
5. Output the best K seeds for use in ParOr or sequential portfolio

Rather than picking arbitrary seeds, this selects COMPLEMENTARY seeds
that maximize coverage of the hard-instance space.
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import z3
from tactic_builder import build_tactic


def evaluate_seed_on_instance(args):
    """Evaluate a single (seed, instance) pair. Returns (seed, file, solved)."""
    seed, smt2_path, preprocess_expr, timeout_ms = args
    try:
        formulas = z3.parse_smt2_file(smt2_path)

        # Build preprocessing + smt with specific seed
        tactic_expr = {
            "type": "Then",
            "children": preprocess_expr + [
                {"type": "With", "tactic": "smt", "params": {"random_seed": seed}}
            ]
        }
        tactic = build_tactic(tactic_expr)
        solver = tactic.solver()
        solver.set("timeout", timeout_ms)

        for f in formulas:
            solver.add(f)

        start = time.perf_counter()
        result = solver.check()
        elapsed_ms = (time.perf_counter() - start) * 1000

        solved = result == z3.sat or result == z3.unsat
        return seed, smt2_path, solved, elapsed_ms

    except Exception:
        return seed, smt2_path, False, timeout_ms


def identify_hard_instances(results_path: str, threshold: int = 3) -> list[str]:
    """Find instances that are hard (solved by <= threshold strategies).

    Uses existing generation results to identify hard instances.
    """
    with open(results_path) as f:
        data = json.load(f)

    # Count how many candidates solve each instance
    instance_solve_count = {}
    for candidate in data.get("candidates", []):
        for inst in candidate.get("instances", []):
            fpath = inst["file"]
            if fpath not in instance_solve_count:
                instance_solve_count[fpath] = 0
            if inst["status"] in ("sat", "unsat"):
                instance_solve_count[fpath] += 1

    # Return instances solved by few or no strategies
    hard = [f for f, count in instance_solve_count.items() if count <= threshold]
    print(f"  Found {len(hard)} hard instances (solved by <= {threshold} strategies)")
    return hard


def identify_timeout_instances(results_path: str) -> list[str]:
    """Find instances that timeout with the best strategy (pure hard core)."""
    with open(results_path) as f:
        data = json.load(f)

    # Use the best candidate (rank 1)
    best = data["candidates"][0]
    timeouts = [
        inst["file"] for inst in best["instances"]
        if inst["status"] in ("timeout", "unknown")
    ]
    print(f"  Found {len(timeouts)} timeout instances from best strategy")
    return timeouts


def sample_seeds(n_seeds: int = 50, seed_range: tuple = (0, 1000000)) -> list[int]:
    """Generate a diverse set of candidate seeds."""
    import random
    random.seed(42)  # Reproducible

    # Include some common seeds + random ones
    seeds = [0, 1, 7, 13, 42, 100, 137, 271, 314, 1000, 2023, 31415, 314159]
    remaining = n_seeds - len(seeds)
    if remaining > 0:
        candidates = set(range(seed_range[0], seed_range[1])) - set(seeds)
        seeds.extend(random.sample(sorted(candidates), remaining))

    return sorted(seeds[:n_seeds])


def evaluate_seeds_on_instances(
    seeds: list[int],
    instances: list[str],
    preprocess_expr: list[dict],
    timeout_ms: int = 10000,
    max_workers: int = 10,
) -> dict[int, set[str]]:
    """Evaluate all seeds on all instances. Returns seed -> set of solved files."""
    total_tasks = len(seeds) * len(instances)
    print(f"\n  Evaluating {len(seeds)} seeds × {len(instances)} instances = {total_tasks} tasks", flush=True)
    print(f"  Workers: {max_workers}, Timeout: {timeout_ms}ms", flush=True)
    est_seconds = total_tasks * timeout_ms / 1000 / max_workers * 0.5  # ~50% timeout
    print(f"  Estimated time: {est_seconds/60:.0f}-{est_seconds*2/60:.0f} minutes", flush=True)

    tasks = [
        (seed, inst, preprocess_expr, timeout_ms)
        for seed in seeds
        for inst in instances
    ]

    solve_matrix = {seed: set() for seed in seeds}
    completed = 0
    start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_seed_on_instance, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                seed, fpath, solved, elapsed = future.result(timeout=timeout_ms / 1000 + 30)
                if solved:
                    solve_matrix[seed].add(fpath)
            except Exception:
                pass

            completed += 1
            if completed % (total_tasks // 20 + 1) == 0:
                elapsed_t = time.perf_counter() - start
                pct = completed / total_tasks * 100
                print(f"    Progress: {completed}/{total_tasks} ({pct:.0f}%) — {elapsed_t:.0f}s elapsed", flush=True)

    elapsed_total = time.perf_counter() - start
    print(f"  Completed in {elapsed_total:.0f}s")

    # Report per-seed solve counts
    for seed in seeds:
        n = len(solve_matrix[seed])
        if n > 0:
            print(f"    Seed {seed:>6d}: solves {n}/{len(instances)} hard instances")

    return solve_matrix


def greedy_set_cover(solve_matrix: dict[int, set[str]], k: int) -> list[tuple[int, int]]:
    """Greedy set cover: pick k seeds that maximize total coverage.

    Returns list of (seed, marginal_gain) in selection order.
    """
    selected = []
    covered = set()
    remaining_seeds = set(solve_matrix.keys())

    for i in range(k):
        best_seed = None
        best_gain = -1

        for seed in remaining_seeds:
            gain = len(solve_matrix[seed] - covered)
            if gain > best_gain:
                best_gain = gain
                best_seed = seed

        if best_seed is None or best_gain == 0:
            print(f"  No more seeds with marginal gain at step {i+1}")
            break

        selected.append((best_seed, best_gain))
        covered |= solve_matrix[best_seed]
        remaining_seeds.remove(best_seed)
        print(f"  Step {i+1}: seed={best_seed}, marginal_gain={best_gain}, "
              f"total_covered={len(covered)}")

    print(f"\n  Selected {len(selected)} seeds, total coverage: {len(covered)} instances")
    return selected


def run_seed_optimization(
    logic: str = "QF_NIA",
    n_candidate_seeds: int = 50,
    k_select: int = 6,
    use_hard_threshold: int = 1,
    timeout_ms: int = 10000,
    max_workers: int = 10,
    config_path: str = None,
):
    """Full seed optimization pipeline."""
    from utils import load_config, get_project_root

    config = load_config(config_path)
    root = get_project_root()

    print(f"\n{'='*60}")
    print(f"  Seed Selection Optimizer — {logic}")
    print(f"  Candidate seeds: {n_candidate_seeds}, Select: {k_select}")
    print(f"{'='*60}")

    # Find the latest results file
    results_dir = os.path.join(root, config["paths"]["results_dir"], logic)
    result_files = sorted(Path(results_dir).glob("gen_*.json"))
    if not result_files:
        print("ERROR: No results files found. Run at least one generation first.")
        sys.exit(1)

    latest_results = str(result_files[-1])
    print(f"\n  Using results from: {latest_results}")

    # Identify hard instances
    hard_instances = identify_hard_instances(latest_results, threshold=use_hard_threshold)
    if len(hard_instances) < 10:
        # Fallback: use timeout instances from best strategy
        hard_instances = identify_timeout_instances(latest_results)

    if not hard_instances:
        print("ERROR: No hard instances found")
        sys.exit(1)

    # Cap at reasonable number for efficiency
    max_hard = 200
    if len(hard_instances) > max_hard:
        import random
        random.seed(42)
        hard_instances = random.sample(hard_instances, max_hard)
        print(f"  Sampled {max_hard} hard instances for efficiency")

    # Preprocessing chain (from our best known strategy)
    preprocess = [
        {"type": "With", "tactic": "simplify", "params": {"som": True, "pull_cheap_ite": True}},
        {"type": "tactic", "name": "propagate-values"},
        {"type": "tactic", "name": "ctx-simplify"},
        {"type": "tactic", "name": "solve-eqs"},
        {"type": "tactic", "name": "elim-uncnstr"},
        {"type": "tactic", "name": "propagate-ineqs"},
    ]

    # Sample seeds
    seeds = sample_seeds(n_candidate_seeds)
    print(f"\n  Candidate seeds: {seeds}")

    # Evaluate all seeds on hard instances
    solve_matrix = evaluate_seeds_on_instances(
        seeds, hard_instances, preprocess, timeout_ms, max_workers
    )

    # Greedy set cover
    print(f"\n  Running greedy set cover (k={k_select})...")
    selected = greedy_set_cover(solve_matrix, k_select)

    # Save results
    output = {
        "logic": logic,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_candidate_seeds": n_candidate_seeds,
        "k_selected": k_select,
        "n_hard_instances": len(hard_instances),
        "timeout_ms": timeout_ms,
        "candidate_seeds": seeds,
        "seed_solve_counts": {str(s): len(solve_matrix[s]) for s in seeds},
        "selected_seeds": [{"seed": s, "marginal_gain": g} for s, g in selected],
        "total_coverage": sum(g for _, g in selected),
    }

    output_path = os.path.join(root, "data", "seed_optimization", logic, "seed_selection.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to: {output_path}")
    print(f"\n  OPTIMAL SEEDS: {[s for s, _ in selected]}")

    return [s for s, _ in selected]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimize seed selection for parallel portfolio")
    parser.add_argument("--logic", default="QF_NIA")
    parser.add_argument("--n-seeds", type=int, default=50, help="Number of candidate seeds to test")
    parser.add_argument("--k-select", type=int, default=6, help="Number of seeds to select")
    parser.add_argument("--hard-threshold", type=int, default=1,
                        help="Instance is 'hard' if solved by <= this many strategies")
    parser.add_argument("--timeout", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    run_seed_optimization(
        logic=args.logic,
        n_candidate_seeds=args.n_seeds,
        k_select=args.k_select,
        use_hard_threshold=args.hard_threshold,
        timeout_ms=args.timeout,
        max_workers=args.workers,
    )
