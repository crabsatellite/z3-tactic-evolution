"""Evaluate frozen strategy on external benchmark families.

Runs the same 4 strategies (Z3, PP-base, PP-ext, Opt-NIA) with 3-block
interleaved design on unseen instances from specified benchmark families.

Usage:
    python src/eval_external.py --dataset MathProblems [--workers 4] [--resume]
    python src/eval_external.py --dataset Dartagnan [--workers 4] [--resume]
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from evaluator import evaluate_candidate

# Reuse strategy definitions from multirun_eval
from multirun_eval import (
    STRATEGIES,
    STRATEGY_ORDER,
    make_block_schedule,
    compute_stats,
    compute_mcnemar,
)

DATASET_CONFIG = {
    "MathProblems": {
        "filelist": "data/benchmarks/QF_NIA_MathProblems/filelist_unseen.json",
        "n_runs": 3,
        "label": "MathProblems (math competition encodings)",
    },
    "Dartagnan": {
        "filelist": "data/benchmarks/QF_NIA_Dartagnan/filelist_unseen.json",
        "n_runs": 3,
        "label": "Dartagnan (memory model verification)",
    },
}


def load_filelist(project_root, dataset_name):
    """Load unseen file list for a dataset."""
    config = DATASET_CONFIG[dataset_name]
    path = os.path.join(project_root, config["filelist"])
    with open(path) as f:
        data = json.load(f)
    # Resolve relative paths to absolute
    files = [os.path.join(project_root, f) if not os.path.isabs(f) else f
             for f in data["files"]]
    print(f"Loaded {len(files)} unseen instances for {dataset_name}")
    print(f"  Family total: {data['total_in_family']}, excluded overlap: {data['excluded_overlap']}")
    return files


def run_external_eval(dataset_name, workers, project_root, resume=False):
    """Run 3-block interleaved evaluation on an external dataset."""
    config = DATASET_CONFIG[dataset_name]
    files = load_filelist(project_root, dataset_name)
    n_runs = config["n_runs"]

    # Write to reproduce/ to avoid overwriting shipped reference results
    output_dir = os.path.join(project_root, "data", "results", "QF_NIA", "reproduce")
    evidence_dir = os.path.join(output_dir, "evidence")
    os.makedirs(evidence_dir, exist_ok=True)

    experiment_start = datetime.now()
    log_lines = []

    def log(msg):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log(f"External Evaluation: {dataset_name} ({config['label']})")
    log(f"Files: {len(files)}, Runs: {n_runs}, Workers: {workers}, Resume: {resume}")

    schedule = make_block_schedule(n_runs, list(STRATEGY_ORDER))
    all_results = defaultdict(list)
    all_instances = defaultdict(dict)
    timeout_ms = 10000
    par_factor = 2
    n_skipped = 0
    n_new = 0

    for block_idx, strategy_order in enumerate(schedule):
        log(f"\n--- Block {block_idx + 1}/{n_runs} | Order: {' -> '.join(strategy_order)} ---")

        for strategy_name in strategy_order:
            evidence_file = os.path.join(
                evidence_dir,
                f"{dataset_name}_block{block_idx+1:02d}_{strategy_name}.json"
            )

            if resume and os.path.exists(evidence_file):
                try:
                    with open(evidence_file) as f:
                        existing = json.load(f)
                    run_data = existing["meta"]
                    all_results[strategy_name].append(run_data)
                    all_instances[block_idx][strategy_name] = existing["instances"]
                    n_skipped += 1
                    log(f"  RESUME: {strategy_name} block {block_idx+1} "
                        f"-> {run_data['solved']}/{run_data['total']}")
                    continue
                except (json.JSONDecodeError, KeyError) as e:
                    log(f"  RESUME: corrupt evidence, re-running: {e}")

            tactic_expr = STRATEGIES[strategy_name]
            run_start = datetime.now()
            log(f"  Running: {strategy_name} (block {block_idx+1}/{n_runs})...")

            result = evaluate_candidate(
                name=strategy_name,
                tactic_expr=tactic_expr,
                benchmark_files=files,
                timeout_ms=timeout_ms,
                max_workers=workers,
                par_factor=par_factor,
            )

            run_end = datetime.now()
            elapsed = (run_end - run_start).total_seconds()
            n_new += 1

            run_data = {
                "strategy": strategy_name,
                "dataset": dataset_name,
                "block": block_idx + 1,
                "n_files": len(files),
                "solved": result.solved_count,
                "total": result.total_count,
                "solve_rate": round(result.solved_count / result.total_count * 100, 2),
                "par2": round(result.par2_score, 1),
                "avg_time_ms": result.avg_time_ms,
                "median_time_ms": result.median_time_ms,
                "sat": result.sat_count,
                "unsat": result.unsat_count,
                "timeout": result.timeout_count,
                "error": result.error_count,
                "start_time": run_start.isoformat(),
                "end_time": run_end.isoformat(),
                "duration_s": round(elapsed, 1),
                "workers": workers,
            }

            all_results[strategy_name].append(run_data)
            all_instances[block_idx][strategy_name] = result.instances

            evidence = {"meta": run_data, "instances": result.instances}
            with open(evidence_file, "w") as f:
                json.dump(evidence, f, indent=1)

            log(f"    -> {strategy_name}: {result.solved_count}/{result.total_count} "
                f"({run_data['solve_rate']}%) PAR2={run_data['par2']:.0f} "
                f"[{elapsed:.1f}s]")

    log(f"\nEvaluations: {n_skipped} resumed, {n_new} newly run")

    # Compute aggregate statistics
    log(f"\n{'='*60}")
    log(f"  {dataset_name} AGGREGATE STATISTICS")
    log(f"{'='*60}")

    summary = {"dataset": dataset_name, "config": config, "n_files": len(files),
               "n_runs": n_runs, "strategies": {}}

    for sn in STRATEGY_ORDER:
        runs = all_results[sn]
        solved_counts = [r["solved"] for r in runs]
        par2_scores = [r["par2"] for r in runs]
        stats_solved = compute_stats(solved_counts)
        stats_par2 = compute_stats(par2_scores)
        summary["strategies"][sn] = {"solved": stats_solved, "par2": stats_par2, "runs": runs}
        log(f"  {sn}: solved={stats_solved['mean']:.1f} +/- {stats_solved['std']:.1f} "
            f"[{stats_solved['ci_low']:.1f}, {stats_solved['ci_high']:.1f}]")

    # McNemar per-run
    mcnemar_results = []
    for block_idx in range(n_runs):
        inst_opt = all_instances[block_idx].get("Opt-NIA", [])
        inst_z3 = all_instances[block_idx].get("Z3", [])
        if inst_opt and inst_z3:
            mc = compute_mcnemar(inst_opt, inst_z3)
            mc["block"] = block_idx + 1
            mcnemar_results.append(mc)
            log(f"  McNemar block {block_idx+1}: b={mc['b']}, c={mc['c']}, "
                f"chi2={mc['chi2']:.2f}, p={mc['p_value']:.4f}")
    summary["mcnemar"] = mcnemar_results

    # Improvement stats
    opt_runs = all_results["Opt-NIA"]
    z3_runs = all_results["Z3"]
    if len(opt_runs) == len(z3_runs):
        diffs = [o["solved"] - z["solved"] for o, z in zip(opt_runs, z3_runs)]
        diff_stats = compute_stats(diffs)
        diff_pp = compute_stats([d / len(files) * 100 for d in diffs])
        summary["improvement"] = {
            "instances": diff_stats,
            "percentage_points": diff_pp,
            "all_positive": all(d > 0 for d in diffs),
            "deltas": diffs,
        }
        log(f"\n  Improvement (Opt-NIA vs Z3): +{diff_stats['mean']:.1f} +/- {diff_stats['std']:.1f} "
            f"instances (+{diff_pp['mean']:.1f}pp)")
        log(f"  All positive: {summary['improvement']['all_positive']}")
        log(f"  Per-run deltas: {diffs}")

    # Save summary
    summary_path = os.path.join(output_dir, f"summary_{dataset_name}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nSummary: {summary_path}")

    # Save log
    log_path = os.path.join(output_dir, f"log_{dataset_name}.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))

    return summary


def main():
    parser = argparse.ArgumentParser(description="External benchmark evaluation")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIG.keys()),
                        help="Dataset to evaluate")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_external_eval(args.dataset, args.workers, project_root, args.resume)


if __name__ == "__main__":
    main()
