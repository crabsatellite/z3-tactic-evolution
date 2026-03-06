"""Multi-run evaluation with interleaved block design for statistical credibility.

Runs each strategy multiple times on each dataset, saving timestamped evidence.
Uses a Latin square-inspired interleaved block design to prevent systematic bias.
Supports resuming from prior runs — skips evaluations with existing evidence files.

Usage:
    python src/multirun_eval.py [--workers 4] [--skip-aprove] [--resume]

Output:
    data/results/QF_NIA/multirun/
        evidence/          Per-run JSON with timestamps and per-instance data
        summary.json       Aggregate statistics (mean, std, CI, per-run McNemar)
        report.txt         Human-readable summary report
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from evaluator import evaluate_candidate

# ============================================================
# Strategy definitions (paper names)
# ============================================================

STRATEGIES = {
    "Z3": {
        "type": "tactic", "name": "smt"
    },
    "PP-base": {
        "type": "Then",
        "children": [
            {"type": "With", "tactic": "simplify", "params": {"som": True, "pull_cheap_ite": True}},
            {"type": "tactic", "name": "propagate-values"},
            {"type": "tactic", "name": "solve-eqs"},
            {"type": "tactic", "name": "elim-uncnstr"},
            {"type": "tactic", "name": "smt"}
        ]
    },
    "PP-ext": {
        "type": "Then",
        "children": [
            {"type": "With", "tactic": "simplify", "params": {"som": True, "pull_cheap_ite": True}},
            {"type": "tactic", "name": "propagate-values"},
            {"type": "tactic", "name": "ctx-simplify"},
            {"type": "tactic", "name": "solve-eqs"},
            {"type": "tactic", "name": "elim-uncnstr"},
            {"type": "tactic", "name": "propagate-ineqs"},
            {"type": "tactic", "name": "smt"}
        ]
    },
    "Opt-NIA": {
        "type": "Then",
        "children": [
            {"type": "With", "tactic": "simplify", "params": {"som": True, "pull_cheap_ite": True}},
            {"type": "tactic", "name": "propagate-values"},
            {"type": "tactic", "name": "ctx-simplify"},
            {"type": "tactic", "name": "solve-eqs"},
            {"type": "tactic", "name": "elim-uncnstr"},
            {"type": "tactic", "name": "propagate-ineqs"},
            {
                "type": "OrElse",
                "children": [
                    {"type": "TryFor", "timeout": 7000,
                     "child": {"type": "With", "tactic": "smt", "params": {"random_seed": 7}}},
                    {"type": "TryFor", "timeout": 2000,
                     "child": {"type": "With", "tactic": "smt", "params": {"random_seed": 31415}}},
                    {"type": "With", "tactic": "smt", "params": {"random_seed": 271}}
                ]
            }
        ]
    },
}

STRATEGY_ORDER = ["Z3", "PP-base", "PP-ext", "Opt-NIA"]

# ============================================================
# Interleaved block schedules (Latin square rotation)
# ============================================================

def make_block_schedule(n_blocks, strategies):
    """Generate interleaved evaluation order for each block."""
    n = len(strategies)
    schedule = []
    for block in range(n_blocks):
        rotated = strategies[block % n:] + strategies[:block % n]
        # Reverse every other block for additional balance
        if block % 2 == 1:
            rotated = list(reversed(rotated))
        schedule.append(rotated)
    return schedule


# ============================================================
# Dataset loading
# ============================================================

def load_datasets(project_root, skip_aprove=False):
    """Load all benchmark datasets."""
    datasets = {}

    split_path = os.path.join(project_root, "data", "benchmarks", "QF_NIA", "split_three_tier.json")
    with open(split_path) as f:
        split = json.load(f)

    # Resolve relative paths in split file to absolute paths
    def resolve_files(file_list):
        return [os.path.join(project_root, f) if not os.path.isabs(f) else f
                for f in file_list]

    datasets["Test"] = {"files": resolve_files(split["test_files"]), "n_runs": 5}
    datasets["Validation"] = {"files": resolve_files(split["validation_locked_files"]), "n_runs": 5}

    if not skip_aprove:
        # AProVE benchmarks are inside the QF_NIA download
        aprove_dir = os.path.join(project_root, "data", "benchmarks", "QF_NIA",
                                  "non-incremental", "QF_NIA", "AProVE")
        if not os.path.isdir(aprove_dir):
            print(f"WARNING: AProVE directory not found at {aprove_dir}")
            print("  Run 'python scripts/setup.py' first, or use --skip-aprove")
        else:
            aprove_files = []
            for root_dir, _, fnames in os.walk(aprove_dir):
                for fn in fnames:
                    if fn.endswith(".smt2"):
                        aprove_files.append(os.path.join(root_dir, fn))
            aprove_files.sort()
            datasets["AProVE"] = {"files": aprove_files, "n_runs": 3}

    return datasets


# ============================================================
# Statistics
# ============================================================

def t_critical(df, alpha=0.05):
    """Approximate t-critical value for two-tailed test (good enough for df >= 2)."""
    # Exact values for small df
    t_table = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
    if df in t_table:
        return t_table[df]
    return 1.96  # fallback for large df


def compute_stats(values):
    """Compute mean, std, min, max, 95% CI from a list of numbers."""
    n = len(values)
    if n == 0:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "ci_low": 0, "ci_high": 0, "n": 0}
    mean = sum(values) / n
    if n == 1:
        return {"mean": mean, "std": 0, "min": mean, "max": mean, "ci_low": mean, "ci_high": mean, "n": 1}
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(variance)
    se = std / math.sqrt(n)
    t_crit = t_critical(n - 1)
    ci_low = mean - t_crit * se
    ci_high = mean + t_crit * se
    return {
        "mean": round(mean, 2),
        "std": round(std, 2),
        "min": min(values),
        "max": max(values),
        "ci_low": round(ci_low, 2),
        "ci_high": round(ci_high, 2),
        "n": n
    }


def compute_mcnemar(instances_a, instances_b):
    """Compute McNemar test from two per-instance result lists."""
    solved_a = {r["file"]: r["status"] in ("sat", "unsat") for r in instances_a}
    solved_b = {r["file"]: r["status"] in ("sat", "unsat") for r in instances_b}

    a = b = c = d = 0
    for f in solved_a:
        sa = solved_a[f]
        sb = solved_b.get(f, False)
        if sa and sb:
            a += 1
        elif sa and not sb:
            b += 1
        elif not sa and sb:
            c += 1
        else:
            d += 1

    # McNemar chi2 with continuity correction
    if b + c == 0:
        chi2 = 0.0
        p_val = 1.0
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        # Approximate p-value from chi2 with 1 df
        z = math.sqrt(chi2)
        p_val = 2 * 0.5 * math.erfc(z / math.sqrt(2))

    return {"a": a, "b": b, "c": c, "d": d, "chi2": round(chi2, 4), "p_value": round(p_val, 6)}


# ============================================================
# Main evaluation loop
# ============================================================

def run_experiment(workers, skip_aprove, project_root, resume=False):
    """Run the full multi-run experiment."""
    # Write to reproduce/ to avoid overwriting shipped reference results in multirun/
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

    log(f"Multi-Run Experiment Started (resume={resume})")
    log(f"Workers: {workers}")
    log(f"Project root: {project_root}")

    # Load datasets
    datasets = load_datasets(project_root, skip_aprove)
    for name, ds in datasets.items():
        log(f"Dataset '{name}': {len(ds['files'])} files, {ds['n_runs']} runs")

    # Collect all results: results[dataset][strategy] = [list of run results]
    all_results = defaultdict(lambda: defaultdict(list))
    # Per-instance data for McNemar: instances[dataset][run_idx][strategy] = [instances]
    all_instances = defaultdict(lambda: defaultdict(dict))

    timeout_ms = 10000
    par_factor = 2

    # Count skipped/new evaluations for logging
    n_skipped = 0
    n_new = 0

    for dataset_name, ds in datasets.items():
        files = ds["files"]
        n_runs = ds["n_runs"]
        schedule = make_block_schedule(n_runs, list(STRATEGY_ORDER))

        log(f"\n{'='*70}")
        log(f"  DATASET: {dataset_name} ({len(files)} files, {n_runs} runs)")
        log(f"{'='*70}")

        for block_idx, strategy_order in enumerate(schedule):
            log(f"\n--- Block {block_idx + 1}/{n_runs} | Order: {' -> '.join(strategy_order)} ---")

            for strategy_name in strategy_order:
                evidence_file = os.path.join(
                    evidence_dir,
                    f"{dataset_name}_block{block_idx+1:02d}_{strategy_name}.json"
                )

                # Resume: load existing evidence if available
                if resume and os.path.exists(evidence_file):
                    try:
                        with open(evidence_file) as f:
                            existing = json.load(f)
                        run_data = existing["meta"]
                        all_results[dataset_name][strategy_name].append(run_data)
                        all_instances[dataset_name][block_idx][strategy_name] = existing["instances"]
                        n_skipped += 1
                        log(f"  RESUME: {strategy_name} on {dataset_name} block {block_idx+1} "
                            f"-> {run_data['solved']}/{run_data['total']} (loaded from evidence)")
                        continue
                    except (json.JSONDecodeError, KeyError) as e:
                        log(f"  RESUME: corrupt evidence file, re-running: {e}")

                tactic_expr = STRATEGIES[strategy_name]
                run_start = datetime.now()

                log(f"  Running: {strategy_name} on {dataset_name} "
                    f"(block {block_idx+1}/{n_runs})...")

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

                # Record summary
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

                all_results[dataset_name][strategy_name].append(run_data)
                all_instances[dataset_name][block_idx][strategy_name] = result.instances

                # Save per-run evidence file
                evidence = {
                    "meta": run_data,
                    "instances": result.instances,
                }
                with open(evidence_file, "w") as f:
                    json.dump(evidence, f, indent=1)

                log(f"    -> {strategy_name}: {result.solved_count}/{result.total_count} "
                    f"({run_data['solve_rate']}%) PAR2={run_data['par2']:.0f} "
                    f"[{elapsed:.1f}s]")

    log(f"\nEvaluations: {n_skipped} resumed from evidence, {n_new} newly run")

    # ============================================================
    # Compute aggregate statistics
    # ============================================================
    log(f"\n{'='*70}")
    log(f"  COMPUTING AGGREGATE STATISTICS")
    log(f"{'='*70}")

    summary = {"experiment": {
        "start": experiment_start.isoformat(),
        "end": datetime.now().isoformat(),
        "workers": workers,
        "timeout_ms": timeout_ms,
        "strategies": STRATEGY_ORDER,
    }, "datasets": {}}

    for dataset_name in datasets:
        ds_summary = {"strategies": {}}
        for strategy_name in STRATEGY_ORDER:
            runs = all_results[dataset_name][strategy_name]
            solved_counts = [r["solved"] for r in runs]
            par2_scores = [r["par2"] for r in runs]

            stats_solved = compute_stats(solved_counts)
            stats_par2 = compute_stats(par2_scores)

            ds_summary["strategies"][strategy_name] = {
                "solved": stats_solved,
                "par2": stats_par2,
                "runs": runs,
            }

            log(f"  {dataset_name} | {strategy_name}: "
                f"solved={stats_solved['mean']:.1f} +/- {stats_solved['std']:.1f} "
                f"[{stats_solved['min']}, {stats_solved['max']}] "
                f"95% CI: [{stats_solved['ci_low']:.1f}, {stats_solved['ci_high']:.1f}]")

        # Compute per-run McNemar (Opt-NIA vs Z3)
        mcnemar_results = []
        n_runs = datasets[dataset_name]["n_runs"]
        for block_idx in range(n_runs):
            instances_opt = all_instances[dataset_name][block_idx].get("Opt-NIA", [])
            instances_z3 = all_instances[dataset_name][block_idx].get("Z3", [])
            if instances_opt and instances_z3:
                mc = compute_mcnemar(instances_opt, instances_z3)
                mc["block"] = block_idx + 1
                mcnemar_results.append(mc)
                log(f"  {dataset_name} | McNemar block {block_idx+1}: "
                    f"b={mc['b']}, c={mc['c']}, chi2={mc['chi2']:.2f}, p={mc['p_value']:.4f}")

        if mcnemar_results:
            chi2_values = [m["chi2"] for m in mcnemar_results]
            p_values = [m["p_value"] for m in mcnemar_results]
            sig_count = sum(1 for p in p_values if p < 0.05)
            ds_summary["mcnemar"] = {
                "per_run": mcnemar_results,
                "mean_chi2": round(sum(chi2_values) / len(chi2_values), 4),
                "min_p": round(min(p_values), 6),
                "max_p": round(max(p_values), 6),
                "significant_runs": f"{sig_count}/{len(p_values)}",
            }
            log(f"  {dataset_name} | McNemar summary: "
                f"mean chi2={ds_summary['mcnemar']['mean_chi2']:.2f}, "
                f"p range=[{ds_summary['mcnemar']['min_p']:.4f}, {ds_summary['mcnemar']['max_p']:.4f}], "
                f"significant: {sig_count}/{len(p_values)}")

        # Compute improvement stats (Opt-NIA - Z3 per run)
        opt_runs = all_results[dataset_name]["Opt-NIA"]
        z3_runs = all_results[dataset_name]["Z3"]
        if len(opt_runs) == len(z3_runs):
            diffs = [o["solved"] - z["solved"] for o, z in zip(opt_runs, z3_runs)]
            diff_stats = compute_stats(diffs)
            n_files = opt_runs[0]["n_files"]
            diff_pp = compute_stats([d / n_files * 100 for d in diffs])
            ds_summary["improvement"] = {
                "instances": diff_stats,
                "percentage_points": diff_pp,
                "all_positive": all(d > 0 for d in diffs),
            }
            log(f"  {dataset_name} | Improvement: "
                f"+{diff_stats['mean']:.1f} +/- {diff_stats['std']:.1f} instances "
                f"(+{diff_pp['mean']:.1f}pp), "
                f"all positive: {ds_summary['improvement']['all_positive']}")

        summary["datasets"][dataset_name] = ds_summary

    # ============================================================
    # Save summary
    # ============================================================
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nSummary saved: {summary_path}")

    # ============================================================
    # Generate human-readable report
    # ============================================================
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("  MULTI-RUN EVALUATION REPORT")
    report_lines.append(f"  Generated: {datetime.now().isoformat()}")
    report_lines.append(f"  Duration: {(datetime.now() - experiment_start).total_seconds() / 3600:.1f} hours")
    report_lines.append(f"  Workers: {workers} | Timeout: {timeout_ms}ms")
    report_lines.append("=" * 70)

    for dataset_name in datasets:
        ds = summary["datasets"][dataset_name]
        n_files = datasets[dataset_name]["files"].__len__()
        n_runs = datasets[dataset_name]["n_runs"]
        report_lines.append(f"\n{'='*50}")
        report_lines.append(f"  {dataset_name} ({n_files} files, {n_runs} runs)")
        report_lines.append(f"{'='*50}")
        report_lines.append("")
        report_lines.append(f"  {'Strategy':<12} {'Solved (mean +/- std)':<24} {'95% CI':<20} {'PAR-2 (mean)':<14}")
        report_lines.append(f"  {'-'*12} {'-'*24} {'-'*20} {'-'*14}")

        for sn in STRATEGY_ORDER:
            ss = ds["strategies"][sn]["solved"]
            ps = ds["strategies"][sn]["par2"]
            report_lines.append(
                f"  {sn:<12} {ss['mean']:>6.1f} +/- {ss['std']:<5.1f}          "
                f"[{ss['ci_low']:>5.1f}, {ss['ci_high']:>5.1f}]       "
                f"{ps['mean']:>12,.0f}"
            )

        if "improvement" in ds:
            imp = ds["improvement"]
            report_lines.append(f"\n  Improvement (Opt-NIA vs Z3):")
            report_lines.append(f"    Instances: +{imp['instances']['mean']:.1f} "
                              f"+/- {imp['instances']['std']:.1f} "
                              f"[{imp['instances']['ci_low']:.1f}, {imp['instances']['ci_high']:.1f}]")
            report_lines.append(f"    Percentage points: +{imp['percentage_points']['mean']:.1f}pp "
                              f"[{imp['percentage_points']['ci_low']:.1f}, {imp['percentage_points']['ci_high']:.1f}]")
            report_lines.append(f"    All runs positive: {imp['all_positive']}")

        if "mcnemar" in ds:
            mc = ds["mcnemar"]
            report_lines.append(f"\n  McNemar's Test (Opt-NIA vs Z3):")
            report_lines.append(f"    Mean chi2: {mc['mean_chi2']:.2f}")
            report_lines.append(f"    p-value range: [{mc['min_p']:.4f}, {mc['max_p']:.4f}]")
            report_lines.append(f"    Significant runs (p<0.05): {mc['significant_runs']}")

    # Validity assessment
    report_lines.append(f"\n{'='*50}")
    report_lines.append("  VALIDITY ASSESSMENT")
    report_lines.append(f"{'='*50}")

    all_valid = True
    for dataset_name in datasets:
        ds = summary["datasets"][dataset_name]
        if "improvement" in ds:
            imp = ds["improvement"]
            valid = imp["all_positive"] and imp["instances"]["ci_low"] > 0
            status = "PASS" if valid else "FAIL"
            report_lines.append(f"  {dataset_name}: mean improvement > 0 and CI excludes 0: {status}")
            if not valid:
                all_valid = False

    for dataset_name in ["Test", "Validation"]:
        if dataset_name in summary["datasets"] and "mcnemar" in summary["datasets"][dataset_name]:
            mc = summary["datasets"][dataset_name]["mcnemar"]
            sig_frac = mc["significant_runs"]
            num, denom = map(int, sig_frac.split("/"))
            valid = num >= denom // 2 + 1  # majority
            status = "PASS" if valid else "FAIL"
            report_lines.append(f"  {dataset_name}: McNemar p<0.05 in majority of runs: {status} ({sig_frac})")
            if not valid:
                all_valid = False

    report_lines.append(f"\n  OVERALL: {'ALL CHECKS PASSED' if all_valid else 'SOME CHECKS FAILED'}")

    report_text = "\n".join(report_lines)
    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    log(f"Report saved: {report_path}")
    print("\n" + report_text)

    # Save execution log
    log_path = os.path.join(output_dir, "execution_log.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))

    return summary


def main():
    parser = argparse.ArgumentParser(description="Multi-run evaluation experiment")
    parser.add_argument("--workers", type=int, default=4,
                       help="Max parallel workers per evaluation (default: 4)")
    parser.add_argument("--skip-aprove", action="store_true",
                       help="Skip AProVE external benchmark (saves ~6 hours)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from existing evidence files (skip completed runs)")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_experiment(args.workers, args.skip_aprove, project_root, resume=args.resume)


if __name__ == "__main__":
    main()
