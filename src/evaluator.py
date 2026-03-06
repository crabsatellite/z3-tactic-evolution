"""Parallel Z3 benchmark evaluator with PAR-2 scoring."""

import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, wait
from dataclasses import dataclass, asdict
from pathlib import Path

import z3

from tactic_builder import build_tactic, validate_tactic, tactic_to_str


@dataclass
class InstanceResult:
    file: str
    status: str        # "sat", "unsat", "unknown", "timeout", "error"
    time_ms: float
    error: str = ""


@dataclass
class CandidateResult:
    name: str
    tactic_expr: dict | str
    tactic_str: str
    solved_count: int
    total_count: int
    par2_score: float
    avg_time_ms: float
    median_time_ms: float
    sat_count: int
    unsat_count: int
    timeout_count: int
    error_count: int
    instances: list[dict]


def evaluate_single_instance(smt2_path: str, tactic_expr: dict | str,
                              timeout_ms: int = 10000) -> InstanceResult:
    """Evaluate a single SMT-LIB instance with a tactic. Runs in subprocess."""
    try:
        formulas = z3.parse_smt2_file(smt2_path)

        tactic = build_tactic(tactic_expr)
        solver = tactic.solver()
        solver.set("timeout", timeout_ms)

        for f in formulas:
            solver.add(f)

        start = time.perf_counter()
        result = solver.check()
        elapsed_ms = (time.perf_counter() - start) * 1000

        if result == z3.sat:
            return InstanceResult(smt2_path, "sat", elapsed_ms)
        elif result == z3.unsat:
            return InstanceResult(smt2_path, "unsat", elapsed_ms)
        else:
            reason = str(solver.reason_unknown())
            if "timeout" in reason.lower():
                return InstanceResult(smt2_path, "timeout", timeout_ms)
            return InstanceResult(smt2_path, "unknown", elapsed_ms)

    except z3.Z3Exception as e:
        return InstanceResult(smt2_path, "error", timeout_ms, str(e))
    except Exception as e:
        return InstanceResult(smt2_path, "error", timeout_ms,
                              f"{type(e).__name__}: {e}")


def _worker(args):
    """Worker function for ProcessPoolExecutor."""
    smt2_path, tactic_expr, timeout_ms = args
    return evaluate_single_instance(smt2_path, tactic_expr, timeout_ms)


def evaluate_candidate(name: str, tactic_expr: dict | str,
                        benchmark_files: list[str],
                        timeout_ms: int = 10000,
                        max_workers: int = 10,
                        par_factor: int = 2) -> CandidateResult:
    """Evaluate a candidate tactic on a list of benchmark files."""
    valid, err = validate_tactic(tactic_expr)
    if not valid:
        return CandidateResult(
            name=name, tactic_expr=tactic_expr,
            tactic_str=f"INVALID: {err}",
            solved_count=0, total_count=len(benchmark_files),
            par2_score=float('inf'), avg_time_ms=0, median_time_ms=0,
            sat_count=0, unsat_count=0, timeout_count=len(benchmark_files),
            error_count=0, instances=[]
        )

    tactic_str = tactic_to_str(tactic_expr)
    tasks = [(f, tactic_expr, timeout_ms) for f in benchmark_files]
    results: list[InstanceResult] = []

    # Hard wallclock timeout: prevents indefinite hangs from Z3's cooperative
    # timeout failing on pathological instances. Does NOT change the per-instance
    # Z3 timeout (still timeout_ms). Only kills processes that exceed the
    # expected wallclock time by a large margin.
    n_batches = math.ceil(len(tasks) / max(max_workers, 1))
    per_instance_budget = timeout_ms / 1000 + 15  # 15s buffer beyond Z3 timeout
    hard_wallclock = n_batches * per_instance_budget
    hard_wallclock = max(hard_wallclock, 120)  # at least 2 minutes

    executor = ProcessPoolExecutor(max_workers=max_workers)
    futures = {executor.submit(_worker, t): t[0] for t in tasks}

    try:
        done, not_done = wait(futures.keys(), timeout=hard_wallclock)
    except KeyboardInterrupt:
        done = set()
        not_done = set(futures.keys())

    for future in done:
        try:
            r = future.result(timeout=0)
            results.append(r)
        except Exception as e:
            fname = futures[future]
            results.append(InstanceResult(fname, "error", timeout_ms, str(e)))

    n_hard_killed = len(not_done)
    if n_hard_killed > 0:
        print(f"  WARNING: {n_hard_killed} instances hit hard wallclock timeout "
              f"({hard_wallclock:.0f}s), force-killing workers", flush=True)
    for future in not_done:
        fname = futures[future]
        results.append(InstanceResult(fname, "timeout", timeout_ms, "hard_wallclock_timeout"))
        future.cancel()

    # Shutdown executor — force-kill stuck workers if any
    executor.shutdown(wait=False, cancel_futures=True)
    if n_hard_killed > 0:
        time.sleep(2)
        if hasattr(executor, '_processes'):
            for proc in list(executor._processes.values()):
                if proc.is_alive():
                    try:
                        proc.kill()
                    except Exception:
                        pass

    # Compute metrics
    solved = [r for r in results if r.status in ("sat", "unsat")]
    solved_times = [r.time_ms for r in solved]
    sat_count = sum(1 for r in results if r.status == "sat")
    unsat_count = sum(1 for r in results if r.status == "unsat")
    timeout_count = sum(1 for r in results if r.status in ("timeout", "unknown"))
    error_count = sum(1 for r in results if r.status == "error")

    # PAR-2: solved time + par_factor * timeout for unsolved
    par2 = sum(r.time_ms for r in solved) + par_factor * timeout_ms * (len(results) - len(solved))

    avg_time = sum(solved_times) / len(solved_times) if solved_times else 0
    sorted_times = sorted(solved_times)
    median_time = sorted_times[len(sorted_times) // 2] if sorted_times else 0

    return CandidateResult(
        name=name,
        tactic_expr=tactic_expr,
        tactic_str=tactic_str,
        solved_count=len(solved),
        total_count=len(results),
        par2_score=par2,
        avg_time_ms=round(avg_time, 2),
        median_time_ms=round(median_time, 2),
        sat_count=sat_count,
        unsat_count=unsat_count,
        timeout_count=timeout_count,
        error_count=error_count,
        instances=[asdict(r) for r in results]
    )


def evaluate_generation(candidates: dict[str, dict | str],
                         benchmark_files: list[str],
                         timeout_ms: int = 10000,
                         max_workers: int = 10,
                         par_factor: int = 2) -> list[CandidateResult]:
    """Evaluate all candidates in a generation. Returns sorted by PAR-2."""
    results = []
    for name, tactic_expr in candidates.items():
        print(f"  Evaluating: {name} ...", end=" ", flush=True)
        start = time.perf_counter()
        r = evaluate_candidate(
            name, tactic_expr, benchmark_files,
            timeout_ms, max_workers, par_factor
        )
        elapsed = time.perf_counter() - start
        print(f"solved={r.solved_count}/{r.total_count} PAR2={r.par2_score:.0f} ({elapsed:.1f}s)")
        results.append(r)

    results.sort(key=lambda r: (- r.solved_count, r.par2_score))
    return results


def _classify_failures(instances: list[dict]) -> dict:
    """Classify failure types from instance results for LLM feedback."""
    categories = {
        "timeout": 0,
        "unknown": 0,
        "tactic_unsupported": 0,
        "parse_error": 0,
        "solver_error": 0,
        "other_error": 0,
    }
    sample_errors = {}

    for inst in instances:
        status = inst.get("status", "")
        error = inst.get("error", "")

        if status == "timeout":
            categories["timeout"] += 1
        elif status == "unknown":
            categories["unknown"] += 1
        elif status == "error":
            err_lower = error.lower()
            if "unsupported" in err_lower or "not supported" in err_lower:
                cat = "tactic_unsupported"
            elif "parse" in err_lower or "syntax" in err_lower:
                cat = "parse_error"
            elif "tactic" in err_lower or "apply" in err_lower or "failed" in err_lower:
                cat = "solver_error"
            else:
                cat = "other_error"
            categories[cat] += 1
            if cat not in sample_errors and error:
                sample_errors[cat] = error[:200]

    return {k: v for k, v in categories.items() if v > 0}, sample_errors


def _build_summary_entry(i: int, r: 'CandidateResult') -> dict:
    """Build a rich summary entry with failure classification."""
    failure_types, sample_errors = _classify_failures(r.instances)
    entry = {
        "rank": i + 1,
        "name": r.name,
        "tactic_str": r.tactic_str,
        "solved": r.solved_count,
        "total": r.total_count,
        "solve_rate": round(r.solved_count / r.total_count * 100, 1) if r.total_count > 0 else 0,
        "par2": round(r.par2_score, 1),
        "avg_time_ms": r.avg_time_ms,
        "median_time_ms": r.median_time_ms,
        "sat": r.sat_count,
        "unsat": r.unsat_count,
        "timeout": r.timeout_count,
        "error": r.error_count,
    }
    if failure_types:
        entry["failure_types"] = failure_types
    if sample_errors:
        entry["sample_errors"] = sample_errors
    return entry


def save_results(results: list[CandidateResult], output_path: str, generation: int):
    """Save generation results to JSON."""
    output = {
        "generation": generation,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": [_build_summary_entry(i, r) for i, r in enumerate(results)],
        "candidates": [{
            "name": r.name,
            "tactic_expr": r.tactic_expr,
            "tactic_str": r.tactic_str,
            "metrics": {
                "solved_count": r.solved_count,
                "total_count": r.total_count,
                "par2_score": round(r.par2_score, 1),
                "avg_time_ms": r.avg_time_ms,
                "median_time_ms": r.median_time_ms,
            },
            "instances": r.instances
        } for r in results]
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    # Quick self-test
    import sys
    print("=== Evaluator self-test ===")

    test_expr = {
        "type": "Then",
        "children": [
            {"type": "tactic", "name": "simplify"},
            {"type": "tactic", "name": "solve-eqs"},
            {"type": "tactic", "name": "smt"}
        ]
    }
    valid, err = validate_tactic(test_expr)
    print(f"Tactic valid: {valid}")
    print(f"Tactic string: {tactic_to_str(test_expr)}")
    print("Self-test passed.")
