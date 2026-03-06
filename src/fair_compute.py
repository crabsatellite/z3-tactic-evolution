"""Fair-compute converter: transform ParOr strategies into equal-CPU-budget sequential versions.

ParOr(s1, s2, s3, s4) with timeout T uses K*T CPU-seconds (K parallel threads).
Fair version: OrElse(TryFor(s1, T/K), TryFor(s2, T/K), ..., TryFor(s_{K-1}, T/K), s_K)
uses <= T CPU-seconds total (sequential, same as single solver).

Fair compute comparisons ensure apples-to-apples evaluation against the baseline.
"""

import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


def paror_to_sequential(tactic_expr: dict, total_timeout_ms: int = 10000) -> dict:
    """Convert a tactic tree's ParOr nodes to sequential OrElse+TryFor.

    Recursively finds ParOr nodes and converts them to sequential equivalents
    that use the same total CPU budget as a single solver.
    """
    if isinstance(tactic_expr, str):
        return tactic_expr

    expr = copy.deepcopy(tactic_expr)
    t = expr.get("type", "")

    if t == "ParOr":
        children = expr.get("children", [])
        k = len(children)
        if k == 0:
            return expr

        # Allocate time budget: equal slices, last child gets remaining time
        time_per_child = total_timeout_ms // k

        # Build OrElse(TryFor(c1, T/K), TryFor(c2, T/K), ..., c_last)
        or_children = []
        for i, child in enumerate(children):
            converted_child = paror_to_sequential(child, time_per_child)
            if i < k - 1:
                # Wrap in TryFor
                or_children.append({
                    "type": "TryFor",
                    "timeout": time_per_child,
                    "child": converted_child
                })
            else:
                # Last child gets remaining time (no TryFor, uses solver's own timeout)
                or_children.append(converted_child)

        return {"type": "OrElse", "children": or_children}

    # Recurse into other composite nodes
    if t in ("Then", "OrElse"):
        expr["children"] = [paror_to_sequential(c, total_timeout_ms)
                            for c in expr.get("children", [])]
    elif t in ("TryFor", "Repeat"):
        child_timeout = expr.get("timeout", total_timeout_ms)
        expr["child"] = paror_to_sequential(expr.get("child", {}), child_timeout)
    elif t == "Cond":
        expr["then"] = paror_to_sequential(expr.get("then", {}), total_timeout_ms)
        expr["else"] = paror_to_sequential(expr.get("else", {}), total_timeout_ms)
    elif t == "When":
        expr["child"] = paror_to_sequential(expr.get("child", {}), total_timeout_ms)
    elif t == "With":
        pass  # Leaf node with params, no recursion needed

    return expr


def generate_fair_variants(candidates: dict, total_timeout_ms: int = 10000) -> dict:
    """Generate fair-compute variants for all candidates that use ParOr."""
    fair_candidates = {}

    for name, expr in candidates.items():
        # Check if this candidate uses ParOr
        if _contains_paror(expr):
            fair_name = f"{name}_fair"
            fair_expr = paror_to_sequential(expr, total_timeout_ms)
            fair_candidates[fair_name] = fair_expr

    return fair_candidates


def _contains_paror(expr: dict | str) -> bool:
    """Check if a tactic expression contains any ParOr nodes."""
    if isinstance(expr, str):
        return False
    t = expr.get("type", "")
    if t == "ParOr":
        return True
    if t in ("Then", "OrElse"):
        return any(_contains_paror(c) for c in expr.get("children", []))
    if t in ("TryFor", "Repeat", "When"):
        return _contains_paror(expr.get("child", {}))
    if t == "Cond":
        return _contains_paror(expr.get("then", {})) or _contains_paror(expr.get("else", {}))
    return False


def create_dual_track_candidates(candidates_path: str, output_path: str = None,
                                   total_timeout_ms: int = 10000):
    """Create a combined candidates file with both Track P and Track F versions."""
    with open(candidates_path) as f:
        data = json.load(f)

    candidates = data.get("candidates", {})
    fair_variants = generate_fair_variants(candidates, total_timeout_ms)

    combined = {}
    combined.update(candidates)  # Track P (parallel)
    combined.update(fair_variants)  # Track F (fair-compute)

    output = {
        "generation": data.get("generation", 0),
        "dual_track": True,
        "track_p_count": len(candidates),
        "track_f_count": len(fair_variants),
        "candidates": combined
    }

    if output_path is None:
        output_path = candidates_path.replace(".json", "_dual.json")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Dual-track candidates created:")
    print(f"  Track P (parallel): {len(candidates)} candidates")
    print(f"  Track F (fair):     {len(fair_variants)} candidates")
    print(f"  Total:              {len(combined)} candidates")
    print(f"  Saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    from tactic_builder import tactic_to_str

    # Example: convert our best strategy
    best = {
        "type": "Then",
        "children": [
            {"type": "With", "tactic": "simplify", "params": {"som": True, "pull_cheap_ite": True}},
            {"type": "tactic", "name": "propagate-values"},
            {"type": "tactic", "name": "ctx-simplify"},
            {"type": "tactic", "name": "solve-eqs"},
            {"type": "tactic", "name": "elim-uncnstr"},
            {"type": "tactic", "name": "propagate-ineqs"},
            {"type": "ParOr", "children": [
                {"type": "With", "tactic": "smt", "params": {"random_seed": 0}},
                {"type": "With", "tactic": "smt", "params": {"random_seed": 42}},
                {"type": "With", "tactic": "smt", "params": {"random_seed": 137}},
                {"type": "With", "tactic": "smt", "params": {"random_seed": 314159}},
            ]}
        ]
    }

    print("=== Fair-Compute Converter ===\n")
    print(f"Original (Track P):")
    print(f"  {tactic_to_str(best)}\n")

    fair = paror_to_sequential(best, 10000)
    print(f"Fair-compute (Track F, 10s budget):")
    print(f"  {tactic_to_str(fair)}\n")

    fair_5s = paror_to_sequential(best, 5000)
    print(f"Fair-compute (Track F, 5s budget):")
    print(f"  {tactic_to_str(fair_5s)}")
