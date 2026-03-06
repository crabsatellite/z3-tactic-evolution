"""Z3 tactic builder: JSON representation <-> Z3 Tactic objects."""

import z3
from typing import Any

# All available Z3 tactics (populated at import time)
AVAILABLE_TACTICS = set(z3.tactics())
AVAILABLE_PROBES = set(z3.probes())


def build_tactic(expr: dict | str) -> z3.Tactic:
    """Build a Z3 Tactic from a JSON-serializable expression tree.

    Supported node types:
        {"type": "tactic", "name": "simplify"}
        {"type": "Then", "children": [...]}
        {"type": "OrElse", "children": [...]}
        {"type": "With", "tactic": "simplify", "params": {"som": true}}
        {"type": "TryFor", "child": {...}, "timeout": 5000}
        {"type": "Repeat", "child": {...}, "max": 10}
        {"type": "ParOr", "children": [...]}
        {"type": "Cond", "probe": "is-qflia", "then": {...}, "else": {...}}
        {"type": "When", "probe": "is-qflia", "child": {...}}
        {"type": "FailIf", "probe": "has-quantifiers"}
    """
    if isinstance(expr, str):
        if expr not in AVAILABLE_TACTICS:
            raise ValueError(f"Unknown tactic: {expr}. Available: {sorted(AVAILABLE_TACTICS)}")
        return z3.Tactic(expr)

    t = expr.get("type", "")

    if t == "tactic":
        name = expr["name"]
        if name not in AVAILABLE_TACTICS:
            raise ValueError(f"Unknown tactic: {name}")
        return z3.Tactic(name)

    if t == "Then":
        children = [build_tactic(c) for c in expr["children"]]
        if len(children) < 2:
            raise ValueError("Then requires at least 2 children")
        return z3.Then(*children)

    if t == "OrElse":
        children = [build_tactic(c) for c in expr["children"]]
        result = children[0]
        for c in children[1:]:
            result = z3.OrElse(result, c)
        return result

    if t == "With":
        tactic_name = expr["tactic"]
        if tactic_name not in AVAILABLE_TACTICS:
            raise ValueError(f"Unknown tactic: {tactic_name}")
        params = expr.get("params", {})
        return z3.With(tactic_name, **params)

    if t == "TryFor":
        child = build_tactic(expr["child"])
        timeout = int(expr.get("timeout", 5000))
        return z3.TryFor(child, timeout)

    if t == "Repeat":
        child = build_tactic(expr["child"])
        max_iter = int(expr.get("max", 10))
        return z3.Repeat(child, max_iter)

    if t == "ParOr":
        children = [build_tactic(c) for c in expr["children"]]
        return z3.ParOr(*children)

    if t == "Cond":
        probe = _build_probe(expr["probe"])
        then_t = build_tactic(expr["then"])
        else_t = build_tactic(expr["else"])
        return z3.Cond(probe, then_t, else_t)

    if t == "When":
        probe = _build_probe(expr["probe"])
        child = build_tactic(expr["child"])
        return z3.When(probe, child)

    if t == "FailIf":
        probe = _build_probe(expr["probe"])
        return z3.FailIf(probe)

    raise ValueError(f"Unknown tactic type: {t}")


def _build_probe(expr: str | dict) -> z3.Probe:
    """Build a Z3 Probe from a string or expression."""
    if isinstance(expr, str):
        if expr not in AVAILABLE_PROBES:
            raise ValueError(f"Unknown probe: {expr}")
        return z3.Probe(expr)

    op = expr.get("op", "")
    if op == ">":
        return z3.Probe(expr["probe"]) > expr["value"]
    if op == "<":
        return z3.Probe(expr["probe"]) < expr["value"]
    if op == "==":
        return z3.Probe(expr["probe"]) == expr["value"]
    if op == "not":
        return z3.Not(z3.Probe(expr["probe"]))
    if op == "and":
        return z3.And(_build_probe(expr["left"]), _build_probe(expr["right"]))
    if op == "or":
        return z3.Or(_build_probe(expr["left"]), _build_probe(expr["right"]))

    raise ValueError(f"Unknown probe expression: {expr}")


MAX_TACTIC_DEPTH = 10
MAX_TACTIC_NODES = 30


def _count_depth_and_nodes(expr: dict | str, depth: int = 0) -> tuple[int, int]:
    """Return (max_depth, total_nodes) of a tactic expression."""
    if isinstance(expr, str):
        return depth, 1
    t = expr.get("type", "")
    if t == "tactic":
        return depth, 1
    if t in ("Then", "OrElse", "ParOr"):
        children = expr.get("children", [])
        max_d = depth
        total = 1
        for c in children:
            d, n = _count_depth_and_nodes(c, depth + 1)
            max_d = max(max_d, d)
            total += n
        return max_d, total
    if t in ("TryFor", "Repeat", "When"):
        d, n = _count_depth_and_nodes(expr.get("child", {}), depth + 1)
        return d, n + 1
    if t == "With":
        return depth, 1
    if t == "Cond":
        d1, n1 = _count_depth_and_nodes(expr.get("then", {}), depth + 1)
        d2, n2 = _count_depth_and_nodes(expr.get("else", {}), depth + 1)
        return max(d1, d2), n1 + n2 + 1
    if t == "FailIf":
        return depth, 1
    return depth, 1


def validate_tactic(expr: dict | str, max_depth: int = MAX_TACTIC_DEPTH,
                    max_nodes: int = MAX_TACTIC_NODES) -> tuple[bool, str]:
    """Validate a tactic expression without running it.
    Returns (is_valid, error_message).
    """
    try:
        depth, nodes = _count_depth_and_nodes(expr)
        if depth > max_depth:
            return False, f"Tactic too deep: {depth} > {max_depth}"
        if nodes > max_nodes:
            return False, f"Tactic too large: {nodes} nodes > {max_nodes}"
        build_tactic(expr)
        return True, ""
    except Exception as e:
        return False, str(e)


def tactic_to_str(expr: dict | str) -> str:
    """Convert a tactic expression to a human-readable string."""
    if isinstance(expr, str):
        return expr
    t = expr.get("type", "")
    if t == "tactic":
        return expr["name"]
    if t == "Then":
        return "Then(" + ", ".join(tactic_to_str(c) for c in expr["children"]) + ")"
    if t == "OrElse":
        return "OrElse(" + ", ".join(tactic_to_str(c) for c in expr["children"]) + ")"
    if t == "With":
        params = ", ".join(f"{k}={v}" for k, v in expr.get("params", {}).items())
        return f"With({expr['tactic']}, {params})"
    if t == "TryFor":
        return f"TryFor({tactic_to_str(expr['child'])}, {expr.get('timeout', 5000)})"
    if t == "Repeat":
        return f"Repeat({tactic_to_str(expr['child'])}, max={expr.get('max', 10)})"
    if t == "ParOr":
        return "ParOr(" + ", ".join(tactic_to_str(c) for c in expr["children"]) + ")"
    if t == "Cond":
        return f"Cond({expr['probe']}, {tactic_to_str(expr['then'])}, {tactic_to_str(expr['else'])})"
    if t == "When":
        return f"When({expr['probe']}, {tactic_to_str(expr['child'])})"
    if t == "FailIf":
        return f"FailIf({expr['probe']})"
    return str(expr)


def get_tactic_catalog() -> dict:
    """Return a catalog of all Z3 tactics grouped by category for LLM context."""
    catalog = {
        "preprocessing": [],
        "arithmetic": [],
        "bitvector": [],
        "solvers": [],
        "strategies": [],
        "quantifiers": [],
        "control": [],
        "other": []
    }

    category_hints = {
        "preprocessing": ["simplify", "propagate", "elim", "solve-eqs", "ctx", "dom", "nnf", "snf", "tseitin", "occf", "distribute", "cofactor", "norm", "reduce-args"],
        "arithmetic": ["lia2", "pb2", "nla2", "purify-arith", "normalize-bounds", "propagate-ineqs", "fm", "diff-neq", "add-bounds", "factor"],
        "bitvector": ["bit-blast", "bv1", "bv-slice", "max-bv", "reduce-bv", "ackermannize", "qfbv-sls"],
        "solvers": ["sat", "smt", "nlsat", "psat", "psmt", "sls-smt"],
        "strategies": ["qflia", "qflra", "qfbv", "qfnia", "qfnra", "qfuf", "ufnia", "auflia", "default"],
        "quantifiers": ["qe", "qsat", "nlqsat", "qe-light", "qe2"],
        "control": ["skip", "fail", "fail-if-undecided", "collect-statistics"]
    }

    for tactic_name in sorted(z3.tactics()):
        placed = False
        for cat, hints in category_hints.items():
            if any(h in tactic_name for h in hints):
                catalog[cat].append(tactic_name)
                placed = True
                break
        if not placed:
            catalog["other"].append(tactic_name)

    return catalog


# Seed strategies for initial population
SEED_STRATEGIES = {
    "z3_default": {"type": "tactic", "name": "smt"},

    "simplify_then_smt": {
        "type": "Then",
        "children": [
            {"type": "tactic", "name": "simplify"},
            {"type": "tactic", "name": "solve-eqs"},
            {"type": "tactic", "name": "smt"}
        ]
    },

    "aggressive_preprocess": {
        "type": "Then",
        "children": [
            {"type": "With", "tactic": "simplify", "params": {"som": True, "pull_cheap_ite": True}},
            {"type": "tactic", "name": "propagate-values"},
            {"type": "tactic", "name": "solve-eqs"},
            {"type": "tactic", "name": "elim-uncnstr"},
            {"type": "tactic", "name": "smt"}
        ]
    },

    "try_builtin_then_fallback": {
        "type": "OrElse",
        "children": [
            {"type": "TryFor", "child": {"type": "tactic", "name": "qfnia"}, "timeout": 5000},
            {"type": "Then", "children": [
                {"type": "tactic", "name": "simplify"},
                {"type": "tactic", "name": "smt"}
            ]}
        ]
    },

    "conditional_by_logic": {
        "type": "Cond",
        "probe": "is-qfnia",
        "then": {"type": "tactic", "name": "qfnia"},
        "else": {"type": "Then", "children": [
            {"type": "tactic", "name": "simplify"},
            {"type": "tactic", "name": "smt"}
        ]}
    }
}
