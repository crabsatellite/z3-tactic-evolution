# Evaluation Protocol Freeze

**Freeze date:** 2026-03-05
**Status:** FROZEN — no changes to strategy, splits, or evaluation parameters permitted after this date.

## Frozen Strategy (Opt-NIA)

```
Then(
  With(simplify, som=True, pull_cheap_ite=True),
  propagate-values,
  ctx-simplify,
  solve-eqs,
  elim-uncnstr,
  propagate-ineqs,
  OrElse(
    TryFor(With(smt, random_seed=7), 7000),
    TryFor(With(smt, random_seed=31415), 2000),
    With(smt, random_seed=271)
  )
)
```

This strategy was finalized on 2026-03-04 and has not been modified since.

## Dataset Splits

| Partition | Count | Purpose | Used during development? |
|-----------|-------|---------|--------------------------|
| Train-Inner | 1,000 | Strategy development | Yes |
| Validation-Locked | 400 | Post-development checkpoint | No |
| Test | 600 | Final reported numbers | No |
| AProVE (external) | 2,409 | Cross-distribution validation | No |
| MathProblems (external) | 993 | Cross-distribution validation | No |
| Dartagnan (external) | 341 | Cross-distribution validation | No |

**Overlap exclusions:** MathProblems originally 1,100 instances (107 overlapped with primary split, excluded). Dartagnan originally 374 instances (33 overlapped, excluded). AProVE has zero overlap.

Split file: `data/benchmarks/QF_NIA/split_three_tier.json`

## Evaluation Parameters (Fixed)

- **Z3 version:** 4.16.0 (z3-solver Python package)
- **Timeout:** 10,000 ms per instance
- **Workers:** 4 parallel workers (instances parallelized, strategies sequential per instance)
- **Hardware:** Intel Core i7-12700K (12C/20T), 64 GB RAM, Windows 11
- **Multi-run protocol:** 5 runs (Test/Validation), 3 runs (external), interleaved block design
