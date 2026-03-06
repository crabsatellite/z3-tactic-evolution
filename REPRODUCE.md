# Reproduction Guide

Step-by-step instructions to reproduce all results from the paper.

## Prerequisites

- Python 3.10+
- 4+ CPU cores (i7-12700K used in paper)
- ~10 GB disk space for benchmarks
- ~5 hours total compute time

```bash
pip install z3-solver==4.16.0
```

Verify: `python -c "import z3; print(z3.get_version_string())"` should print `4.16.0`.

## Stage 1: Setup (~20 minutes)

```bash
# Download QF_NIA benchmarks from SMT-LIB (Zenodo record 10607722)
python scripts/setup.py

# Verify benchmark count
python -c "
import json
with open('data/benchmarks/QF_NIA/split_three_tier.json') as f:
    d = json.load(f)
print(f'Train-Inner: {d[\"train_inner_count\"]}')
print(f'Validation-Locked: {d[\"validation_locked_count\"]}')
print(f'Test: {d[\"test_count\"]}')
"
# Expected: Train-Inner: 1000, Validation-Locked: 400, Test: 600
```

## Stage 2: Quick Sanity Check (~30 minutes)

Run each strategy once on each dataset (no repeated blocks):

```bash
python scripts/quick_test.py --workers 4
```

Expected: Opt-NIA solves more than Z3 on all datasets. Single-run counts may vary by +/-3 instances due to solver non-determinism.

To skip external benchmarks and only test on Test + Validation (~15 min):

```bash
python scripts/quick_test.py --workers 4 --skip-external
```

To run a single dataset and/or strategy (useful for incremental testing):

```bash
# Single dataset, both strategies:
python scripts/quick_test.py --dataset Test --workers 4

# Single dataset + single strategy:
python scripts/quick_test.py --dataset Test --strategy Z3 --workers 4
python scripts/quick_test.py --dataset Validation --strategy Opt-NIA --workers 4

# Available datasets: Test, Validation, AProVE, MathProblems, Dartagnan
# Available strategies: Z3, Opt-NIA
```

## Stage 3: Full Multi-Run Evaluation (~2 hours)

Run the full 5-run interleaved block evaluation:

```bash
python src/multirun_eval.py --workers 4
```

Opt-NIA should consistently solve more instances than Z3 on both Test and Validation sets across all 5 runs. Your results are saved to `data/results/QF_NIA/reproduce/`. The original author results (with per-instance evidence) are preserved in `data/results/QF_NIA/multirun/` for comparison.

## Stage 4: External Benchmarks (~1 hour each)

```bash
# AProVE (termination analysis, 2409 instances)
# Note: AProVE is included in multirun_eval.py by default (Stage 3 above).
# To run AProVE separately, re-run Stage 3 without --skip-aprove.

# MathProblems (math competition, 993 unseen instances)
python src/eval_external.py --dataset MathProblems --workers 4

# Dartagnan (memory verification, 341 unseen instances)
python src/eval_external.py --dataset Dartagnan --workers 4
```

Opt-NIA should solve more instances than Z3 on all three external datasets.

## Validation Criteria

Results are valid if:
1. Opt-NIA solves more instances than Z3 in every individual run across all 5 datasets
2. The 95% CI for the improvement excludes zero on all datasets

## Hardware Notes

- **Workers**: Use `--workers N` where N <= physical cores / 2. Paper uses 4 workers on 12-core CPU.
- **Timing**: Absolute solve times vary by hardware. Solve **counts** should be stable (+/- 2-3 instances from run-to-run non-determinism).
- **OS**: Developed on Windows 11. Linux/macOS should work without changes.

## Protocol Integrity

The evaluation protocol is frozen in `PROTOCOL_FREEZE.md`. The strategy was fixed before any evaluation on held-out data.
