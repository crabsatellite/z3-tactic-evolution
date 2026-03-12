# Preprocessing, Seeds, and Time: A Simple Sequential Portfolio for QF_NIA

This repository contains the code and data for an optimized sequential portfolio strategy for QF_NIA (Quantifier-Free Nonlinear Integer Arithmetic) that improves Z3's solve rate using identical CPU budget.

## Key Results

| Dataset | Size | Runs | Z3 | Opt-NIA | Improvement | 95% CI |
|---------|------|------|----|---------|-------------|--------|
| Validation-Locked | 400 | 5 | 62.4% | 66.7% | **+4.3pp** | [+2.4, +6.2] |
| Test | 600 | 5 | 63.1% | 65.3% | **+2.2pp** | [+0.9, +3.5] |
| AProVE (external) | 2,409 | 3 | 88.5% | 89.3% | **+0.8pp** | [+0.1, +1.5] |
| MathProblems (external) | 993 | 3 | 7.7% | 11.4% | **+3.7pp** | [+3.3, +4.2] |
| Dartagnan (external) | 341 | 3 | 93.1% | 96.4% | **+3.3pp** | [+0.8, +5.9] |

All 19/19 runs positive. One-sided sign test across 5 independent benchmark sets: p = 0.031.

## The Strategy

The complete tactic string (copy-pasteable for any Z3 user):

```python
import z3

tactic = z3.Then(
    z3.With("simplify", som=True, pull_cheap_ite=True),
    "propagate-values",
    "ctx-simplify",
    "solve-eqs",
    "elim-uncnstr",
    "propagate-ineqs",
    z3.OrElse(
        z3.TryFor(z3.With("smt", random_seed=7), 7000),
        z3.TryFor(z3.With("smt", random_seed=31415), 2000),
        z3.With("smt", random_seed=271)
    )
)

solver = tactic.solver()
solver.add(your_constraints)
result = solver.check()
```

## Quick Start

```bash
# 1. Install dependencies
pip install z3-solver==4.16.0

# 2. Download benchmarks from SMT-LIB (Zenodo)
python scripts/setup.py

# 3. Quick sanity check (~15 min)
python src/multirun_eval.py --workers 4 --skip-aprove

# 4. Full multi-run evaluation (~2 hours)
python src/multirun_eval.py --workers 4

# 5. External benchmark evaluation (~1 hour each)
python src/eval_external.py --dataset MathProblems --workers 4
python src/eval_external.py --dataset Dartagnan --workers 4
```

## Three Techniques

1. **6-tactic preprocessing chain**: simplify(som) -> propagate-values -> ctx-simplify -> solve-eqs -> elim-uncnstr -> propagate-ineqs. Contributes +2.9pp alone.

2. **Greedy set cover for seed selection**: Selects [7, 31415, 271] for complementarity from candidate seeds. Outperforms arbitrary seeds by +11 instances.

3. **Front-loaded time allocation**: (7s, 2s, 1s) within 10s budget. Best seed gets 70% of time.

## Benchmarks

Benchmark .smt2 files are not included due to size (~8 GB).
Available from the SMT-LIB 2023 release on Zenodo:

- **QF_NIA**: [Zenodo record 10607722](https://zenodo.org/records/10607722)
- Run `python scripts/setup.py` to download automatically

## Requirements

- Python 3.10+
- z3-solver 4.16.0
- 4+ CPU cores recommended
- ~10 GB disk space for benchmarks

## Paper

*Preprocessing, Seeds, and Time: A Simple Sequential Portfolio for QF_NIA*
Alex Chengyu Li, 2026

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18909825.svg)](https://doi.org/10.5281/zenodo.18909825)
## Citation

```bibtex
@misc{li2026qfnia,
  author    = {Li, Alex Chengyu},
  title     = {Preprocessing, Seeds, and Time:
               {A} Simple Sequential Portfolio for {QF\_NIA}},
  year      = {2026},
  doi       = {10.5281/zenodo.18909825},
  note      = {Preprint, Zenodo}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

Benchmarks from the [SMT-LIB](https://smtlib.cs.uiowa.edu/) initiative.
Z3 solver by [Microsoft Research](https://github.com/Z3Prover/z3).
