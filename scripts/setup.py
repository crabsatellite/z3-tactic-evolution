"""Setup script: download benchmarks and verify dependencies."""

import json
import os
import sys
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


def check_dependencies():
    """Check all required tools are available."""
    print("=== Checking dependencies ===\n")

    ok = True

    # Python
    print(f"  Python: {sys.version.split()[0]}")

    # Z3
    try:
        import z3
        ver = z3.get_version_string()
        print(f"  Z3: {ver} ({len(z3.tactics())} tactics)")
    except ImportError:
        print("  Z3: NOT INSTALLED")
        print("    Fix: python -m pip install z3-solver==4.16.0")
        ok = False

    print()
    return ok


def download_benchmarks():
    """Download QF_NIA benchmarks from SMT-LIB (Zenodo)."""
    from benchmark_manager import download_benchmarks as do_download
    from utils import load_config

    config = load_config()
    logic = config["benchmarks"]["primary_logic"]
    bench_dir = os.path.join(PROJECT_ROOT, config["paths"]["benchmarks_dir"])
    max_instances = config["benchmarks"].get("max_instances_per_logic", 2000)
    target = os.path.join(bench_dir, logic, "non-incremental")

    if os.path.exists(target) and len(os.listdir(target)) > 100:
        print(f"  Benchmarks already downloaded ({len(os.listdir(target))} files).\n")
        return True

    print("=== Downloading benchmarks: QF_NIA ===\n")
    print("  This may take several minutes (git shallow clone)...\n")

    try:
        do_download(logic, bench_dir, max_instances)
        print("  Benchmarks ready.\n")
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        print("  Download manually from https://zenodo.org/records/10607722\n")
        return False


def verify_split():
    """Verify the three-tier split file exists and has correct counts."""
    split_path = os.path.join(PROJECT_ROOT, "data", "benchmarks", "QF_NIA", "split_three_tier.json")
    if not os.path.exists(split_path):
        print("  Split file not found.\n")
        return False

    with open(split_path) as f:
        split = json.load(f)

    print("=== Data split ===\n")
    print(f"  Train-Inner:        {split.get('train_inner_count', '?')}")
    print(f"  Validation-Locked:  {split.get('validation_locked_count', '?')}")
    print(f"  Test:               {split.get('test_count', '?')}")
    print()
    return True


def main():
    print(f"\n{'='*60}")
    print(f"  Z3 Tactic Evolution — Setup")
    print(f"{'='*60}\n")

    if not check_dependencies():
        sys.exit(1)

    download_benchmarks()
    verify_split()

    print("Setup complete. Next steps:")
    print("  python src/multirun_eval.py --workers 4 --skip-aprove")
    print()


if __name__ == "__main__":
    main()
