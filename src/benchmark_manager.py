"""SMT-LIB benchmark downloader, manager, and train/test splitter."""

import json
import os
import random
import subprocess
import sys
from pathlib import Path


ZENODO_URLS = {
    "QF_NIA": "https://zenodo.org/records/10607722/files/QF_NIA.tar.zst?download=1",
    "QF_NRA": "https://zenodo.org/records/10607722/files/QF_NRA.tar.zst?download=1",
    "QF_BV": "https://zenodo.org/records/10607722/files/QF_BV.tar.zst?download=1",
    "QF_LIA": "https://zenodo.org/records/10607722/files/QF_LIA.tar.zst?download=1",
}

SMTLIB_GIT_BASE = "https://clc-gitlab.cs.uiowa.edu:2443/SMT-LIB-benchmarks"
LOGIC_REPOS = {k: f"{SMTLIB_GIT_BASE}/{k}.git" for k in ZENODO_URLS}


def download_benchmarks(logic: str, output_dir: str, max_instances: int = 2000,
                         method: str = "auto") -> list[str]:
    """Download SMT-LIB benchmarks. Tries Zenodo first, falls back to git."""
    logic_dir = os.path.join(output_dir, logic)
    os.makedirs(logic_dir, exist_ok=True)

    # Check if already downloaded
    existing = sorted(str(p) for p in Path(logic_dir).rglob("*.smt2"))
    if existing:
        print(f"  Already have {len(existing)} .smt2 files for {logic}")
        if len(existing) > max_instances:
            random.seed(42)
            existing = sorted(random.sample(existing, max_instances))
            print(f"  Sampled {max_instances}")
        return existing

    if method in ("auto", "zenodo"):
        try:
            return _download_zenodo(logic, logic_dir, max_instances)
        except Exception as e:
            print(f"  Zenodo failed: {e}")
            if method == "auto":
                print(f"  Trying git clone...")
                return _download_git_shallow(logic, logic_dir, max_instances)
            raise
    elif method == "git_shallow":
        return _download_git_shallow(logic, logic_dir, max_instances)
    else:
        raise ValueError(f"Unknown method: {method}")


def _download_zenodo(logic: str, logic_dir: str, max_instances: int) -> list[str]:
    """Download benchmarks from Zenodo (same source as z3alpha)."""
    if logic not in ZENODO_URLS:
        raise ValueError(f"Unknown logic: {logic}")

    url = ZENODO_URLS[logic]
    archive_path = os.path.join(logic_dir, f"{logic}.tar.zst")
    tar_path = os.path.join(logic_dir, f"{logic}.tar")

    print(f"  Downloading {logic} from Zenodo...")
    subprocess.run(
        ["curl", "-L", "-o", archive_path, "--progress-bar", url],
        check=True
    )
    size_mb = os.path.getsize(archive_path) / 1024 / 1024
    print(f"  Downloaded: {size_mb:.1f} MB")

    # Decompress zstd
    try:
        subprocess.run(["zstd", "--version"], capture_output=True, check=True)
        print(f"  Decompressing with zstd...")
        subprocess.run(["zstd", "-d", archive_path, "-o", tar_path],
                       check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print(f"  Decompressing with Python zstandard...")
        _python_zstd_decompress(archive_path, tar_path)

    # Extract tar
    print(f"  Extracting...")
    import tarfile
    with tarfile.open(tar_path) as tf:
        tf.extractall(logic_dir, filter="data")

    # Cleanup
    for p in (archive_path, tar_path):
        if os.path.exists(p):
            os.remove(p)

    # Find .smt2 files
    smt2_files = sorted(str(p) for p in Path(logic_dir).rglob("*.smt2"))
    print(f"  Found {len(smt2_files)} .smt2 files")

    if len(smt2_files) > max_instances:
        random.seed(42)
        smt2_files = sorted(random.sample(smt2_files, max_instances))
        print(f"  Sampled {max_instances}")

    return smt2_files


def _python_zstd_decompress(src: str, dst: str):
    """Decompress .zst using Python zstandard package."""
    try:
        import zstandard
    except ImportError:
        print(f"  Installing zstandard...")
        subprocess.run([sys.executable, "-m", "pip", "install", "zstandard"],
                       check=True, capture_output=True)
        import zstandard

    dctx = zstandard.ZstdDecompressor()
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        dctx.copy_stream(fin, fout)


def _download_git_shallow(logic: str, logic_dir: str, max_instances: int) -> list[str]:
    """Shallow clone SMT-LIB benchmark repo (fallback)."""
    if logic not in LOGIC_REPOS:
        raise ValueError(f"Unknown logic: {logic}")

    repo_dir = os.path.join(logic_dir, "repo")
    if os.path.isdir(repo_dir):
        print(f"  Repo exists: {repo_dir}")
    else:
        repo_url = LOGIC_REPOS[logic]
        print(f"  Cloning {repo_url} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, repo_dir],
            check=True, capture_output=True, text=True
        )

    smt2_files = sorted(str(p) for p in Path(repo_dir).rglob("*.smt2"))
    print(f"  Found {len(smt2_files)} .smt2 files")

    if len(smt2_files) > max_instances:
        random.seed(42)
        smt2_files = sorted(random.sample(smt2_files, max_instances))
        print(f"  Sampled {max_instances}")

    return smt2_files


def create_train_test_split(smt2_files: list[str], logic: str, output_dir: str,
                             train_ratio: float = 0.7, seed: int = 42) -> dict:
    """Split benchmark files into train/test sets."""
    random.seed(seed)
    shuffled = list(smt2_files)
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_files = sorted(shuffled[:split_idx])
    test_files = sorted(shuffled[split_idx:])

    split_info = {
        "logic": logic,
        "total": len(smt2_files),
        "train_count": len(train_files),
        "test_count": len(test_files),
        "train_ratio": train_ratio,
        "seed": seed,
        "train_files": train_files,
        "test_files": test_files
    }

    split_path = os.path.join(output_dir, logic, "split.json")
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    with open(split_path, "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"  Split: {len(train_files)} train / {len(test_files)} test")
    return split_info


def load_split(output_dir: str, logic: str) -> dict:
    """Load an existing train/test split."""
    split_path = os.path.join(output_dir, logic, "split.json")
    with open(split_path) as f:
        return json.load(f)


def get_benchmark_stats(smt2_files: list[str], sample_size: int = 50) -> dict:
    """Get quick stats about benchmark files."""
    import z3

    sizes = [os.path.getsize(f) for f in smt2_files[:sample_size]]
    parse_ok = parse_fail = 0

    for f in smt2_files[:sample_size]:
        try:
            z3.parse_smt2_file(f)
            parse_ok += 1
        except Exception:
            parse_fail += 1

    return {
        "total_files": len(smt2_files),
        "sample_size": min(sample_size, len(smt2_files)),
        "parse_success": parse_ok,
        "parse_failure": parse_fail,
        "avg_file_size_kb": round(sum(sizes) / len(sizes) / 1024, 1) if sizes else 0,
    }


if __name__ == "__main__":
    print("=== Benchmark Manager ===")
    print(f"Available logics: {list(ZENODO_URLS.keys())}")
    if len(sys.argv) > 1:
        logic = sys.argv[1]
        bench_dir = os.path.join(os.path.dirname(__file__), "..", "data", "benchmarks")
        files = download_benchmarks(logic, bench_dir, max_instances=100)
        print(f"Downloaded {len(files)} files")
    else:
        print("Usage: python benchmark_manager.py <logic>")
