"""Create three-tier data split for rigorous experimental design.

Train-Inner (1000): used for evolution/optimization
Validation-Locked (400): used ONLY for selecting top strategies, never for search
Test (600): final evaluation, run ONCE at the end

The 600 test files are UNCHANGED from the original split.
The 1400 train files are sub-split into 1000 + 400 deterministically.
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from utils import get_project_root


def create_three_tier_split(seed: int = 123):
    """Sub-split existing 1400 train into 1000 Train-Inner + 400 Validation-Locked."""
    root = get_project_root()
    split_path = os.path.join(root, "data", "benchmarks", "QF_NIA", "split.json")

    with open(split_path) as f:
        original = json.load(f)

    train_files = original["train_files"]  # 1400 files
    test_files = original["test_files"]    # 600 files

    assert len(train_files) == 1400, f"Expected 1400 train files, got {len(train_files)}"
    assert len(test_files) == 600, f"Expected 600 test files, got {len(test_files)}"

    # Deterministic sub-split of train
    random.seed(seed)
    shuffled_train = list(train_files)
    random.shuffle(shuffled_train)

    train_inner = sorted(shuffled_train[:1000])
    validation_locked = sorted(shuffled_train[1000:])

    assert len(train_inner) == 1000
    assert len(validation_locked) == 400
    assert len(set(train_inner) & set(validation_locked)) == 0, "Overlap between train-inner and validation!"
    assert len(set(train_inner) & set(test_files)) == 0, "Overlap between train-inner and test!"
    assert len(set(validation_locked) & set(test_files)) == 0, "Overlap between validation and test!"

    three_tier = {
        "logic": "QF_NIA",
        "total": len(train_files) + len(test_files),
        "split_version": "three_tier_v1",
        "split_seed": seed,
        "original_train_ratio": original["train_ratio"],
        "original_seed": original["seed"],
        "train_inner_count": len(train_inner),
        "validation_locked_count": len(validation_locked),
        "test_count": len(test_files),
        "train_inner_files": train_inner,
        "validation_locked_files": validation_locked,
        "test_files": test_files,
        # Backward compatibility
        "train_files": train_files,
        "train_count": len(train_files),
    }

    output_path = os.path.join(root, "data", "benchmarks", "QF_NIA", "split_three_tier.json")
    with open(output_path, "w") as f:
        json.dump(three_tier, f, indent=2)

    print(f"Three-tier split created:")
    print(f"  Train-Inner:        {len(train_inner)} files (for development)")
    print(f"  Validation-Locked:  {len(validation_locked)} files (for strategy selection)")
    print(f"  Test:               {len(test_files)} files (final evaluation)")
    print(f"  Saved to: {output_path}")

    return three_tier


if __name__ == "__main__":
    create_three_tier_split()
