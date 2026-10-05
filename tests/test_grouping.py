"""Tests for detection.grouping.

Grouping is tested with hand-made hashes, so the distance between two images is
known exactly. Hashing is tested with tiny images created by Pillow in ``tmp_path``.
"""

import random
from pathlib import Path

import imagehash
import pytest
from PIL import Image

from detection.dataset_check import SPLITS
from detection.grouping import (
    REPORT_NAME,
    ImageRecord,
    count_images_with_twin,
    find_groups,
    format_report,
    hash_dataset,
    hash_image,
    main,
)


def bits(number_of_ones: int) -> imagehash.ImageHash:
    """Return a 64-bit hash whose lowest ``number_of_ones`` bits are 1.

    The distance between ``bits(a)`` and ``bits(b)`` is exactly ``abs(a - b)``.
    """
    return imagehash.hex_to_hash(f"{(1 << number_of_ones) - 1:016x}")


def make_image(path: Path, seed: int) -> None:
    """Save a 64x64 grey noise image; the same seed gives the same image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = random.Random(seed).randbytes(64 * 64)
    Image.frombytes("L", (64, 64), pixels).save(path)


def make_records(splits: list[str]) -> list[ImageRecord]:
    """Create one record per entry in ``splits``; the hash does not matter here."""
    return [ImageRecord(split, Path(f"img_{i}.jpg"), bits(0)) for i, split in enumerate(splits)]


# --- hash_image and hash_dataset ---------------------------------------------


def test_hash_image_is_equal_for_equal_images(tmp_path: Path) -> None:
    make_image(tmp_path / "a.png", 1)
    make_image(tmp_path / "b.png", 1)
    make_image(tmp_path / "c.png", 2)

    first, second, other = (hash_image(tmp_path / name) for name in ("a.png", "b.png", "c.png"))

    assert first - second == 0
    assert first - other > 16


def test_hash_image_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Image not found"):
        hash_image(tmp_path / "missing.png")


def test_hash_dataset_returns_one_record_per_image(tmp_path: Path) -> None:
    for split in SPLITS:
        make_image(tmp_path / split / "images" / "a.png", 1)
    make_image(tmp_path / "train" / "images" / "b.png", 2)

    records = hash_dataset(tmp_path)

    assert [(r.split, r.path.name) for r in records] == [
        ("train", "a.png"),
        ("train", "b.png"),
        ("valid", "a.png"),
        ("test", "a.png"),
    ]


def test_hash_dataset_missing_split_raises(tmp_path: Path) -> None:
    make_image(tmp_path / "train" / "images" / "a.png", 1)

    with pytest.raises(FileNotFoundError, match="valid"):
        hash_dataset(tmp_path)


# --- find_groups -------------------------------------------------------------


def test_find_groups_puts_close_hashes_together() -> None:
    hashes = [bits(0), bits(40), bits(2), bits(64)]

    assert find_groups(hashes, max_distance=4) == [[0, 2], [1], [3]]


def test_find_groups_limit_is_inclusive() -> None:
    assert find_groups([bits(0), bits(14)], max_distance=14) == [[0, 1]]
    assert find_groups([bits(0), bits(15)], max_distance=14) == [[0], [1]]


def test_find_groups_is_transitive() -> None:
    # 0-10 and 10-20 are close, 0-20 is not: still one group via the middle image.
    hashes = [bits(0), bits(20), bits(10)]

    assert find_groups(hashes, max_distance=10) == [[0, 1, 2]]


def test_find_groups_empty_list() -> None:
    assert find_groups([], max_distance=14) == []


def test_find_groups_negative_distance_raises() -> None:
    with pytest.raises(ValueError, match="negative"):
        find_groups([bits(0)], max_distance=-1)


# --- count_images_with_twin --------------------------------------------------


def test_count_images_with_twin_counts_only_mixed_groups() -> None:
    records = make_records(["train", "test", "test", "test", "valid"])
    # Group 1: train + two test images. Group 2: a test image alone. Group 3: valid alone.
    groups = [[0, 1, 2], [3], [4]]

    assert count_images_with_twin(records, groups, "test", "train") == 2
    assert count_images_with_twin(records, groups, "valid", "train") == 0
    assert count_images_with_twin(records, groups, "test", "valid") == 0


def test_count_images_with_twin_same_split_raises() -> None:
    with pytest.raises(ValueError, match="differ"):
        count_images_with_twin(make_records(["test"]), [[0]], "test", "test")


# --- format_report and main --------------------------------------------------


def test_format_report_lists_leakage_and_groups() -> None:
    records = make_records(["train", "test", "valid"])

    report = format_report(records, [[0, 1], [2]], max_distance=14)

    assert "| Test images with a twin in train | 1 of 1 (100%) |" in report
    assert "| Valid images with a twin in train | 0 of 1 (0%) |" in report
    assert "**Group 1** (2 images):" in report
    assert "- test: img_1.jpg" in report


def test_main_writes_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset = tmp_path / "dataset"
    make_image(dataset / "train" / "images" / "a.png", 1)
    make_image(dataset / "valid" / "images" / "b.png", 2)
    make_image(dataset / "test" / "images" / "c.png", 1)
    reports = tmp_path / "reports"

    main(["--data", str(dataset), "--reports", str(reports)])

    assert (reports / REPORT_NAME).is_file()
    assert "1 of 1 test images have a near twin in train." in capsys.readouterr().out
