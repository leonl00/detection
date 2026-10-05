"""Tests for detection.resplit.

The assignment logic is tested with plain lists of group sizes. The full run
builds a small dataset of noise images in ``tmp_path``; images saved with the
same seed are identical and therefore twins.
"""

import random
from collections import Counter
from pathlib import Path

import pytest
import yaml
from PIL import Image

from detection.dataset_check import SPLITS
from detection.grouping import ImageRecord
from detection.resplit import (
    BACKGROUND,
    SPLIT_RATIOS,
    assign_groups,
    check_ratios,
    main,
    main_class,
    prepare_output,
    read_class_ids,
    resplit_dataset,
    split_groups,
    write_data_yaml,
)

CLASS_NAMES = ["spaghetti", "warping"]


def make_sample(root: Path, split: str, name: str, seed: int, label: str) -> None:
    """Save a 64x64 noise image and its label file into a split of ``root``."""
    for subfolder in ("images", "labels"):
        (root / split / subfolder).mkdir(parents=True, exist_ok=True)
    pixels = random.Random(seed).randbytes(64 * 64)
    Image.frombytes("L", (64, 64), pixels).save(root / split / "images" / f"{name}.png")
    (root / split / "labels" / f"{name}.txt").write_text(label, encoding="utf-8")


def make_dataset(root: Path, num_images: int = 20) -> Path:
    """Create a dataset of distinct images plus one twin pair split over train and test."""
    (root).mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(f"names: {CLASS_NAMES}\n", encoding="utf-8")
    for index in range(num_images):
        split = SPLITS[index % len(SPLITS)]
        make_sample(root, split, f"img_{index}", seed=index, label="0 0.5 0.5 0.2 0.2\n")
    make_sample(root, "train", "twin_a", seed=999, label="1 0.5 0.5 0.2 0.2\n")
    make_sample(root, "test", "twin_b", seed=999, label="1 0.5 0.5 0.2 0.2\n")
    return root


def split_of(dataset: Path, image_name: str) -> str:
    """Return the split in which an image file lies."""
    return next(split for split in SPLITS if (dataset / split / "images" / image_name).is_file())


# --- check_ratios ------------------------------------------------------------


def test_check_ratios_accepts_default() -> None:
    check_ratios(SPLIT_RATIOS)


@pytest.mark.parametrize(
    ("ratios", "message"),
    [
        ({"train": 0.8, "test": 0.2}, "keys"),
        ({"train": 1.0, "valid": 0.0, "test": 0.0}, "greater than 0"),
        ({"train": 0.7, "valid": 0.2, "test": 0.2}, "add up to 1"),
    ],
)
def test_check_ratios_rejects_bad_values(ratios: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        check_ratios(ratios)


# --- read_class_ids and main_class -------------------------------------------


def test_read_class_ids_reads_label_file(tmp_path: Path) -> None:
    make_sample(tmp_path, "train", "a", seed=1, label="0 0.5 0.5 0.1 0.1\n1 0.5 0.5 0.1 0.1\n")
    record = ImageRecord("train", tmp_path / "train" / "images" / "a.png", phash=None)

    assert read_class_ids(record, tmp_path, num_classes=2) == {0, 1}


def test_read_class_ids_missing_label_raises(tmp_path: Path) -> None:
    record = ImageRecord("train", tmp_path / "train" / "images" / "a.png", phash=None)

    with pytest.raises(FileNotFoundError):
        read_class_ids(record, tmp_path, num_classes=2)


def test_main_class_is_rarest_class() -> None:
    totals = Counter({0: 200, 1: 30, 2: 30})

    assert main_class({0, 1}, totals) == 1
    assert main_class({0, 1, 2}, totals) == 1  # tie: lower ID
    assert main_class(set(), totals) == BACKGROUND


# --- assign_groups and split_groups ------------------------------------------


def test_assign_groups_reaches_target_ratio() -> None:
    splits = assign_groups([1] * 100, SPLIT_RATIOS)

    assert Counter(splits) == {"train": 70, "valid": 15, "test": 15}


def test_assign_groups_counts_group_sizes() -> None:
    splits = assign_groups([7, 1, 1, 1], SPLIT_RATIOS)

    assert splits[0] == "train"
    assert set(splits[1:]) == {"valid", "test"}


def test_assign_groups_bad_ratios_raise() -> None:
    with pytest.raises(ValueError):
        assign_groups([1], {"train": 1.0})


def test_split_groups_is_stratified_and_reproducible() -> None:
    groups = [[index] for index in range(40)]
    main_classes = [0] * 20 + [1] * 20

    first = split_groups(groups, main_classes, SPLIT_RATIOS, seed=1)
    second = split_groups(groups, main_classes, SPLIT_RATIOS, seed=1)

    assert first == second
    for class_id in (0, 1):
        class_splits = [first[i] for i, main in enumerate(main_classes) if main == class_id]
        assert Counter(class_splits) == {"train": 14, "valid": 3, "test": 3}


def test_split_groups_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="main classes"):
        split_groups([[0], [1]], [0], SPLIT_RATIOS, seed=1)


# --- prepare_output and write_data_yaml --------------------------------------


def test_prepare_output_refuses_existing_folder(tmp_path: Path) -> None:
    (tmp_path / "out").mkdir()

    with pytest.raises(FileExistsError, match="overwrite"):
        prepare_output(tmp_path / "out", tmp_path / "src", overwrite=False)


def test_prepare_output_overwrite_removes_folder(tmp_path: Path) -> None:
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "old.txt").write_text("x", encoding="utf-8")

    prepare_output(tmp_path / "out", tmp_path / "src", overwrite=True)

    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("output", ["src", "src/clean"])
def test_prepare_output_refuses_source_folder(tmp_path: Path, output: str) -> None:
    (tmp_path / "src").mkdir()

    with pytest.raises(ValueError, match="separate"):
        prepare_output(tmp_path / output, tmp_path / "src", overwrite=True)
    assert (tmp_path / "src").is_dir()


def test_write_data_yaml_uses_relative_paths(tmp_path: Path) -> None:
    content = yaml.safe_load(write_data_yaml(tmp_path, CLASS_NAMES).read_text(encoding="utf-8"))

    assert content["train"] == "train/images"
    assert content["val"] == "valid/images"
    assert content["names"] == CLASS_NAMES
    assert content["nc"] == 2


# --- resplit_dataset and main ------------------------------------------------


def test_resplit_keeps_twins_together(tmp_path: Path) -> None:
    source = make_dataset(tmp_path / "dataset")
    output = tmp_path / "clean"

    counts = resplit_dataset(source, output, seed=3)

    assert sum(counts.values()) == 22
    assert split_of(output, "twin_a.png") == split_of(output, "twin_b.png")
    for split in SPLITS:
        images = {file.stem for file in (output / split / "images").iterdir()}
        labels = {file.stem for file in (output / split / "labels").iterdir()}
        assert images == labels


def test_resplit_is_reproducible(tmp_path: Path) -> None:
    source = make_dataset(tmp_path / "dataset")

    resplit_dataset(source, tmp_path / "first", seed=5)
    resplit_dataset(source, tmp_path / "second", seed=5)

    for split in SPLITS:
        first = sorted(f.name for f in (tmp_path / "first" / split / "images").iterdir())
        second = sorted(f.name for f in (tmp_path / "second" / split / "images").iterdir())
        assert first == second


def test_main_prints_counts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = make_dataset(tmp_path / "dataset")
    output = tmp_path / "clean"

    main(["--data", str(source), "--output", str(output)])

    assert (output / "data.yaml").is_file()
    assert "New dataset written to" in capsys.readouterr().out
