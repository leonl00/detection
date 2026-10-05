"""Tests for detection.dataset_check.

The tests build a tiny fake dataset in ``tmp_path``. The image files are empty,
because the structure check only looks at file names, not at file content.
"""

from pathlib import Path

import pytest

from detection.dataset_check import SPLITS, check_dataset_structure, read_class_names


def make_dataset(root: Path, images_per_split: int = 2) -> Path:
    """Create a minimal YOLOv8 folder layout with empty image files."""
    (root / "data.yaml").write_text("names: [spaghetti, stringing]\n", encoding="utf-8")
    for split in SPLITS:
        (root / split / "labels").mkdir(parents=True)
        images = root / split / "images"
        images.mkdir(parents=True)
        for index in range(images_per_split):
            (images / f"img_{index}.jpg").touch()
    return root


# --- check_dataset_structure -----------------------------------------------


def test_structure_counts_images_per_split(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path, images_per_split=3)

    assert check_dataset_structure(dataset) == {"train": 3, "valid": 3, "test": 3}


def test_structure_ignores_non_image_files(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path, images_per_split=1)
    (dataset / "train" / "images" / "notes.txt").touch()
    (dataset / "train" / "images" / "upper.PNG").touch()

    assert check_dataset_structure(dataset)["train"] == 2


def test_structure_without_data_yaml_raises(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)
    (dataset / "data.yaml").unlink()

    with pytest.raises(FileNotFoundError, match="data.yaml"):
        check_dataset_structure(dataset)


@pytest.mark.parametrize("missing", ["train/images", "valid/labels", "test/images"])
def test_structure_with_missing_folder_raises(tmp_path: Path, missing: str) -> None:
    dataset = make_dataset(tmp_path, images_per_split=0)
    (dataset / missing).rmdir()

    with pytest.raises(FileNotFoundError, match="missing"):
        check_dataset_structure(dataset)


# --- read_class_names ------------------------------------------------------


def test_class_names_as_list(tmp_path: Path) -> None:
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("nc: 2\nnames: [spaghetti, stringing]\n", encoding="utf-8")

    assert read_class_names(data_yaml) == ["spaghetti", "stringing"]


def test_class_names_as_dict_are_sorted_by_id(tmp_path: Path) -> None:
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names:\n  1: stringing\n  0: spaghetti\n", encoding="utf-8")

    assert read_class_names(data_yaml) == ["spaghetti", "stringing"]


def test_class_names_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_class_names(tmp_path / "data.yaml")


@pytest.mark.parametrize("content", ["nc: 2\n", "names: []\n", "names: spaghetti\n", ""])
def test_class_names_without_valid_names_raises(tmp_path: Path, content: str) -> None:
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="names"):
        read_class_names(data_yaml)
