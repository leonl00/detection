"""Checks on a dataset in YOLOv8 format.

A YOLOv8 dataset (as exported by Roboflow) looks like this::

    dataset/
    ├── data.yaml          paths and class names
    ├── train/images, train/labels
    ├── valid/images, valid/labels
    └── test/images,  test/labels

Stage 2 only checks this layout. Stage 3 adds the detailed checks
(class counts, invalid labels, image and box sizes, report).
"""

from pathlib import Path

import yaml

SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def check_dataset_structure(dataset_dir: Path) -> dict[str, int]:
    """Check that the downloaded dataset has the expected YOLOv8 layout.

    Expected layout::

        dataset_dir/
        ├── data.yaml
        ├── train/images, train/labels
        ├── valid/images, valid/labels
        └── test/images,  test/labels

    Args:
        dataset_dir: Root folder of the dataset.

    Returns:
        Number of images per split, for example ``{"train": 700, ...}``.

    Raises:
        FileNotFoundError: If ``data.yaml`` or one of the folders is missing.
    """
    if not (dataset_dir / "data.yaml").is_file():
        raise FileNotFoundError(f"data.yaml not found in {dataset_dir}")

    image_counts = {}
    for split in SPLITS:
        for subfolder in ("images", "labels"):
            folder = dataset_dir / split / subfolder
            if not folder.is_dir():
                raise FileNotFoundError(f"Expected folder is missing: {folder}")

        images = dataset_dir / split / "images"
        image_counts[split] = sum(
            1 for file in images.iterdir() if file.suffix.lower() in IMAGE_SUFFIXES
        )
    return image_counts


def read_class_names(data_yaml: Path) -> list[str]:
    """Read the class names from a YOLOv8 ``data.yaml``.

    Args:
        data_yaml: Path to the ``data.yaml`` file.

    Returns:
        The class names in the order of their class IDs.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file has no list of class names under ``names``.
    """
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    content = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = content.get("names") if isinstance(content, dict) else None

    # Ultralytics allows names as a list or as a dict {0: "a", 1: "b"}.
    if isinstance(names, dict):
        names = [names[class_id] for class_id in sorted(names)]
    if not isinstance(names, list) or not names:
        raise ValueError(f"{data_yaml} has no list of class names under 'names'")
    return [str(name) for name in names]
