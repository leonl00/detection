"""Split a YOLOv8 dataset anew, keeping groups of near-identical images together.

The original split of a dataset can put two copies of the same scene into train
and test (see ``detection.grouping``). This module builds the groups of
near-identical images and assigns every group as a whole to exactly one split,
so the test set only contains scenes the model has never seen.

The split is also stratified by class: every group gets a *main class*, the
rarest class among its boxes, and the groups of each main class are divided in
the target ratio on their own. That way even a rare class appears in all three
splits in about the right share.

The result is a new dataset folder with its own ``data.yaml``::

    data/dataset_clean/
    ├── data.yaml
    ├── train/images, train/labels
    ├── valid/images, valid/labels
    └── test/images,  test/labels

Usage (from the project root):
    python -m detection.resplit --data data/dataset --output data/dataset_clean
"""

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml

from detection.dataset_check import SPLITS, read_class_names, read_label_file
from detection.grouping import DEFAULT_MAX_DISTANCE, ImageRecord, find_groups, hash_dataset

# Target share of images per split (see docs/decisions.md).
SPLIT_RATIOS = {"train": 0.70, "valid": 0.15, "test": 0.15}

# Main class of groups whose images contain no boxes at all.
BACKGROUND = -1

DEFAULT_SEED = 42


# --- Checking the inputs -----------------------------------------------------


def check_ratios(ratios: dict[str, float]) -> None:
    """Check that the split ratios are usable.

    Args:
        ratios: Share of images per split, for example ``{"train": 0.7, ...}``.

    Raises:
        ValueError: If the splits are not exactly train, valid and test, a share
            is not positive, or the shares do not add up to 1.
    """
    if set(ratios) != set(SPLITS):
        raise ValueError(f"ratios must have the keys {SPLITS}, got {sorted(ratios)}")
    if any(share <= 0 for share in ratios.values()):
        raise ValueError(f"every share must be greater than 0, got {ratios}")
    if abs(sum(ratios.values()) - 1) > 1e-6:
        raise ValueError(f"shares must add up to 1, got {sum(ratios.values())}")


# --- Main class of a group ---------------------------------------------------


def read_class_ids(record: ImageRecord, dataset_dir: Path, num_classes: int) -> set[int]:
    """Read which classes occur in the label file of one image.

    Args:
        record: The image.
        dataset_dir: Root folder of the dataset.
        num_classes: Number of classes in ``data.yaml``.

    Returns:
        The class IDs of all valid boxes; empty for a background image.

    Raises:
        FileNotFoundError: If the image has no label file.
    """
    label_file = dataset_dir / record.split / "labels" / f"{record.path.stem}.txt"
    boxes, _ = read_label_file(label_file, num_classes)
    return {box.class_id for box in boxes}


def main_class(class_ids: set[int], class_totals: Counter) -> int:
    """Pick the rarest class of a group as its main class.

    Args:
        class_ids: Class IDs that occur in the group.
        class_totals: Number of boxes per class ID in the whole dataset.

    Returns:
        The class ID with the fewest boxes overall (the lower ID on a tie),
        or ``BACKGROUND`` if the group has no boxes.
    """
    if not class_ids:
        return BACKGROUND
    return min(class_ids, key=lambda class_id: (class_totals[class_id], class_id))


# --- Assigning groups to splits ----------------------------------------------


def assign_groups(group_sizes: list[int], ratios: dict[str, float]) -> list[str]:
    """Assign groups one after another to the split that is furthest below its target.

    After each step, the split that needs the most images to reach its share of
    all images assigned so far gets the next group. With many small groups this
    ends close to the target ratio.

    Args:
        group_sizes: Number of images per group, in the order of assignment.
        ratios: Target share of images per split.

    Returns:
        The split name for each group, in the same order as ``group_sizes``.

    Raises:
        ValueError: If the ratios are not usable (see ``check_ratios``).
    """
    check_ratios(ratios)
    counts = dict.fromkeys(SPLITS, 0)
    assigned = []
    for size in group_sizes:
        total_after = sum(counts.values()) + size
        # max() keeps the first of equal values, so ties go to train, then valid, then test.
        split = max(SPLITS, key=lambda name: ratios[name] * total_after - counts[name])
        counts[split] += size
        assigned.append(split)
    return assigned


def split_groups(
    groups: list[list[int]], main_classes: list[int], ratios: dict[str, float], seed: int
) -> list[str]:
    """Assign every group to a split, separately for each main class.

    Args:
        groups: Groups of image indices, as returned by ``find_groups``.
        main_classes: Main class of each group.
        ratios: Target share of images per split.
        seed: Seed for shuffling the groups; the same seed gives the same split.

    Returns:
        The split name for each group, in the order of ``groups``.

    Raises:
        ValueError: If the lists differ in length or the ratios are not usable.
    """
    if len(groups) != len(main_classes):
        raise ValueError(f"got {len(groups)} groups but {len(main_classes)} main classes")

    rng = random.Random(seed)
    group_splits = [""] * len(groups)
    for class_id in sorted(set(main_classes)):
        members = [index for index, main in enumerate(main_classes) if main == class_id]
        rng.shuffle(members)
        sizes = [len(groups[index]) for index in members]
        for index, split in zip(members, assign_groups(sizes, ratios), strict=True):
            group_splits[index] = split
    return group_splits


# --- Writing the new dataset -------------------------------------------------


def prepare_output(output_dir: Path, source_dir: Path, overwrite: bool) -> None:
    """Make sure the output folder is empty and separate from the source.

    Args:
        output_dir: Folder for the new dataset.
        source_dir: Folder of the original dataset.
        overwrite: Delete an existing output folder instead of stopping.

    Raises:
        ValueError: If output and source are the same folder or one lies inside the other.
        FileExistsError: If the output folder exists and ``overwrite`` is False.
    """
    output, source = output_dir.resolve(), source_dir.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError(f"output {output_dir} must be separate from the source {source_dir}")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists; use --overwrite to replace it")
        shutil.rmtree(output_dir)


def copy_image_and_label(
    record: ImageRecord, source_dir: Path, output_dir: Path, split: str
) -> None:
    """Copy one image and its label file into a split of the new dataset.

    Args:
        record: The image to copy.
        source_dir: Root folder of the original dataset.
        output_dir: Root folder of the new dataset.
        split: Split of the new dataset the image goes to.
    """
    label_file = source_dir / record.split / "labels" / f"{record.path.stem}.txt"
    images_dir = output_dir / split / "images"
    labels_dir = output_dir / split / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(record.path, images_dir / record.path.name)
    shutil.copy2(label_file, labels_dir / label_file.name)


def write_data_yaml(output_dir: Path, class_names: list[str]) -> Path:
    """Write the ``data.yaml`` of the new dataset.

    The paths are relative to the folder of the ``data.yaml``, which is where
    Ultralytics looks for them when no ``path`` entry is given.

    Args:
        output_dir: Root folder of the new dataset.
        class_names: Class names in the order of their IDs.

    Returns:
        Path of the written file.
    """
    content = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }
    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    return data_yaml


def resplit_dataset(
    source_dir: Path,
    output_dir: Path,
    ratios: dict[str, float] = SPLIT_RATIOS,
    seed: int = DEFAULT_SEED,
    max_distance: int = DEFAULT_MAX_DISTANCE,
    overwrite: bool = False,
) -> dict[str, int]:
    """Build the groups, assign them to splits and write the new dataset.

    Args:
        source_dir: Root folder of the original dataset.
        output_dir: Folder for the new dataset.
        ratios: Target share of images per split.
        seed: Seed for the assignment.
        max_distance: Largest phash distance that counts as a twin.
        overwrite: Replace an existing output folder.

    Returns:
        Number of images per split of the new dataset.

    Raises:
        FileNotFoundError: If the source dataset is incomplete.
        FileExistsError: If the output exists and ``overwrite`` is False.
        ValueError: If the ratios are not usable or the folders overlap.
    """
    check_ratios(ratios)
    prepare_output(output_dir, source_dir, overwrite)
    class_names = read_class_names(source_dir / "data.yaml")

    records = hash_dataset(source_dir)
    groups = find_groups([record.phash for record in records], max_distance)

    class_ids = [read_class_ids(record, source_dir, len(class_names)) for record in records]
    class_totals = Counter(class_id for ids in class_ids for class_id in ids)
    group_classes = [set().union(*(class_ids[index] for index in group)) for group in groups]
    main_classes = [main_class(ids, class_totals) for ids in group_classes]

    image_counts = dict.fromkeys(SPLITS, 0)
    for group, split in zip(groups, split_groups(groups, main_classes, ratios, seed), strict=True):
        for index in group:
            copy_image_and_label(records[index], source_dir, output_dir, split)
            image_counts[split] += 1

    write_data_yaml(output_dir, class_names)
    return image_counts


# --- Command line ------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line options.

    Args:
        argv: Options to parse; ``None`` means the real command line.

    Returns:
        The parsed options ``data``, ``output``, ``seed``, ``max_distance`` and ``overwrite``.
    """
    parser = argparse.ArgumentParser(description="Split a dataset anew, group by group.")
    parser.add_argument("--data", type=Path, default=Path("data/dataset"), help="source dataset")
    parser.add_argument(
        "--output", type=Path, default=Path("data/dataset_clean"), help="new dataset folder"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed for the split")
    parser.add_argument(
        "--max-distance",
        type=int,
        default=DEFAULT_MAX_DISTANCE,
        help="largest phash distance that counts as a twin",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Write the new split and print the number of images per split."""
    args = parse_args(argv)
    image_counts = resplit_dataset(
        args.data,
        args.output,
        seed=args.seed,
        max_distance=args.max_distance,
        overwrite=args.overwrite,
    )
    total = sum(image_counts.values())
    for split, count in image_counts.items():
        print(f"{split}: {count} images ({count / total:.0%})")
    print(f"New dataset written to {args.output}")


if __name__ == "__main__":
    main()
