"""Find near-identical images and check whether they leak across splits.

Many datasets contain the same photo more than once: as a re-upload under another
name, slightly cropped, or as neighbouring frames of a video. If one copy lands in
train and another in test, the model is tested on a scene it has already seen and
the test metrics are too optimistic.

To find such copies, every image gets a perceptual hash (``phash``): a 64-bit
fingerprint of its coarse brightness pattern. Similar images have similar hashes,
and the number of differing bits (the Hamming distance) measures how alike they
are. Images closer than ``max_distance`` are linked, and all images that are
linked directly or through other images form one group.

The main result is one number: how many test images have a near twin in train.

Usage (from the project root):
    python -m detection.grouping --data data/dataset --reports reports
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import imagehash
from PIL import Image

from detection.dataset_check import SPLITS, list_images

# Largest Hamming distance (of 64 bits) at which two images count as the same scene.
# Chosen by looking at image pairs of the original dataset (see docs/decisions.md):
# up to 12 all pairs showed the same scene, at 14 three of four, at 16 none.
# Missing a twin causes leakage, while a wrong match only makes a group larger,
# so the limit is set to 14.
DEFAULT_MAX_DISTANCE = 14

REPORT_NAME = "grouping_report.md"


@dataclass
class ImageRecord:
    """One image of the dataset with its perceptual hash.

    Attributes:
        split: Split the image belongs to (train, valid or test).
        path: Path to the image file.
        phash: Perceptual hash of the image.
    """

    split: str
    path: Path
    phash: imagehash.ImageHash


# --- Hashing -----------------------------------------------------------------


def hash_image(image_file: Path) -> imagehash.ImageHash:
    """Compute the perceptual hash of one image.

    Args:
        image_file: Path to the image.

    Returns:
        The 64-bit perceptual hash.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not image_file.is_file():
        raise FileNotFoundError(f"Image not found: {image_file}")
    with Image.open(image_file) as image:
        return imagehash.phash(image)


def hash_dataset(dataset_dir: Path) -> list[ImageRecord]:
    """Hash every image of all splits of a YOLOv8 dataset.

    Args:
        dataset_dir: Root folder of the dataset.

    Returns:
        One record per image, ordered by split and file name.

    Raises:
        FileNotFoundError: If an image folder is missing.
    """
    records = []
    for split in SPLITS:
        for image_file in list_images(dataset_dir / split / "images"):
            records.append(ImageRecord(split, image_file, hash_image(image_file)))
    return records


# --- Grouping ----------------------------------------------------------------


def find_groups(hashes: list[imagehash.ImageHash], max_distance: int) -> list[list[int]]:
    """Group images whose hashes are at most ``max_distance`` bits apart.

    The grouping is transitive: if A is close to B and B is close to C, then A,
    B and C form one group even if A and C are far apart. Every image is in
    exactly one group; an image without a twin forms a group of its own.

    Args:
        hashes: One hash per image.
        max_distance: Largest Hamming distance that still counts as a twin.

    Returns:
        Groups as lists of indices into ``hashes``. Each group is sorted, and the
        groups are sorted by their first index.

    Raises:
        ValueError: If ``max_distance`` is negative.
    """
    if max_distance < 0:
        raise ValueError(f"max_distance must not be negative, got {max_distance}")

    # For each image, the list of images that are close enough to it.
    neighbours: list[list[int]] = [[] for _ in hashes]
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if hashes[i] - hashes[j] <= max_distance:
                neighbours[i].append(j)
                neighbours[j].append(i)

    # Collect each group by following the links from a start image (depth-first search).
    group_of = [-1] * len(hashes)
    groups = []
    for start in range(len(hashes)):
        if group_of[start] != -1:
            continue
        group_of[start] = len(groups)
        members, to_visit = [], [start]
        while to_visit:
            current = to_visit.pop()
            members.append(current)
            for other in neighbours[current]:
                if group_of[other] == -1:
                    group_of[other] = len(groups)
                    to_visit.append(other)
        groups.append(sorted(members))
    return groups


def count_images_with_twin(
    records: list[ImageRecord], groups: list[list[int]], split: str, other_split: str
) -> int:
    """Count images of one split whose group also contains an image of another split.

    Args:
        records: All images, as returned by ``hash_dataset``.
        groups: Groups of indices into ``records``, as returned by ``find_groups``.
        split: Split whose images are counted, for example ``"test"``.
        other_split: Split in which a twin is searched, for example ``"train"``.

    Returns:
        Number of images in ``split`` with at least one twin in ``other_split``.

    Raises:
        ValueError: If ``split`` and ``other_split`` are the same.
    """
    if split == other_split:
        raise ValueError(f"split and other_split must differ, both are {split!r}")

    count = 0
    for group in groups:
        splits_in_group = [records[index].split for index in group]
        if other_split in splits_in_group:
            count += splits_in_group.count(split)
    return count


# --- Report ------------------------------------------------------------------


def format_report(records: list[ImageRecord], groups: list[list[int]], max_distance: int) -> str:
    """Write the grouping results as a Markdown text.

    Args:
        records: All images, as returned by ``hash_dataset``.
        groups: Groups of indices into ``records``.
        max_distance: The distance limit used for the grouping.

    Returns:
        The report as Markdown.
    """
    shared = [group for group in groups if len(group) > 1]
    mixed = [group for group in shared if len({records[i].split for i in group}) > 1]

    lines = [
        "# Grouping report",
        "",
        "Created by `python -m detection.grouping`. Two images count as twins when their",
        f"perceptual hashes (phash, 64 bit) differ in at most {max_distance} bits.",
        "",
        "## Leakage between splits",
        "",
        "| Question | Images |",
        "|---|---:|",
        _leak_row("Test images with a twin in train", records, groups, "test", "train"),
        _leak_row("Valid images with a twin in train", records, groups, "valid", "train"),
        _leak_row("Test images with a twin in valid", records, groups, "test", "valid"),
        "",
        "## Groups",
        "",
        f"- Images: {len(records)}",
        f"- Groups: {len(groups)}",
        f"- Groups with more than one image: {len(shared)} "
        f"(containing {sum(len(group) for group in shared)} images)",
        f"- Groups spread over more than one split: {len(mixed)}",
        f"- Largest group: {max((len(group) for group in groups), default=0)} images",
        "",
        "## Groups with more than one image",
    ]
    for number, group in enumerate(shared, start=1):
        lines += ["", f"**Group {number}** ({len(group)} images):", ""]
        lines += [f"- {records[i].split}: {records[i].path.name}" for i in group]
    return "\n".join(lines) + "\n"


def _leak_row(
    label: str,
    records: list[ImageRecord],
    groups: list[list[int]],
    split: str,
    other_split: str,
) -> str:
    """Format one table row: ``| label | count of total (share) |``."""
    count = count_images_with_twin(records, groups, split, other_split)
    total = sum(1 for record in records if record.split == split)
    share = f" ({count / total:.0%})" if total else ""
    return f"| {label} | {count} of {total}{share} |"


# --- Command line ------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line options.

    Args:
        argv: Options to parse; ``None`` means the real command line.

    Returns:
        The parsed options ``data``, ``reports`` and ``max_distance``.
    """
    parser = argparse.ArgumentParser(description="Find near-identical images across splits.")
    parser.add_argument("--data", type=Path, default=Path("data/dataset"), help="dataset folder")
    parser.add_argument("--reports", type=Path, default=Path("reports"), help="output folder")
    parser.add_argument(
        "--max-distance",
        type=int,
        default=DEFAULT_MAX_DISTANCE,
        help="largest phash distance that counts as a twin",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Hash all images, group them and write the report."""
    args = parse_args(argv)
    records = hash_dataset(args.data)
    groups = find_groups([record.phash for record in records], args.max_distance)

    args.reports.mkdir(parents=True, exist_ok=True)
    report_file = args.reports / REPORT_NAME
    report_file.write_text(format_report(records, groups, args.max_distance), encoding="utf-8")

    leaked = count_images_with_twin(records, groups, "test", "train")
    num_test = sum(1 for record in records if record.split == "test")
    print(f"{leaked} of {num_test} test images have a near twin in train.")
    print(f"Report written to {report_file}")


if __name__ == "__main__":
    main()
