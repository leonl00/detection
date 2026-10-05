"""Checks on a dataset in YOLOv8 format.

A YOLOv8 dataset (as exported by Roboflow) looks like this::

    dataset/
    ├── data.yaml          paths and class names
    ├── train/images, train/labels
    ├── valid/images, valid/labels
    └── test/images,  test/labels

Every image ``abc.jpg`` has a label file ``abc.txt`` with one line per object::

    class_id x_center y_center width height

where all four coordinates are normalised to the range 0 to 1.

The checks answer six questions per split (train, valid, test):

1. How many images are there?
2. How often does each class occur?
3. Are there images without a label file, or label files without an image?
4. Are there invalid label lines (broken lines, unknown class IDs, values outside 0..1)?
5. How large are the images and the boxes?
6. All of the above as a Markdown report in ``reports/dataset_report.md``.

Usage (from the project root):
    python -m detection.dataset_check --data data/dataset --reports reports
"""

import argparse
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from matplotlib.figure import Figure
from PIL import Image, UnidentifiedImageError

SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Classes with fewer examples than this are flagged in the report (see docs/SPEC.md).
MIN_EXAMPLES_PER_CLASS = 30

# Box size groups by the share of the image area a box covers.
# Relative values are used because the images can have different resolutions.
SMALL_BOX_MAX_AREA = 0.01  # below 1 % of the image
LARGE_BOX_MIN_AREA = 0.10  # 10 % of the image or more

REPORT_NAME = "dataset_report.md"
CHART_NAME = "class_counts.png"


@dataclass
class Box:
    """One object from a label file, in normalised coordinates (0 to 1)."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass
class SplitSummary:
    """Collected check results for one split (train, valid or test).

    Attributes:
        name: Name of the split.
        num_images: Number of image files.
        class_counts: Number of boxes per class name.
        background_images: Images whose label file is empty (no objects).
        images_without_label: File names of images that have no label file.
        labels_without_image: File names of label files that have no image.
        label_problems: Human-readable descriptions of invalid label lines.
        unreadable_images: File names of images that could not be opened.
        image_sizes: How often each image size ``(width, height)`` occurs.
        box_areas: Area of every valid box as a share of the image area.
    """

    name: str
    num_images: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    background_images: int = 0
    images_without_label: list[str] = field(default_factory=list)
    labels_without_image: list[str] = field(default_factory=list)
    label_problems: list[str] = field(default_factory=list)
    unreadable_images: list[str] = field(default_factory=list)
    image_sizes: Counter = field(default_factory=Counter)
    box_areas: list[float] = field(default_factory=list)


# --- Structure and class names ----------------------------------------------


def check_dataset_structure(dataset_dir: Path) -> dict[str, int]:
    """Check that the dataset has the expected YOLOv8 layout.

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
        image_counts[split] = len(list_images(dataset_dir / split / "images"))
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


def list_images(folder: Path) -> list[Path]:
    """List all image files in a folder, sorted by name.

    Args:
        folder: Folder to search (not recursive).

    Returns:
        Paths of all files with an image suffix.

    Raises:
        FileNotFoundError: If the folder does not exist.
    """
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    return sorted(file for file in folder.iterdir() if file.suffix.lower() in IMAGE_SUFFIXES)


# --- Check 3: images and labels that belong together -------------------------


def find_unmatched_files(images_dir: Path, labels_dir: Path) -> tuple[list[str], list[str]]:
    """Find images without a label file and label files without an image.

    An image and its label belong together when they share the file name
    without suffix, for example ``abc.jpg`` and ``abc.txt``.

    Args:
        images_dir: Folder with the images.
        labels_dir: Folder with the ``.txt`` label files.

    Returns:
        Two sorted lists of file names: images without label, labels without image.

    Raises:
        FileNotFoundError: If one of the folders does not exist.
    """
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Label folder not found: {labels_dir}")

    images = {image.stem: image.name for image in list_images(images_dir)}
    labels = {label.stem: label.name for label in labels_dir.glob("*.txt")}

    images_without_label = sorted(images[stem] for stem in images.keys() - labels.keys())
    labels_without_image = sorted(labels[stem] for stem in labels.keys() - images.keys())
    return images_without_label, labels_without_image


# --- Check 4: valid label lines ----------------------------------------------


def parse_label_line(line: str, num_classes: int) -> Box:
    """Turn one line of a label file into a ``Box`` and check its values.

    Args:
        line: One line, for example ``"0 0.5 0.5 0.2 0.1"``.
        num_classes: Number of classes in ``data.yaml``; valid IDs are 0 to n-1.

    Returns:
        The parsed box.

    Raises:
        ValueError: If the line does not have five numbers, the class ID is
            unknown, or a coordinate is outside 0 to 1.
    """
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"expected 5 values, got {len(parts)}")

    try:
        class_id = int(parts[0])
        x_center, y_center, width, height = (float(part) for part in parts[1:])
    except ValueError as error:
        raise ValueError(f"not a number: {line.strip()!r}") from error

    if not 0 <= class_id < num_classes:
        raise ValueError(f"unknown class ID {class_id}")
    for name, value in [("x_center", x_center), ("y_center", y_center)]:
        if not 0 <= value <= 1:
            raise ValueError(f"{name} {value} is outside 0..1")
    for name, value in [("width", width), ("height", height)]:
        if not 0 < value <= 1:
            raise ValueError(f"{name} {value} must be greater than 0 and at most 1")

    return Box(class_id, x_center, y_center, width, height)


def read_label_file(label_file: Path, num_classes: int) -> tuple[list[Box], list[str]]:
    """Read all boxes from a label file and collect the invalid lines.

    Invalid lines do not stop the reading; they are reported so that all
    problems of a dataset are visible at once.

    Args:
        label_file: Path to the ``.txt`` file.
        num_classes: Number of classes in ``data.yaml``.

    Returns:
        The valid boxes and one description per invalid line.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not label_file.is_file():
        raise FileNotFoundError(f"Label file not found: {label_file}")

    boxes, problems = [], []
    lines = label_file.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            boxes.append(parse_label_line(line, num_classes))
        except ValueError as error:
            problems.append(f"{label_file.name}, line {line_number}: {error}")
    return boxes, problems


# --- Checks 2 and 5: class counts, image and box sizes ----------------------


def count_classes(boxes: list[Box], class_names: list[str]) -> dict[str, int]:
    """Count the boxes per class, including classes that do not occur.

    Args:
        boxes: Boxes with valid class IDs.
        class_names: Class names in the order of their IDs.

    Returns:
        Number of boxes per class name.
    """
    counts = Counter(box.class_id for box in boxes)
    return {name: counts[class_id] for class_id, name in enumerate(class_names)}


def read_image_size(image_file: Path) -> tuple[int, int]:
    """Read width and height of an image without loading all pixels.

    Args:
        image_file: Path to the image.

    Returns:
        ``(width, height)`` in pixels.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a readable image.
    """
    if not image_file.is_file():
        raise FileNotFoundError(f"Image not found: {image_file}")
    try:
        with Image.open(image_file) as image:
            return image.size
    except UnidentifiedImageError as error:
        raise ValueError(f"not a readable image: {image_file.name}") from error


def group_box_sizes(box_areas: list[float]) -> dict[str, int]:
    """Sort boxes into small, medium and large by their share of the image area.

    Args:
        box_areas: Box area divided by image area, one value per box.

    Returns:
        Number of boxes per group.
    """
    small = sum(1 for area in box_areas if area < SMALL_BOX_MAX_AREA)
    large = sum(1 for area in box_areas if area >= LARGE_BOX_MIN_AREA)
    return {"small": small, "medium": len(box_areas) - small - large, "large": large}


# --- Running all checks on a split ------------------------------------------


def summarize_split(dataset_dir: Path, split: str, class_names: list[str]) -> SplitSummary:
    """Run all checks on one split.

    Args:
        dataset_dir: Root folder of the dataset.
        split: Name of the split, for example ``"train"``.
        class_names: Class names from ``data.yaml``.

    Returns:
        The collected results.

    Raises:
        FileNotFoundError: If the image or label folder of the split is missing.
    """
    images_dir = dataset_dir / split / "images"
    labels_dir = dataset_dir / split / "labels"
    summary = SplitSummary(name=split)

    images = list_images(images_dir)
    summary.num_images = len(images)
    summary.images_without_label, summary.labels_without_image = find_unmatched_files(
        images_dir, labels_dir
    )

    for image_file in images:
        try:
            summary.image_sizes[read_image_size(image_file)] += 1
        except ValueError:
            summary.unreadable_images.append(image_file.name)

    all_boxes = []
    for label_file in sorted(labels_dir.glob("*.txt")):
        boxes, problems = read_label_file(label_file, len(class_names))
        all_boxes.extend(boxes)
        summary.label_problems.extend(problems)
        if not boxes and not problems:
            summary.background_images += 1

    summary.class_counts = count_classes(all_boxes, class_names)
    summary.box_areas = [box.width * box.height for box in all_boxes]
    return summary


def check_dataset(dataset_dir: Path) -> tuple[list[str], list[SplitSummary]]:
    """Run all checks on all splits of a dataset.

    Args:
        dataset_dir: Root folder of the dataset.

    Returns:
        The class names and one summary per split.

    Raises:
        FileNotFoundError: If the dataset layout is incomplete.
        ValueError: If ``data.yaml`` has no class names.
    """
    check_dataset_structure(dataset_dir)
    class_names = read_class_names(dataset_dir / "data.yaml")
    summaries = [summarize_split(dataset_dir, split, class_names) for split in SPLITS]
    return class_names, summaries


# --- Check 6: chart and report -----------------------------------------------


def plot_class_counts(summaries: list[SplitSummary], output_file: Path) -> None:
    """Save a bar chart of the boxes per class, one panel per split.

    Each split gets its own panel and x-axis, because train is much larger
    than valid and test and would otherwise hide their bars.

    Args:
        summaries: Results of ``summarize_split`` for each split.
        output_file: Path of the PNG file to write.

    Raises:
        ValueError: If ``summaries`` is empty.
    """
    if not summaries:
        raise ValueError("Need at least one split to plot.")

    surface, text, muted, grid, bar_color = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
    class_names = list(summaries[0].class_counts)

    # Figure is used directly instead of pyplot, so no window opens and no global state is kept.
    figure = Figure(figsize=(4 * len(summaries), 0.6 * len(class_names) + 1.6), facecolor=surface)
    axes = figure.subplots(1, len(summaries), sharey=True, squeeze=False)[0]

    for ax, summary in zip(axes, summaries, strict=True):
        counts = [summary.class_counts[name] for name in class_names]
        bars = ax.barh(class_names, counts, height=0.6, color=bar_color)
        ax.bar_label(bars, padding=3, color=muted, fontsize=9)

        ax.set_title(f"{summary.name} ({summary.num_images} images)", color=text, fontsize=11)
        ax.set_facecolor(surface)
        ax.invert_yaxis()  # first class at the top
        ax.set_xlim(0, max(max(counts), 1) * 1.2)  # room for the value labels
        ax.grid(axis="x", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=muted, labelsize=9, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    figure.suptitle("Boxes per class", color=text, fontsize=12, x=0.01, ha="left")
    figure.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=150)


def format_report(class_names: list[str], summaries: list[SplitSummary]) -> str:
    """Write all check results as a Markdown text.

    Args:
        class_names: Class names from ``data.yaml``.
        summaries: Results of ``summarize_split`` for each split.

    Returns:
        The report as Markdown.
    """
    lines = ["# Dataset report", "", "Created by `python -m detection.dataset_check`."]
    lines += _section_images(summaries)
    lines += _section_classes(class_names, summaries)
    lines += _section_unmatched(summaries)
    lines += _section_invalid(summaries)
    lines += _section_sizes(summaries)
    return "\n".join(lines) + "\n"


def _section_images(summaries: list[SplitSummary]) -> list[str]:
    """Report section 1: number of images per split."""
    return [
        "",
        "## 1. Images per split",
        "",
        *_table_header(summaries),
        _table_row("Images", [s.num_images for s in summaries]),
        _table_row("Background images (empty label)", [s.background_images for s in summaries]),
    ]


def _section_classes(class_names: list[str], summaries: list[SplitSummary]) -> list[str]:
    """Report section 2: boxes per class and split, with a hint on rare classes."""
    lines = ["", "## 2. Boxes per class", "", f"![Boxes per class]({CHART_NAME})", ""]
    lines += _table_header(summaries, extra_column="total")
    rare = []
    for name in class_names:
        counts = [s.class_counts[name] for s in summaries]
        lines.append(_table_row(f"`{name}`", [*counts, sum(counts)]))
        if sum(counts) < MIN_EXAMPLES_PER_CLASS:
            rare.append(name)
    if rare:
        lines += ["", f"Classes with fewer than {MIN_EXAMPLES_PER_CLASS} boxes: {', '.join(rare)}"]
    return lines


def _section_unmatched(summaries: list[SplitSummary]) -> list[str]:
    """Report section 3: images without label and labels without image."""
    lines = ["", "## 3. Images and labels without partner", "", *_table_header(summaries)]
    lines.append(
        _table_row("Images without label", [len(s.images_without_label) for s in summaries])
    )
    lines.append(
        _table_row("Labels without image", [len(s.labels_without_image) for s in summaries])
    )
    lines += _detail_list(
        [(s.name, s.images_without_label + s.labels_without_image) for s in summaries]
    )
    return lines


def _section_invalid(summaries: list[SplitSummary]) -> list[str]:
    """Report section 4: invalid label lines and unreadable images."""
    lines = ["", "## 4. Invalid labels and unreadable images", "", *_table_header(summaries)]
    lines.append(_table_row("Invalid label lines", [len(s.label_problems) for s in summaries]))
    lines.append(_table_row("Unreadable images", [len(s.unreadable_images) for s in summaries]))
    lines += _detail_list([(s.name, s.label_problems + s.unreadable_images) for s in summaries])
    return lines


def _section_sizes(summaries: list[SplitSummary]) -> list[str]:
    """Report section 5: most common image sizes and box size groups."""
    lines = ["", "## 5. Image and box sizes", "", "Most common image sizes (width x height):", ""]
    for s in summaries:
        common = ", ".join(f"{w}x{h} ({n})" for (w, h), n in s.image_sizes.most_common(3))
        lines.append(f"- **{s.name}**: {len(s.image_sizes)} different sizes; {common or '-'}")

    lines += [
        "",
        "Box size as a share of the image area "
        f"(small < {SMALL_BOX_MAX_AREA:.0%}, large >= {LARGE_BOX_MIN_AREA:.0%}):",
        "",
    ]
    lines += _table_header(summaries)
    groups = [group_box_sizes(s.box_areas) for s in summaries]
    for group in ("small", "medium", "large"):
        lines.append(_table_row(f"{group.capitalize()} boxes", [g[group] for g in groups]))
    medians = [f"{statistics.median(s.box_areas):.1%}" if s.box_areas else "-" for s in summaries]
    lines.append(_table_row("Median box area", medians))
    return lines


def _table_header(summaries: list[SplitSummary], extra_column: str = "") -> list[str]:
    """Return the two header lines of a Markdown table with one column per split."""
    columns = [s.name for s in summaries] + ([extra_column] if extra_column else [])
    return [_table_row("", columns), "|---|" + "---:|" * len(columns)]


def _table_row(label: str, values: list[object]) -> str:
    """Format one Markdown table row: ``| label | v1 | v2 | ... |``."""
    return f"| {label} | " + " | ".join(str(value) for value in values) + " |"


def _detail_list(entries: list[tuple[str, list[str]]], limit: int = 20) -> list[str]:
    """List file names per split under a table, at most ``limit`` per split."""
    lines = []
    for split, items in entries:
        if items:
            lines += ["", f"**{split}** ({len(items)}):", ""]
            lines += [f"- {item}" for item in items[:limit]]
            if len(items) > limit:
                lines.append(f"- ... and {len(items) - limit} more")
    return lines


# --- Command line ------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line options.

    Args:
        argv: Options to parse; ``None`` means the real command line.

    Returns:
        The parsed options ``data`` and ``reports``.
    """
    parser = argparse.ArgumentParser(description="Check a YOLOv8 dataset and write a report.")
    parser.add_argument("--data", type=Path, default=Path("data/dataset"), help="dataset folder")
    parser.add_argument("--reports", type=Path, default=Path("reports"), help="output folder")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Check the dataset, then write the chart and the report."""
    args = parse_args(argv)
    class_names, summaries = check_dataset(args.data)

    args.reports.mkdir(parents=True, exist_ok=True)
    plot_class_counts(summaries, args.reports / CHART_NAME)
    report_file = args.reports / REPORT_NAME
    report_file.write_text(format_report(class_names, summaries), encoding="utf-8")
    print(f"Report written to {report_file}")


if __name__ == "__main__":
    main()
