"""Tests for detection.dataset_check.

The tests build a tiny fake dataset in ``tmp_path`` with real, very small images
created by Pillow, so they need neither the real dataset nor the internet.
"""

from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from detection.dataset_check import (
    CHART_NAME,
    REPORT_NAME,
    SPLITS,
    Box,
    SplitSummary,
    check_dataset,
    check_dataset_structure,
    count_classes,
    find_unmatched_files,
    format_report,
    group_box_sizes,
    list_images,
    main,
    parse_label_line,
    plot_class_counts,
    read_class_names,
    read_image_size,
    read_label_file,
    summarize_split,
)

CLASS_NAMES = ["spaghetti", "stringing"]


def make_image(path: Path, size: tuple[int, int] = (64, 48)) -> None:
    """Save a small grey JPEG image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(128, 128, 128)).save(path)


def make_dataset(root: Path, images_per_split: int = 2) -> Path:
    """Create a minimal valid YOLOv8 dataset: one spaghetti box per image."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(f"names: {CLASS_NAMES}\n", encoding="utf-8")
    for split in SPLITS:
        (root / split / "labels").mkdir(parents=True)
        (root / split / "images").mkdir(parents=True)
        for index in range(images_per_split):
            make_image(root / split / "images" / f"img_{index}.jpg")
            label = root / split / "labels" / f"img_{index}.txt"
            label.write_text("0 0.5 0.5 0.2 0.1\n", encoding="utf-8")
    return root


# --- check_dataset_structure -------------------------------------------------


def test_structure_counts_images_per_split(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path, images_per_split=3)

    assert check_dataset_structure(dataset) == {"train": 3, "valid": 3, "test": 3}


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


# --- read_class_names --------------------------------------------------------


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


# --- list_images -------------------------------------------------------------


def test_list_images_ignores_other_files_and_sorts(tmp_path: Path) -> None:
    for name in ["b.jpg", "a.PNG", "notes.txt"]:
        (tmp_path / name).touch()

    assert [path.name for path in list_images(tmp_path)] == ["a.PNG", "b.jpg"]


def test_list_images_missing_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Image folder"):
        list_images(tmp_path / "missing")


# --- find_unmatched_files ----------------------------------------------------


def test_unmatched_files_are_found(tmp_path: Path) -> None:
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for name in ["both.jpg", "only_image.jpg"]:
        (images / name).touch()
    for name in ["both.txt", "only_label.txt"]:
        (labels / name).touch()

    assert find_unmatched_files(images, labels) == (["only_image.jpg"], ["only_label.txt"])


def test_unmatched_files_missing_label_folder_raises(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()

    with pytest.raises(FileNotFoundError, match="Label folder"):
        find_unmatched_files(tmp_path / "images", tmp_path / "labels")


# --- parse_label_line --------------------------------------------------------


def test_valid_label_line_is_parsed() -> None:
    assert parse_label_line("1 0.5 0.25 0.2 0.1", num_classes=2) == Box(1, 0.5, 0.25, 0.2, 0.1)


@pytest.mark.parametrize(
    ("line", "reason"),
    [
        ("0 0.5 0.5 0.2", "expected 5 values"),
        ("0 0.1 0.1 0.2 0.2 0.3 0.3", "expected 5 values"),  # polygon instead of box
        ("a 0.5 0.5 0.2 0.1", "not a number"),
        ("0.5 0.5 0.5 0.2 0.1", "not a number"),
        ("2 0.5 0.5 0.2 0.1", "unknown class ID 2"),
        ("-1 0.5 0.5 0.2 0.1", "unknown class ID -1"),
        ("0 1.2 0.5 0.2 0.1", "x_center"),
        ("0 0.5 -0.1 0.2 0.1", "y_center"),
        ("0 0.5 0.5 0 0.1", "width"),
        ("0 0.5 0.5 0.2 1.5", "height"),
    ],
)
def test_invalid_label_line_raises(line: str, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        parse_label_line(line, num_classes=2)


# --- read_label_file ---------------------------------------------------------


def test_label_file_collects_boxes_and_problems(tmp_path: Path) -> None:
    label = tmp_path / "img.txt"
    label.write_text(
        "0 0.5 0.5 0.2 0.1\n\n5 0.5 0.5 0.2 0.1\n1 0.3 0.3 0.1 0.1\n", encoding="utf-8"
    )

    boxes, problems = read_label_file(label, num_classes=2)

    assert [box.class_id for box in boxes] == [0, 1]
    assert problems == ["img.txt, line 3: unknown class ID 5"]


def test_empty_label_file_has_no_boxes(tmp_path: Path) -> None:
    label = tmp_path / "img.txt"
    label.write_text("", encoding="utf-8")

    assert read_label_file(label, num_classes=2) == ([], [])


def test_missing_label_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_label_file(tmp_path / "missing.txt", num_classes=2)


# --- count_classes -----------------------------------------------------------


def test_count_classes_includes_classes_without_boxes() -> None:
    boxes = [Box(0, 0.5, 0.5, 0.1, 0.1), Box(0, 0.5, 0.5, 0.1, 0.1)]

    assert count_classes(boxes, CLASS_NAMES) == {"spaghetti": 2, "stringing": 0}


def test_count_classes_without_boxes() -> None:
    assert count_classes([], CLASS_NAMES) == {"spaghetti": 0, "stringing": 0}


# --- read_image_size ---------------------------------------------------------


def test_image_size_is_read(tmp_path: Path) -> None:
    make_image(tmp_path / "img.jpg", size=(80, 60))

    assert read_image_size(tmp_path / "img.jpg") == (80, 60)


def test_broken_image_raises_value_error(tmp_path: Path) -> None:
    broken = tmp_path / "broken.jpg"
    broken.write_text("this is not an image", encoding="utf-8")

    with pytest.raises(ValueError, match="not a readable image"):
        read_image_size(broken)


def test_missing_image_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_image_size(tmp_path / "missing.jpg")


# --- group_box_sizes ---------------------------------------------------------


def test_box_sizes_are_grouped_at_the_limits() -> None:
    areas = [0.005, 0.01, 0.05, 0.0999, 0.10, 0.5]

    assert group_box_sizes(areas) == {"small": 1, "medium": 3, "large": 2}


def test_no_boxes_give_empty_groups() -> None:
    assert group_box_sizes([]) == {"small": 0, "medium": 0, "large": 0}


# --- summarize_split ---------------------------------------------------------


def test_summarize_split_collects_all_results(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path, images_per_split=2)
    train = dataset / "train"
    make_image(train / "images" / "no_label.jpg", size=(32, 32))
    (train / "labels" / "no_image.txt").write_text("1 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    (train / "labels" / "img_1.txt").write_text("", encoding="utf-8")  # background image
    (train / "labels" / "img_0.txt").write_text(
        "0 0.5 0.5 0.2 0.1\n9 0.5 0.5 0.2 0.1\n", encoding="utf-8"
    )

    summary = summarize_split(dataset, "train", CLASS_NAMES)

    assert summary.num_images == 3
    assert summary.class_counts == {"spaghetti": 1, "stringing": 1}
    assert summary.background_images == 1
    assert summary.images_without_label == ["no_label.jpg"]
    assert summary.labels_without_image == ["no_image.txt"]
    assert summary.label_problems == ["img_0.txt, line 2: unknown class ID 9"]
    assert summary.image_sizes == Counter({(64, 48): 2, (32, 32): 1})
    assert summary.box_areas == pytest.approx([0.02, 0.25])


def test_summarize_split_reports_unreadable_images(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path, images_per_split=1)
    (dataset / "train" / "images" / "broken.jpg").write_text("no image", encoding="utf-8")

    summary = summarize_split(dataset, "train", CLASS_NAMES)

    assert summary.unreadable_images == ["broken.jpg"]


def test_summarize_split_missing_split_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        summarize_split(tmp_path, "train", CLASS_NAMES)


# --- check_dataset -----------------------------------------------------------


def test_check_dataset_returns_one_summary_per_split(tmp_path: Path) -> None:
    class_names, summaries = check_dataset(make_dataset(tmp_path))

    assert class_names == CLASS_NAMES
    assert [s.name for s in summaries] == list(SPLITS)
    assert all(s.class_counts["spaghetti"] == 2 for s in summaries)


def test_check_dataset_incomplete_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        check_dataset(tmp_path)


# --- plot_class_counts and format_report -------------------------------------


def make_summaries() -> list[SplitSummary]:
    """Summaries with hand-picked numbers for chart and report tests."""
    return [
        SplitSummary(
            name="train",
            num_images=10,
            class_counts={"spaghetti": 40, "stringing": 5},
            image_sizes=Counter({(640, 640): 10}),
            box_areas=[0.005, 0.05, 0.2],
        ),
        SplitSummary(
            name="test",
            num_images=2,
            class_counts={"spaghetti": 3, "stringing": 0},
            label_problems=["a.txt, line 1: unknown class ID 7"],
        ),
    ]


def test_plot_writes_png(tmp_path: Path) -> None:
    output = tmp_path / "charts" / CHART_NAME

    plot_class_counts(make_summaries(), output)

    assert output.read_bytes().startswith(b"\x89PNG")


def test_plot_without_summaries_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one split"):
        plot_class_counts([], tmp_path / CHART_NAME)


def test_report_contains_counts_and_hints() -> None:
    report = format_report(["spaghetti", "stringing"], make_summaries())

    assert "| `spaghetti` | 40 | 3 | 43 |" in report
    assert "Classes with fewer than 30 boxes: stringing" in report
    assert "a.txt, line 1: unknown class ID 7" in report
    assert "640x640 (10)" in report
    assert "| Small boxes | 1 | 0 |" in report
    assert "| Median box area | 5.0% | - |" in report


def test_report_without_rare_classes_has_no_hint() -> None:
    summaries = make_summaries()
    summaries[0].class_counts["stringing"] = 50

    assert "fewer than" not in format_report(["spaghetti", "stringing"], summaries)


# --- main --------------------------------------------------------------------


def test_main_writes_report_and_chart(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "dataset")
    reports = tmp_path / "reports"

    main(["--data", str(dataset), "--reports", str(reports)])

    assert "# Dataset report" in (reports / REPORT_NAME).read_text(encoding="utf-8")
    assert (reports / CHART_NAME).is_file()


def test_main_with_missing_dataset_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        main(["--data", str(tmp_path / "missing"), "--reports", str(tmp_path)])
