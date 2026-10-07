"""Evaluate a trained model on the test split and find its worst predictions.

The evaluation writes everything into one report folder:

- ``metrics.md`` and ``metrics.json``: mAP50, mAP50-95, precision and recall,
  overall and per class
- ``confusion_matrix.png``, ``BoxPR_curve.png`` and further plots, drawn by
  Ultralytics during validation
- ``worst/``: the images with the most errors, with the true boxes in green and
  the predicted boxes in red, plus ``worst_predictions.md`` as an overview

An error is a missed defect (a true box without a matching prediction, "false
negative") or a false alarm (a prediction without a matching true box, "false
positive"). A prediction matches a true box when both have the same class and
their overlap (IoU) is at least 0.5.

Usage (from the project root):
    python -m detection.evaluate --model runs/baseline/weights/best.pt \\
        --data data/dataset/data.yaml --output reports/baseline
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO

from detection.dataset_check import list_images, read_class_names, read_label_file

# A prediction counts as a hit when it overlaps a true box of the same class by this much.
IOU_THRESHOLD = 0.5

# Confidence from which a prediction counts for the error analysis. 0.25 is the
# Ultralytics default; the threshold for the application is chosen in stage 8.
DEFAULT_CONF = 0.25

NUM_WORST = 20
WORST_DIR_NAME = "worst"

TRUE_COLOR = (0, 200, 0)
PREDICTED_COLOR = (230, 0, 0)


@dataclass
class PixelBox:
    """A box in pixel coordinates of the original image.

    Attributes:
        class_id: Class of the box.
        x1: Left edge.
        y1: Top edge.
        x2: Right edge.
        y2: Bottom edge.
        confidence: Confidence of a prediction; 1.0 for a true box.
    """

    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0


@dataclass
class ImageErrors:
    """True and predicted boxes of one image and the errors between them.

    Attributes:
        image: Path to the image.
        true_boxes: Boxes from the label file.
        predicted_boxes: Boxes predicted by the model.
        missed: Number of true boxes without a matching prediction.
        false_alarms: Number of predictions without a matching true box.
    """

    image: Path
    true_boxes: list[PixelBox]
    predicted_boxes: list[PixelBox]
    missed: int
    false_alarms: int


@dataclass
class ClassMetrics:
    """Detection metrics of one class, or of all classes together.

    Attributes:
        name: Class name, or ``"all"`` for the average over all classes.
        boxes: Number of true boxes in the evaluated split.
        precision: Share of predictions that are correct.
        recall: Share of true boxes that are found.
        map50: Average precision at an IoU of 0.5.
        map50_95: Average precision averaged over IoU 0.5 to 0.95.
    """

    name: str
    boxes: int
    precision: float
    recall: float
    map50: float
    map50_95: float


# --- Matching predictions to true boxes --------------------------------------


def box_iou(a: PixelBox, b: PixelBox) -> float:
    """Compute the overlap of two boxes as intersection over union (IoU).

    Args:
        a: First box.
        b: Second box.

    Returns:
        A value from 0 (no overlap) to 1 (identical boxes).
    """
    width = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    height = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    intersection = width * height
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def count_errors(
    true_boxes: list[PixelBox], predicted_boxes: list[PixelBox], iou_threshold: float
) -> tuple[int, int]:
    """Match predictions to true boxes and count what is left over.

    Predictions are handled from the most to the least confident. Each one takes
    the still unmatched true box of the same class that it overlaps most, if the
    overlap reaches ``iou_threshold``. A true box can be matched only once.

    Args:
        true_boxes: Boxes from the label file.
        predicted_boxes: Boxes predicted by the model.
        iou_threshold: Smallest IoU that counts as a hit.

    Returns:
        Number of missed true boxes and number of false alarms.

    Raises:
        ValueError: If ``iou_threshold`` is not between 0 and 1.
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}")

    unmatched = list(true_boxes)
    false_alarms = 0
    for prediction in sorted(predicted_boxes, key=lambda box: -box.confidence):
        candidates = [box for box in unmatched if box.class_id == prediction.class_id]
        best = max(candidates, key=lambda box: box_iou(box, prediction), default=None)
        if best is not None and box_iou(best, prediction) >= iou_threshold:
            unmatched.remove(best)
        else:
            false_alarms += 1
    return len(unmatched), false_alarms


def rank_worst(results: list[ImageErrors], count: int) -> list[ImageErrors]:
    """Pick the images with the most errors.

    Images with more errors come first. On a tie, the image with more missed
    defects comes first, because a missed defect costs more than a false alarm.
    The file name decides the remaining ties, so the order is reproducible.

    Args:
        results: Errors per image.
        count: How many images to return.

    Returns:
        At most ``count`` images, worst first. Images without errors are left out.
    """
    with_errors = [result for result in results if result.missed + result.false_alarms > 0]
    ordered = sorted(
        with_errors,
        key=lambda result: (
            -(result.missed + result.false_alarms),
            -result.missed,
            result.image.name,
        ),
    )
    return ordered[:count]


# --- Reading true boxes and running the model --------------------------------


def read_true_boxes(
    label_file: Path, image_size: tuple[int, int], num_classes: int
) -> list[PixelBox]:
    """Read the boxes of a label file and convert them to pixel coordinates.

    Invalid lines (for example polygons) are skipped, as in ``dataset_check``.

    Args:
        label_file: YOLO label file with normalised coordinates.
        image_size: ``(width, height)`` of the image in pixels.
        num_classes: Number of classes in ``data.yaml``.

    Returns:
        The boxes in pixel coordinates.

    Raises:
        FileNotFoundError: If the label file does not exist.
    """
    width, height = image_size
    boxes, _ = read_label_file(label_file, num_classes)
    return [
        PixelBox(
            box.class_id,
            (box.x_center - box.width / 2) * width,
            (box.y_center - box.height / 2) * height,
            (box.x_center + box.width / 2) * width,
            (box.y_center + box.height / 2) * height,
        )
        for box in boxes
    ]


def predict_boxes(model: YOLO, image: Path, conf: float, imgsz: int) -> list[PixelBox]:
    """Run the model on one image.

    Args:
        model: The loaded model.
        image: Path to the image.
        conf: Smallest confidence a prediction must have.
        imgsz: Image size used during training.

    Returns:
        The predicted boxes in pixel coordinates of the original image.
    """
    result = model.predict(source=str(image), conf=conf, imgsz=imgsz, verbose=False)[0]
    return [
        PixelBox(int(class_id), *map(float, xyxy), confidence=float(confidence))
        for xyxy, class_id, confidence in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.cls.tolist(),
            result.boxes.conf.tolist(),
            strict=True,
        )
    ]


def find_errors(
    model: YOLO, images_dir: Path, labels_dir: Path, num_classes: int, conf: float, imgsz: int
) -> list[ImageErrors]:
    """Compare predictions and true boxes for every image of a split.

    Args:
        model: The loaded model.
        images_dir: Folder with the images of the split.
        labels_dir: Folder with the label files of the split.
        num_classes: Number of classes in ``data.yaml``.
        conf: Smallest confidence a prediction must have.
        imgsz: Image size used during training.

    Returns:
        The errors of every image.

    Raises:
        FileNotFoundError: If a folder or a label file is missing.
    """
    results = []
    for image in list_images(images_dir):
        with Image.open(image) as opened:
            size = opened.size
        true_boxes = read_true_boxes(labels_dir / f"{image.stem}.txt", size, num_classes)
        predicted = predict_boxes(model, image, conf, imgsz)
        missed, false_alarms = count_errors(true_boxes, predicted, IOU_THRESHOLD)
        results.append(ImageErrors(image, true_boxes, predicted, missed, false_alarms))
    return results


# --- Metrics -----------------------------------------------------------------


def collect_metrics(metrics: object) -> list[ClassMetrics]:
    """Turn the result of ``YOLO.val()`` into one row per class plus an overall row.

    Args:
        metrics: The object returned by ``YOLO.val()``; it provides ``box``,
            ``names`` and ``nt_per_class``.

    Returns:
        The overall row (name ``"all"``) followed by one row per class that
        occurs in the evaluated split.
    """
    box = metrics.box
    rows = [
        ClassMetrics(
            "all",
            int(sum(metrics.nt_per_class)),
            float(box.mp),
            float(box.mr),
            float(box.map50),
            float(box.map),
        )
    ]
    for index, class_id in enumerate(box.ap_class_index):
        precision, recall, map50, map50_95 = box.class_result(index)
        rows.append(
            ClassMetrics(
                metrics.names[int(class_id)],
                int(metrics.nt_per_class[int(class_id)]),
                float(precision),
                float(recall),
                float(map50),
                float(map50_95),
            )
        )
    return rows


def format_metrics(rows: list[ClassMetrics], model: Path, data_yaml: Path, split: str) -> str:
    """Write the metrics as a Markdown table.

    Args:
        rows: Result of ``collect_metrics``.
        model: Path of the evaluated weights.
        data_yaml: Dataset the model was evaluated on.
        split: Name of the evaluated split.

    Returns:
        The report as Markdown.
    """
    lines = [
        "# Evaluation",
        "",
        f"Model `{model.as_posix()}` on the `{split}` split of `{data_yaml.as_posix()}`.",
        "",
        "| Class | Boxes | Precision | Recall | mAP50 | mAP50-95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        name = "**all**" if row.name == "all" else f"`{row.name}`"
        lines.append(
            f"| {name} | {row.boxes} | {row.precision:.3f} | {row.recall:.3f} "
            f"| {row.map50:.3f} | {row.map50_95:.3f} |"
        )
    lines += [
        "",
        "![Confusion matrix](confusion_matrix_normalized.png)",
        "",
        "![Precision-recall curve](BoxPR_curve.png)",
    ]
    return "\n".join(lines) + "\n"


# --- Drawing the worst predictions -------------------------------------------


def draw_errors(result: ImageErrors, class_names: list[str], output_file: Path) -> None:
    """Save an image with true boxes in green and predicted boxes in red.

    Args:
        result: The image and its boxes.
        class_names: Class names in the order of their IDs.
        output_file: Path of the image to write.
    """
    with Image.open(result.image) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = max(2, round(max(image.size) / 300))

    for box in result.true_boxes:
        draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline=TRUE_COLOR, width=line_width)
        draw.text((box.x1 + 4, box.y1 + 2), class_names[box.class_id], fill=TRUE_COLOR)
    for box in result.predicted_boxes:
        label = f"{class_names[box.class_id]} {box.confidence:.2f}"
        draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline=PREDICTED_COLOR, width=line_width)
        draw.text((box.x1 + 4, box.y2 - 14), label, fill=PREDICTED_COLOR)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_file)


def format_worst(worst: list[ImageErrors], conf: float) -> str:
    """List the worst images as a Markdown table that links the drawn images.

    Args:
        worst: Result of ``rank_worst``.
        conf: Confidence threshold used for the predictions.

    Returns:
        The overview as Markdown.
    """
    lines = [
        "# Worst predictions",
        "",
        f"Predictions with confidence >= {conf}, matched at IoU >= {IOU_THRESHOLD}.",
        "Green: true boxes. Red: predicted boxes with confidence.",
        "",
        "| Rank | Image | Missed | False alarms |",
        "|---:|---|---:|---:|",
    ]
    for rank, result in enumerate(worst, start=1):
        link = f"[{result.image.name}]({WORST_DIR_NAME}/{worst_file_name(rank, result)})"
        lines.append(f"| {rank} | {link} | {result.missed} | {result.false_alarms} |")
    return "\n".join(lines) + "\n"


def worst_file_name(rank: int, result: ImageErrors) -> str:
    """Return the file name of a drawn worst image, for example ``01_abc.jpg``."""
    return f"{rank:02d}_{result.image.stem}.jpg"


# --- Running the evaluation --------------------------------------------------


def evaluate(
    model_path: Path, data_yaml: Path, output_dir: Path, split: str, conf: float, imgsz: int
) -> list[ClassMetrics]:
    """Evaluate a model and write metrics, plots and the worst predictions.

    Args:
        model_path: Trained weights, for example ``runs/baseline/weights/best.pt``.
        data_yaml: The dataset's ``data.yaml``.
        output_dir: Folder for the report.
        split: Split to evaluate, usually ``"test"``.
        conf: Confidence threshold for the error analysis.
        imgsz: Image size used during training.

    Returns:
        The metrics, overall row first.

    Raises:
        FileNotFoundError: If the weights, the dataset or the split is missing.
    """
    if not model_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    class_names = read_class_names(data_yaml)
    split_dir = data_yaml.parent / ("valid" if split == "val" else split)

    model = YOLO(str(model_path))
    output_dir = output_dir.resolve()
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        # An absolute path is required: Ultralytics puts relative paths into its own folder.
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
        plots=True,
    )
    rows = collect_metrics(metrics)
    (output_dir / "metrics.json").write_text(
        json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8"
    )
    (output_dir / "metrics.md").write_text(
        format_metrics(rows, model_path, data_yaml, split), encoding="utf-8"
    )

    errors = find_errors(
        model, split_dir / "images", split_dir / "labels", len(class_names), conf, imgsz
    )
    worst = rank_worst(errors, NUM_WORST)
    for rank, result in enumerate(worst, start=1):
        draw_errors(
            result, class_names, output_dir / WORST_DIR_NAME / worst_file_name(rank, result)
        )
    (output_dir / "worst_predictions.md").write_text(format_worst(worst, conf), encoding="utf-8")
    return rows


# --- Command line ------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line options.

    Args:
        argv: Options to parse; ``None`` means the real command line.

    Returns:
        The parsed options ``model``, ``data``, ``output``, ``split``, ``conf`` and ``imgsz``.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a dataset split.")
    parser.add_argument("--model", type=Path, required=True, help="trained weights (.pt)")
    parser.add_argument("--data", type=Path, required=True, help="data.yaml of the dataset")
    parser.add_argument("--output", type=Path, required=True, help="folder for the report")
    parser.add_argument("--split", default="test", choices=["val", "test"], help="split to use")
    parser.add_argument(
        "--conf", type=float, default=DEFAULT_CONF, help="confidence for the error analysis"
    )
    parser.add_argument("--imgsz", type=int, default=640, help="image size used in training")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Evaluate the model and print the overall metrics."""
    args = parse_args(argv)
    rows = evaluate(args.model, args.data, args.output, args.split, args.conf, args.imgsz)
    overall = rows[0]
    print(
        f"mAP50 {overall.map50:.3f}, mAP50-95 {overall.map50_95:.3f}, "
        f"precision {overall.precision:.3f}, recall {overall.recall:.3f}"
    )
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
