"""Tests for detection.evaluate.

The matching logic is tested with hand-made boxes. Evaluating a real model takes
time and needs trained weights, so the full run uses a fake YOLO class whose
``val`` and ``predict`` return fixed results in the same shape as Ultralytics.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from detection import evaluate as evaluate_module
from detection.evaluate import (
    ClassMetrics,
    ImageErrors,
    PixelBox,
    box_iou,
    collect_metrics,
    count_errors,
    draw_errors,
    evaluate,
    format_metrics,
    format_worst,
    main,
    rank_worst,
    read_true_boxes,
)

CLASS_NAMES = ["spaghetti", "warping"]


def errors(name: str, missed: int, false_alarms: int) -> ImageErrors:
    """Create an ``ImageErrors`` without boxes for ranking tests."""
    return ImageErrors(Path(name), [], [], missed, false_alarms)


# --- box_iou -----------------------------------------------------------------


def test_box_iou_identical_boxes() -> None:
    box = PixelBox(0, 0, 0, 10, 10)

    assert box_iou(box, box) == 1.0


def test_box_iou_half_overlap() -> None:
    # Overlap 5x10 = 50, union 100 + 100 - 50 = 150.
    assert box_iou(PixelBox(0, 0, 0, 10, 10), PixelBox(0, 5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_box_iou_no_overlap_and_empty_box() -> None:
    assert box_iou(PixelBox(0, 0, 0, 10, 10), PixelBox(0, 20, 20, 30, 30)) == 0.0
    assert box_iou(PixelBox(0, 5, 5, 5, 5), PixelBox(0, 5, 5, 5, 5)) == 0.0


# --- count_errors ------------------------------------------------------------


def test_count_errors_perfect_prediction() -> None:
    truth = [PixelBox(0, 0, 0, 10, 10)]

    assert count_errors(truth, [PixelBox(0, 0, 0, 10, 10, 0.9)], 0.5) == (0, 0)


def test_count_errors_wrong_class_is_miss_and_false_alarm() -> None:
    truth = [PixelBox(0, 0, 0, 10, 10)]

    assert count_errors(truth, [PixelBox(1, 0, 0, 10, 10, 0.9)], 0.5) == (1, 1)


def test_count_errors_low_overlap_is_miss_and_false_alarm() -> None:
    truth = [PixelBox(0, 0, 0, 10, 10)]

    assert count_errors(truth, [PixelBox(0, 5, 0, 15, 10, 0.9)], 0.5) == (1, 1)


def test_count_errors_true_box_matched_only_once() -> None:
    truth = [PixelBox(0, 0, 0, 10, 10)]
    duplicates = [PixelBox(0, 0, 0, 10, 10, 0.9), PixelBox(0, 0, 0, 10, 10, 0.8)]

    assert count_errors(truth, duplicates, 0.5) == (0, 1)


def test_count_errors_background_image() -> None:
    assert count_errors([], [PixelBox(0, 0, 0, 10, 10, 0.4)], 0.5) == (0, 1)
    assert count_errors([], [], 0.5) == (0, 0)


def test_count_errors_bad_threshold_raises() -> None:
    with pytest.raises(ValueError, match="iou_threshold"):
        count_errors([], [], 0)


# --- rank_worst --------------------------------------------------------------


def test_rank_worst_orders_by_errors_then_missed() -> None:
    results = [
        errors("a.jpg", missed=0, false_alarms=2),
        errors("b.jpg", missed=2, false_alarms=0),
        errors("c.jpg", missed=0, false_alarms=0),
        errors("d.jpg", missed=3, false_alarms=1),
    ]

    worst = rank_worst(results, count=10)

    assert [result.image.name for result in worst] == ["d.jpg", "b.jpg", "a.jpg"]


def test_rank_worst_limits_count() -> None:
    results = [errors(f"{index}.jpg", 1, 0) for index in range(5)]

    assert len(rank_worst(results, count=3)) == 3


# --- read_true_boxes ---------------------------------------------------------


def test_read_true_boxes_converts_to_pixels(tmp_path: Path) -> None:
    label = tmp_path / "a.txt"
    label.write_text("1 0.5 0.5 0.2 0.4\n", encoding="utf-8")

    (box,) = read_true_boxes(label, (100, 50), num_classes=2)

    assert (box.class_id, box.x1, box.y1, box.x2, box.y2) == pytest.approx((1, 40, 15, 60, 35))


def test_read_true_boxes_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_true_boxes(tmp_path / "missing.txt", (100, 100), num_classes=2)


# --- collect_metrics and the reports -----------------------------------------


def fake_metrics() -> SimpleNamespace:
    """Build an object shaped like the result of ``YOLO.val()``."""
    per_class = {0: (0.8, 0.7, 0.75, 0.5), 1: (0.6, 0.4, 0.45, 0.2)}
    box = SimpleNamespace(
        mp=0.7,
        mr=0.55,
        map50=0.6,
        map=0.35,
        ap_class_index=[0, 1],
        class_result=lambda index: per_class[index],
    )
    return SimpleNamespace(box=box, names={0: "spaghetti", 1: "warping"}, nt_per_class=[12, 8])


def test_collect_metrics_has_overall_and_class_rows() -> None:
    rows = collect_metrics(fake_metrics())

    assert rows[0] == ClassMetrics("all", 20, 0.7, 0.55, 0.6, 0.35)
    assert rows[2] == ClassMetrics("warping", 8, 0.6, 0.4, 0.45, 0.2)


def test_format_metrics_writes_table() -> None:
    rows = collect_metrics(fake_metrics())

    report = format_metrics(rows, Path("runs/x/best.pt"), Path("data/data.yaml"), "test")

    assert "| **all** | 20 | 0.700 | 0.550 | 0.600 | 0.350 |" in report
    assert "| `warping` | 8 | 0.600 | 0.400 | 0.450 | 0.200 |" in report


def test_format_worst_links_images() -> None:
    report = format_worst([errors("abc.jpg", 2, 1)], conf=0.25)

    assert "| 1 | [abc.jpg](worst/01_abc.jpg) | 2 | 1 |" in report


def test_draw_errors_writes_image(tmp_path: Path) -> None:
    image = tmp_path / "a.png"
    Image.new("RGB", (64, 48), "white").save(image)
    result = ImageErrors(
        image, [PixelBox(0, 5, 5, 30, 30)], [PixelBox(1, 10, 10, 40, 40, 0.7)], 1, 1
    )

    draw_errors(result, CLASS_NAMES, tmp_path / "out" / "a.jpg")

    with Image.open(tmp_path / "out" / "a.jpg") as drawn:
        assert drawn.size == (64, 48)


# --- evaluate and main with a fake model -------------------------------------


class FakeList(list):
    """A list with ``tolist()``, like the tensors in an Ultralytics result."""

    def tolist(self) -> list:
        return list(self)


class FakeYOLO:
    """Stands in for ``ultralytics.YOLO``: fixed metrics, one box per image."""

    def __init__(self, model: str) -> None:
        self.model = model

    def val(self, **kwargs: object) -> SimpleNamespace:
        Path(kwargs["project"], kwargs["name"]).mkdir(parents=True, exist_ok=True)
        return fake_metrics()

    def predict(self, **kwargs: object) -> list[SimpleNamespace]:
        boxes = SimpleNamespace(
            xyxy=FakeList([[0.0, 0.0, 8.0, 8.0]]), cls=FakeList([1.0]), conf=FakeList([0.9])
        )
        return [SimpleNamespace(boxes=boxes)]


def make_dataset(root: Path) -> Path:
    """Create a test split with two images: one matched, one missed."""
    for subfolder in ("images", "labels"):
        (root / "test" / subfolder).mkdir(parents=True)
    (root / "data.yaml").write_text(f"names: {CLASS_NAMES}\n", encoding="utf-8")
    for name, label in [("hit", "1 0.125 0.125 0.25 0.25\n"), ("miss", "0 0.75 0.75 0.2 0.2\n")]:
        Image.new("RGB", (32, 32), "white").save(root / "test" / "images" / f"{name}.png")
        (root / "test" / "labels" / f"{name}.txt").write_text(label, encoding="utf-8")
    return root / "data.yaml"


def test_evaluate_writes_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluate_module, "YOLO", FakeYOLO)
    data_yaml = make_dataset(tmp_path / "dataset")
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"")
    output = tmp_path / "reports" / "baseline"

    rows = evaluate(weights, data_yaml, output, "test", conf=0.25, imgsz=32)

    assert rows[0].map50 == 0.6
    assert json.loads((output / "metrics.json").read_text(encoding="utf-8"))[0]["name"] == "all"
    worst = (output / "worst_predictions.md").read_text(encoding="utf-8")
    assert "| 1 | [miss.png](worst/01_miss.jpg) | 1 | 1 |" in worst
    assert "hit.png" not in worst
    assert (output / "worst" / "01_miss.jpg").is_file()


def test_evaluate_missing_model_raises(tmp_path: Path) -> None:
    data_yaml = make_dataset(tmp_path / "dataset")

    with pytest.raises(FileNotFoundError, match="weights"):
        evaluate(tmp_path / "missing.pt", data_yaml, tmp_path / "out", "test", 0.25, 32)


def test_main_prints_overall_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(evaluate_module, "YOLO", FakeYOLO)
    data_yaml = make_dataset(tmp_path / "dataset")
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"")

    main(["--model", str(weights), "--data", str(data_yaml), "--output", str(tmp_path / "out")])

    assert "mAP50 0.600" in capsys.readouterr().out
