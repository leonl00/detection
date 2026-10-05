"""Tests for detection.train.

A real training takes minutes to hours, so the tests replace the YOLO class by
a small fake that only records how it was called. This checks that the settings
from the configuration reach Ultralytics, without training anything.
"""

from pathlib import Path

import pytest
import yaml

from detection import train as train_module
from detection.config import Config
from detection.train import (
    RUN_INFO_NAME,
    check_data_yaml,
    collect_run_info,
    main,
    prepare_run_dir,
    train,
    write_run_info,
)


class FakeYOLO:
    """Stands in for ``ultralytics.YOLO`` and remembers the call arguments."""

    calls: list[dict] = []

    def __init__(self, model: str) -> None:
        self.model = model

    def train(self, **kwargs: object) -> None:
        FakeYOLO.calls.append({"model": self.model, **kwargs})


@pytest.fixture
def fake_yolo(monkeypatch: pytest.MonkeyPatch) -> type[FakeYOLO]:
    """Replace YOLO in the train module for one test."""
    FakeYOLO.calls = []
    monkeypatch.setattr(train_module, "YOLO", FakeYOLO)
    return FakeYOLO


def make_config(tmp_path: Path, name: str = "test_run") -> Config:
    """Create a config whose data.yaml exists in ``tmp_path``."""
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names: [a]\n", encoding="utf-8")
    return Config(name, data_yaml, "yolov8n.pt", epochs=3, imgsz=320, seed=7)


# --- check_data_yaml ---------------------------------------------------------


def test_check_data_yaml_accepts_existing_file(tmp_path: Path) -> None:
    check_data_yaml(make_config(tmp_path).data_yaml)


def test_check_data_yaml_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        check_data_yaml(tmp_path / "data.yaml")


# --- prepare_run_dir ---------------------------------------------------------


def test_prepare_run_dir_creates_absolute_folder(tmp_path: Path) -> None:
    run_dir = prepare_run_dir(tmp_path / "runs", "baseline")

    assert run_dir.is_dir()
    assert run_dir.is_absolute()
    assert run_dir.name == "baseline"


def test_prepare_run_dir_existing_folder_raises(tmp_path: Path) -> None:
    (tmp_path / "runs" / "baseline").mkdir(parents=True)

    with pytest.raises(FileExistsError, match="already exists"):
        prepare_run_dir(tmp_path / "runs", "baseline")


# --- collect_run_info and write_run_info -------------------------------------


def test_collect_run_info_contains_config_and_versions(tmp_path: Path) -> None:
    info = collect_run_info(make_config(tmp_path))

    assert info["config"]["seed"] == 7
    assert info["config"]["data_yaml"].endswith("data.yaml")
    assert "ultralytics" in {name.lower() for name in info["packages"]}
    assert info["device"]


def test_write_run_info_saves_yaml(tmp_path: Path) -> None:
    info_file = write_run_info(tmp_path, {"config": {"seed": 7}})

    assert yaml.safe_load(info_file.read_text(encoding="utf-8")) == {"config": {"seed": 7}}


def test_write_run_info_missing_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Run folder"):
        write_run_info(tmp_path / "missing", {})


# --- train and main ----------------------------------------------------------


def test_train_passes_config_to_yolo(tmp_path: Path, fake_yolo: type[FakeYOLO]) -> None:
    config = make_config(tmp_path)

    run_dir = train(config, tmp_path / "runs")

    assert (run_dir / RUN_INFO_NAME).is_file()
    call = fake_yolo.calls[0]
    assert call["model"] == "yolov8n.pt"
    assert (call["epochs"], call["imgsz"], call["seed"]) == (3, 320, 7)
    assert call["deterministic"] is True
    assert Path(call["project"]).is_absolute()
    assert Path(call["project"]) / call["name"] == run_dir


def test_train_without_dataset_does_not_start(tmp_path: Path, fake_yolo: type[FakeYOLO]) -> None:
    config = make_config(tmp_path)
    config.data_yaml.unlink()

    with pytest.raises(FileNotFoundError):
        train(config, tmp_path / "runs")
    assert fake_yolo.calls == []
    assert not (tmp_path / "runs").exists()


def test_main_reads_config_file(tmp_path: Path, fake_yolo: type[FakeYOLO]) -> None:
    config = make_config(tmp_path)
    config_file = tmp_path / "config.yaml"
    values = {
        "name": config.name,
        "data_yaml": str(config.data_yaml),
        "model": config.model,
        "epochs": config.epochs,
        "imgsz": config.imgsz,
        "seed": config.seed,
    }
    config_file.write_text(yaml.safe_dump(values), encoding="utf-8")

    main(["--config", str(config_file), "--project", str(tmp_path / "runs")])

    assert (tmp_path / "runs" / "test_run" / RUN_INFO_NAME).is_file()
    assert len(fake_yolo.calls) == 1
