"""Tests for detection.config.

Every test writes its own YAML file into ``tmp_path``, a temporary folder that
pytest creates for each test and deletes afterwards. The tests therefore need
no real files from the project, except the last test, which checks the
committed ``configs/baseline.yaml``.
"""

from pathlib import Path

import pytest
import yaml

from detection.config import Config, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALID_VALUES = {
    "name": "test_run",
    "data_yaml": "data/dataset/data.yaml",
    "model": "yolov8n.pt",
    "epochs": 10,
    "imgsz": 640,
    "seed": 0,
}


def write_config(folder: Path, values: dict) -> Path:
    """Write ``values`` as a YAML file into ``folder`` and return its path."""
    path = folder / "config.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return path


# --- Normal cases ----------------------------------------------------------


def test_load_valid_config(tmp_path: Path) -> None:
    path = write_config(tmp_path, VALID_VALUES)

    config = load_config(path)

    assert config == Config(
        name="test_run",
        data_yaml=Path("data/dataset/data.yaml"),
        model="yolov8n.pt",
        epochs=10,
        imgsz=640,
        seed=0,
    )


def test_load_config_accepts_string_path(tmp_path: Path) -> None:
    path = write_config(tmp_path, VALID_VALUES)

    config = load_config(str(path))

    assert config.name == "test_run"


def test_data_yaml_does_not_need_to_exist(tmp_path: Path) -> None:
    values = {**VALID_VALUES, "data_yaml": "does/not/exist.yaml"}
    path = write_config(tmp_path, values)

    config = load_config(path)

    assert config.data_yaml == Path("does/not/exist.yaml")


def test_config_is_read_only(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, VALID_VALUES))

    with pytest.raises(AttributeError):
        config.epochs = 99  # type: ignore[misc]


def test_baseline_config_in_repository_is_valid() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "baseline.yaml")

    assert config.name == "baseline"


def test_clean_config_differs_from_baseline_only_in_dataset() -> None:
    # The split comparison is only fair if both runs use the same settings.
    baseline = load_config(PROJECT_ROOT / "configs" / "baseline.yaml")
    clean = load_config(PROJECT_ROOT / "configs" / "baseline_clean.yaml")

    assert clean.name == "baseline_clean"
    assert clean.data_yaml == Path("data/dataset_clean/data.yaml")
    assert (clean.model, clean.epochs, clean.imgsz, clean.seed) == (
        baseline.model,
        baseline.epochs,
        baseline.imgsz,
        baseline.seed,
    )


# --- Error cases: the file itself ------------------------------------------


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_config(tmp_path / "missing.yaml")


def test_empty_file_raises_value_error(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="key: value"):
        load_config(path)


def test_broken_yaml_raises_value_error(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("name: [unclosed", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid YAML"):
        load_config(path)


def test_list_instead_of_mapping_raises_value_error(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- name\n- epochs\n", encoding="utf-8")

    with pytest.raises(ValueError, match="key: value"):
        load_config(path)


# --- Error cases: missing and unknown fields -------------------------------


@pytest.mark.parametrize("key", list(VALID_VALUES))
def test_missing_field_raises_value_error(tmp_path: Path, key: str) -> None:
    values = {k: v for k, v in VALID_VALUES.items() if k != key}
    path = write_config(tmp_path, values)

    with pytest.raises(ValueError, match=f"Missing config fields: \\['{key}'\\]"):
        load_config(path)


def test_unknown_field_raises_value_error(tmp_path: Path) -> None:
    values = {**VALID_VALUES, "epoch": 5}  # typo of "epochs"
    path = write_config(tmp_path, values)

    with pytest.raises(ValueError, match="Unknown config fields: \\['epoch'\\]"):
        load_config(path)


# --- Error cases: invalid values -------------------------------------------
# ``parametrize`` runs the same test once per line, each time with other values.


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("name", ""),
        ("name", "my run"),
        ("name", "../escape"),
        ("name", "run\n"),
        ("name", 123),
        ("data_yaml", ""),
        ("data_yaml", None),
        ("model", "yolov8n"),
        ("model", "yolov8n.onnx"),
        ("model", None),
        ("epochs", 0),
        ("epochs", -5),
        ("epochs", 2.5),
        ("epochs", "10"),
        ("epochs", True),
        ("imgsz", 0),
        ("imgsz", 650),
        ("imgsz", 16),
        ("imgsz", 640.0),
        ("seed", -1),
        ("seed", "42"),
        ("seed", None),
    ],
)
def test_invalid_value_raises_value_error(tmp_path: Path, key: str, bad_value: object) -> None:
    values = {**VALID_VALUES, key: bad_value}
    path = write_config(tmp_path, values)

    with pytest.raises(ValueError, match=f"'{key}'"):
        load_config(path)


@pytest.mark.parametrize("imgsz", [32, 320, 640, 1280])
def test_imgsz_multiples_of_32_are_accepted(tmp_path: Path, imgsz: int) -> None:
    path = write_config(tmp_path, {**VALID_VALUES, "imgsz": imgsz})

    assert load_config(path).imgsz == imgsz
