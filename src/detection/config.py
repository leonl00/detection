"""Load and validate experiment configurations from YAML files.

Every experiment (training run, evaluation, ...) is described by one YAML file in
``configs/``. This module turns such a file into a ``Config`` object and rejects
nonsensical values early, before any expensive work starts.

Example:
    >>> from detection.config import load_config
    >>> config = load_config("configs/baseline.yaml")
    >>> config.epochs
    50
"""

import re
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

# The name becomes a folder name (runs/<name>/), so only allow safe characters.
NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]+")

# Ultralytics accepts pretrained weights (.pt) or a model definition (.yaml).
MODEL_SUFFIXES = (".pt", ".yaml")

# YOLO downsamples the image by a factor of 32, so the size must be a multiple of it.
IMGSZ_DIVISOR = 32


@dataclass(frozen=True)
class Config:
    """Settings for one experiment.

    ``frozen=True`` makes the object read-only, so no code can change a setting
    by accident after it was loaded and validated.

    Attributes:
        name: Short experiment name, used as the output folder ``runs/<name>/``.
        data_yaml: Path to the dataset description file (``data.yaml``).
        model: Pretrained weights or model definition, for example ``yolov8n.pt``.
        epochs: Number of passes over the training data.
        imgsz: Edge length in pixels that images are resized to.
        seed: Random seed that makes the training reproducible.
    """

    name: str
    data_yaml: Path
    model: str
    epochs: int
    imgsz: int
    seed: int


def load_config(path: Path | str) -> Config:
    """Read a YAML file and return a validated ``Config``.

    The file must contain exactly the fields of ``Config``. Missing and unknown
    fields are both errors, so that a typo like ``epoch:`` is noticed at once.
    Whether ``data_yaml`` exists is not checked here, because the dataset may
    not be downloaded yet; the training script checks it before it starts.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        The validated configuration.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid YAML, has missing or unknown fields,
            or contains invalid values.
    """
    path = Path(path)
    raw = read_yaml(path)
    check_keys(raw)
    return Config(
        name=parse_name(raw["name"]),
        data_yaml=parse_data_yaml(raw["data_yaml"]),
        model=parse_model(raw["model"]),
        epochs=parse_int(raw["epochs"], key="epochs", minimum=1),
        imgsz=parse_imgsz(raw["imgsz"]),
        seed=parse_int(raw["seed"], key="seed", minimum=0),
    )


def read_yaml(path: Path) -> dict:
    """Read a YAML file that must contain key-value pairs.

    Args:
        path: Path to the YAML file.

    Returns:
        The file content as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid YAML or not a mapping.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        with path.open(encoding="utf-8") as file:
            content = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise ValueError(f"Config file {path} is not valid YAML: {error}") from error

    if not isinstance(content, dict):
        raise ValueError(f"Config file {path} must contain 'key: value' pairs.")
    return content


def check_keys(raw: dict) -> None:
    """Check that ``raw`` has exactly the fields of ``Config``.

    Args:
        raw: The dictionary read from the YAML file.

    Raises:
        ValueError: If fields are missing or unknown fields are present.
    """
    expected = {field.name for field in fields(Config)}
    missing = expected - raw.keys()
    unknown = raw.keys() - expected

    if missing:
        raise ValueError(f"Missing config fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")


def parse_name(value: object) -> str:
    """Validate the experiment name.

    Args:
        value: Raw value from the YAML file.

    Returns:
        The name.

    Raises:
        ValueError: If the name is not a string of letters, digits, '-' or '_'.
    """
    if not isinstance(value, str) or not NAME_PATTERN.fullmatch(value):
        raise ValueError(f"'name' must only contain letters, digits, '-' and '_', got {value!r}")
    return value


def parse_data_yaml(value: object) -> Path:
    """Validate the path to the dataset description file.

    Args:
        value: Raw value from the YAML file.

    Returns:
        The path as a ``Path`` object.

    Raises:
        ValueError: If the value is not a non-empty string.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'data_yaml' must be a non-empty path, got {value!r}")
    return Path(value)


def parse_model(value: object) -> str:
    """Validate the model weights or definition file name.

    Args:
        value: Raw value from the YAML file.

    Returns:
        The model file name.

    Raises:
        ValueError: If the value is not a string ending in ``.pt`` or ``.yaml``.
    """
    if not isinstance(value, str) or not value.endswith(MODEL_SUFFIXES):
        raise ValueError(f"'model' must end with one of {MODEL_SUFFIXES}, got {value!r}")
    return value


def parse_int(value: object, key: str, minimum: int) -> int:
    """Validate a whole number with a lower bound.

    Args:
        value: Raw value from the YAML file.
        key: Field name, used in the error message.
        minimum: Smallest allowed value.

    Returns:
        The number.

    Raises:
        ValueError: If the value is not a whole number or smaller than ``minimum``.
    """
    # In Python, True and False count as integers (1 and 0), so exclude them explicitly.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"'{key}' must be a whole number, got {value!r}")
    if value < minimum:
        raise ValueError(f"'{key}' must be at least {minimum}, got {value}")
    return value


def parse_imgsz(value: object) -> int:
    """Validate the image size.

    Args:
        value: Raw value from the YAML file.

    Returns:
        The image size in pixels.

    Raises:
        ValueError: If the value is not a positive multiple of 32.
    """
    imgsz = parse_int(value, key="imgsz", minimum=IMGSZ_DIVISOR)
    if imgsz % IMGSZ_DIVISOR != 0:
        raise ValueError(f"'imgsz' must be a multiple of {IMGSZ_DIVISOR}, got {imgsz}")
    return imgsz
