"""Train a YOLOv8 model as described by a configuration file.

All settings come from a YAML file in ``configs/`` (see ``detection.config``).
Every run gets its own folder ``<project>/<name>/``, and before the training
starts, a ``run_info.yaml`` is written there with the configuration, the seed,
the hardware and the versions of all installed packages. A run can therefore be
traced and repeated later, even if it was interrupted.

Usage (from the project root):
    python -m detection.train --config configs/baseline.yaml

In Google Colab the results should be stored on Google Drive, so they survive
the end of the session:
    python -m detection.train --config configs/baseline.yaml \\
        --project /content/drive/MyDrive/detection/runs
"""

import argparse
import platform
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import distributions
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

from detection.config import Config, load_config

RUN_INFO_NAME = "run_info.yaml"


# --- Preparing the run -------------------------------------------------------


def check_data_yaml(data_yaml: Path) -> None:
    """Check that the dataset description file exists before training starts.

    Args:
        data_yaml: Path to the ``data.yaml`` of the dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_yaml}. Download it first, see the README."
        )


def prepare_run_dir(project: Path, name: str) -> Path:
    """Create the output folder of a run, refusing to reuse an existing one.

    Ultralytics would quietly write to ``<name>2`` instead, so a repeated run
    could be mistaken for the original one.

    Args:
        project: Folder that holds all runs, for example ``runs``.
        name: Name of this run, taken from the configuration.

    Returns:
        The absolute path of the new, empty run folder.

    Raises:
        FileExistsError: If the run folder already exists.
    """
    run_dir = (project / name).resolve()
    if run_dir.exists():
        raise FileExistsError(
            f"{run_dir} already exists. Delete it or choose another 'name' in the config."
        )
    run_dir.mkdir(parents=True)
    return run_dir


def collect_run_info(config: Config) -> dict:
    """Collect everything needed to understand and repeat a run.

    Args:
        config: The configuration of the run.

    Returns:
        Configuration, start time, hardware and package versions as a dictionary.
    """
    settings = asdict(config)
    settings["data_yaml"] = config.data_yaml.as_posix()
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    packages = {dist.metadata["Name"]: dist.version for dist in distributions()}
    return {
        "config": settings,
        "started": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
    }


def write_run_info(run_dir: Path, info: dict) -> Path:
    """Save the run information as YAML in the run folder.

    Args:
        run_dir: Folder of the run.
        info: Result of ``collect_run_info``.

    Returns:
        Path of the written file.

    Raises:
        FileNotFoundError: If the run folder does not exist.
    """
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run folder not found: {run_dir}")
    info_file = run_dir / RUN_INFO_NAME
    info_file.write_text(yaml.safe_dump(info, sort_keys=False), encoding="utf-8")
    return info_file


# --- Training ----------------------------------------------------------------


def train(config: Config, project: Path) -> Path:
    """Train a model with the settings of ``config``.

    Args:
        config: The validated configuration.
        project: Folder that holds all runs.

    Returns:
        The run folder with weights (``weights/best.pt``), plots and metrics.

    Raises:
        FileNotFoundError: If the dataset is missing.
        FileExistsError: If a run with the same name already exists.
    """
    check_data_yaml(config.data_yaml)
    run_dir = prepare_run_dir(project, config.name)
    write_run_info(run_dir, collect_run_info(config))

    model = YOLO(config.model)
    model.train(
        data=str(config.data_yaml),
        epochs=config.epochs,
        imgsz=config.imgsz,
        seed=config.seed,
        deterministic=True,
        # An absolute path is required: Ultralytics puts relative paths into its own folder.
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,  # the folder was created above and only holds run_info.yaml
    )
    return run_dir


# --- Command line ------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line options.

    Args:
        argv: Options to parse; ``None`` means the real command line.

    Returns:
        The parsed options ``config`` and ``project``.
    """
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model from a config file.")
    parser.add_argument("--config", type=Path, required=True, help="experiment YAML file")
    parser.add_argument("--project", type=Path, default=Path("runs"), help="folder for all runs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load the configuration and start the training."""
    args = parse_args(argv)
    config = load_config(args.config)
    run_dir = train(config, args.project)
    print(f"Training finished. Results in {run_dir}")


if __name__ == "__main__":
    main()
