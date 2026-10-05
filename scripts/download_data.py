"""Download the 3D printing defect dataset from Roboflow in YOLOv8 format.

Usage (from the project root):
    python scripts/download_data.py

The API key is read from the file ``.env`` (see ``.env.example``). The dataset is
written to ``data/dataset/`` and is never committed to Git.
"""

import os
from pathlib import Path

from dotenv import dotenv_values

from detection.dataset_check import check_dataset_structure, read_class_names

# --- Dataset on Roboflow Universe --------------------------------------------
# Copy these three values from "Download Dataset -> YOLOv8 -> show download code"
# on the dataset page. They are deliberately left empty until a dataset is chosen.
WORKSPACE = ""
PROJECT = ""
VERSION = 0

# --- Local paths --------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
TARGET_DIR = PROJECT_ROOT / "data" / "dataset"

API_KEY_NAME = "ROBOFLOW_API_KEY"


def check_dataset_constants(workspace: str, project: str, version: int) -> None:
    """Make sure the dataset to download has been chosen.

    Args:
        workspace: Roboflow workspace name.
        project: Roboflow project name.
        version: Dataset version number.

    Raises:
        ValueError: If one of the values is still empty or not positive.
    """
    if not workspace or not project or version < 1:
        raise ValueError(
            "No dataset selected yet. Set WORKSPACE, PROJECT and VERSION at the top of "
            "scripts/download_data.py (see 'Download Dataset -> show download code' "
            "on the Roboflow dataset page)."
        )


def get_api_key(env_file: Path) -> str:
    """Read the Roboflow API key from the environment or a ``.env`` file.

    Args:
        env_file: Path to the ``.env`` file.

    Returns:
        The API key.

    Raises:
        ValueError: If no key is set or the key is still the example placeholder.
    """
    # A key set in the environment wins over the .env file. dotenv_values only reads
    # the file and, unlike load_dotenv, does not change the environment as a side effect.
    api_key = os.getenv(API_KEY_NAME) or dotenv_values(env_file).get(API_KEY_NAME) or ""
    api_key = api_key.strip()

    if not api_key or api_key == "your_api_key_here":
        raise ValueError(
            f"{API_KEY_NAME} is missing. Copy .env.example to .env and insert your key "
            "from https://app.roboflow.com/settings/api"
        )
    return api_key


def check_target_is_free(target_dir: Path) -> None:
    """Refuse to overwrite a dataset that was already downloaded.

    Args:
        target_dir: Folder the dataset will be written to.

    Raises:
        FileExistsError: If the folder exists and is not empty.
    """
    if target_dir.exists() and any(target_dir.iterdir()):
        raise FileExistsError(
            f"{target_dir} already contains data. Delete the folder to download again."
        )


def download_dataset(api_key: str, target_dir: Path) -> None:
    """Download the dataset in YOLOv8 format.

    This is the only function that talks to the internet, so it has no unit test.

    Args:
        api_key: Roboflow API key.
        target_dir: Folder the dataset is written to.
    """
    # Imported here so that the other functions (and their tests) work without
    # loading the large roboflow package.
    from roboflow import Roboflow

    project = Roboflow(api_key=api_key).workspace(WORKSPACE).project(PROJECT)
    project.version(VERSION).download("yolov8", location=str(target_dir))


def main() -> None:
    """Download the dataset and print a short summary."""
    check_dataset_constants(WORKSPACE, PROJECT, VERSION)
    api_key = get_api_key(ENV_FILE)
    check_target_is_free(TARGET_DIR)

    print(f"Downloading {WORKSPACE}/{PROJECT} version {VERSION} to {TARGET_DIR} ...")
    download_dataset(api_key, TARGET_DIR)

    image_counts = check_dataset_structure(TARGET_DIR)
    class_names = read_class_names(TARGET_DIR / "data.yaml")

    print("\nDownload complete.")
    for split, count in image_counts.items():
        print(f"  {split:<6} {count:>5} images")
    print(f"  classes: {', '.join(class_names)}")


if __name__ == "__main__":
    main()
