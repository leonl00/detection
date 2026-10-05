"""Tests for scripts/download_data.py.

Only the offline parts are tested. ``download_dataset`` needs the internet and
a real API key, so it is checked by hand when the dataset is downloaded.
``monkeypatch`` changes environment variables only for the duration of one test.
"""

from pathlib import Path

import pytest

from download_data import (
    API_KEY_NAME,
    check_dataset_constants,
    check_target_is_free,
    get_api_key,
)

# --- check_dataset_constants -----------------------------------------------


def test_constants_complete_are_accepted() -> None:
    check_dataset_constants("my-workspace", "3d-print-defects", 2)


@pytest.mark.parametrize(
    ("workspace", "project", "version"),
    [("", "project", 1), ("workspace", "", 1), ("workspace", "project", 0)],
)
def test_constants_incomplete_raise(workspace: str, project: str, version: int) -> None:
    with pytest.raises(ValueError, match="No dataset selected"):
        check_dataset_constants(workspace, project, version)


# --- get_api_key -----------------------------------------------------------


def test_api_key_is_read_from_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(API_KEY_NAME, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(f"{API_KEY_NAME}=abc123\n", encoding="utf-8")

    assert get_api_key(env_file) == "abc123"


def test_api_key_from_environment_without_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(API_KEY_NAME, "from-environment")

    assert get_api_key(tmp_path / "missing.env") == "from-environment"


def test_missing_api_key_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(API_KEY_NAME, raising=False)

    with pytest.raises(ValueError, match=f"{API_KEY_NAME} is missing"):
        get_api_key(tmp_path / "missing.env")


def test_placeholder_api_key_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(API_KEY_NAME, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(f"{API_KEY_NAME}=your_api_key_here\n", encoding="utf-8")

    with pytest.raises(ValueError, match="is missing"):
        get_api_key(env_file)


# --- check_target_is_free --------------------------------------------------


def test_target_that_does_not_exist_is_free(tmp_path: Path) -> None:
    check_target_is_free(tmp_path / "dataset")


def test_empty_target_is_free(tmp_path: Path) -> None:
    check_target_is_free(tmp_path)


def test_target_with_files_raises(tmp_path: Path) -> None:
    (tmp_path / "data.yaml").touch()

    with pytest.raises(FileExistsError, match="already contains data"):
        check_target_is_free(tmp_path)
