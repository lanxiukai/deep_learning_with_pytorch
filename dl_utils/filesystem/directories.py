import shutil
import os

from .project_root import infer_project_root


def clean_pycache():
    """Clean the __pycache__ folder when the program exits."""
    root = infer_project_root()
    for folder in root.rglob('__pycache__'):
        shutil.rmtree(folder, ignore_errors=True)


def reset_dir(path: str) -> None:
    """
    Remove a file/dir at `path` if it exists, then (re)create it as a directory.

    This is useful for experiment output folders (ensure a clean run).
    """
    if not path:
        raise ValueError("reset_dir: path is empty.")

    abs_path = os.path.abspath(path)
    if abs_path == os.path.abspath(os.path.sep):
        raise ValueError(f"reset_dir: refuse to delete root directory: {path!r}")

    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    os.makedirs(path, exist_ok=True)
