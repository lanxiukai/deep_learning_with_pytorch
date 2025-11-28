'''
Import the module d2l_save from the project root, 
and clean the __pycache__ folder when the program exits
'''

import sys
import atexit
import shutil
from pathlib import Path

def _infer_project_root() -> Path:
    if '__file__' in globals():
        return Path(__file__).resolve().parent.parent
    return Path.cwd().resolve()

ROOT = _infer_project_root()
ROOT_STR = str(ROOT.resolve())
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

import d2l_save  # noqa: E402

def clean_pycache():
    for folder in ROOT.rglob('__pycache__'):
        shutil.rmtree(folder, ignore_errors=True)

atexit.register(clean_pycache)
