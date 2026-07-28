"""Dataset download and archive extraction primitives."""

import hashlib
import os
import shutil
import tarfile
import zipfile

import requests

from dl_utils.filesystem.project_root import infer_project_root


DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

# map DATA_HUB keys to subdirectories under data/ (avoids scattering files at root)
_DOWNLOAD_SUBDIR = {
    'airfoil':             'airfoil_self_noise',
    'kaggle_house_train':  'kaggle_house_price',
    'kaggle_house_test':   'kaggle_house_price',
    'time_machine':        'time_machine',
    'fra-eng':             'fra_eng',
    'pokemon':             'pokemon',
}


def download(name, cache_dir=None):
    """
    Download a file inserted into DATA_HUB, and return the local filename.
    
    Args:
        name: the name of the file to download
        cache_dir: the folder to cache the file (Default: None)
    Returns:
        the path to the downloaded file
    """
    assert name in DATA_HUB, f"{name} does not exist in {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    if cache_dir is None:
        repo_root = infer_project_root()
        cache_dir = os.path.join(str(repo_root), 'data')
        subdir = _DOWNLOAD_SUBDIR.get(name)
        if subdir:
            cache_dir = os.path.join(cache_dir, subdir)
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # cache hit
    print(f"Downloading {fname} from {url}...")
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


def download_extract(name, folder=None):
    """
    Download and extract a zip/tar file.
    
    Args:
        name: the name of the file to download
        folder: the folder to extract the file to (Default: None)
    Returns:
        the path to the extracted file
    """
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        with zipfile.ZipFile(fname, 'r') as fp:
            archive_entries = fp.namelist()
            top_dirs = {
                entry.strip('/').split('/', 1)[0]
                for entry in archive_entries if entry.strip('/')
            }
            if _is_flattened_archive_complete(
                    base_dir, data_dir, archive_entries, top_dirs):
                data_dir = base_dir
            else:
                fp.extractall(base_dir)
    elif ext in ('.tar', '.gz'):
        with tarfile.open(fname, 'r') as fp:
            archive_entries = fp.getnames()
            top_dirs = {
                entry.strip('/').split('/', 1)[0]
                for entry in archive_entries if entry.strip('/')
            }
            if _is_flattened_archive_complete(
                    base_dir, data_dir, archive_entries, top_dirs):
                data_dir = base_dir
            else:
                fp.extractall(base_dir)
    else:
        assert False, 'only zip/tar files can be extracted'

    # Flatten archives that contain a single top-level directory
    # (e.g. fra-eng.zip → fra-eng/fra.txt → avoid fra_eng/fra-eng/fra.txt)
    if os.path.isdir(data_dir):
        if len(top_dirs) == 1:
            inner_dir = os.path.join(base_dir, top_dirs.pop())
            if os.path.isdir(inner_dir) and inner_dir == data_dir:
                for item in os.listdir(inner_dir):
                    shutil.move(os.path.join(inner_dir, item),
                                os.path.join(base_dir, item))
                os.rmdir(inner_dir)
                data_dir = base_dir   # flattened — point to parent

    return os.path.join(data_dir, folder) if folder else data_dir


def _is_flattened_archive_complete(base_dir, data_dir, entries, top_dirs):
    """Return whether a single-root archive is already flattened in base_dir."""
    if len(top_dirs) != 1:
        return False
    top_dir = next(iter(top_dirs))
    if os.path.join(base_dir, top_dir) != data_dir:
        return False

    prefix = f'{top_dir}/'
    top_level_items = {
        entry.strip('/')[len(prefix):].split('/', 1)[0]
        for entry in entries
        if entry.strip('/').startswith(prefix)
        and entry.strip('/')[len(prefix):]
    }
    return bool(top_level_items) and all(
        os.path.exists(os.path.join(base_dir, item))
        for item in top_level_items
    )


def download_all():
    """Download all files in DATA_HUB"""
    for name in DATA_HUB:
        download(name)


DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

DATA_HUB['time_machine'] = (
    DATA_URL + 'timemachine.txt',
    '090b5e7e70c295757f55df93cb0a180b9691891a')

DATA_HUB['fra-eng'] = (DATA_URL + 'fra-eng.zip',
                        '94646ad1522d915e7b0f9296181140edcf86a4f5')

DATA_HUB['pokemon'] = (DATA_URL + 'pokemon.zip',
                       'c065c0e2593b8b161a2d7873e42418bf6a21106c')

DATA_HUB['airfoil'] = (DATA_URL + 'airfoil_self_noise.dat',
                       '76e5be1548fd8222e5074cf0faae75edff8cf93f')
