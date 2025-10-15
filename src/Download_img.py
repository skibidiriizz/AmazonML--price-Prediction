"""Improved image downloader for the dataset.

Usage (from workspace root):
    & ".venv/Scripts/python.exe" src/Download_img.py

This script reads `dataset/train.csv` and `dataset/test.csv`, extracts `image_link`,
deduplicates, and downloads images into `dataset/images_all/`.
"""
from pathlib import Path
import sys
import re
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urlparse, unquote
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = WORKSPACE_ROOT / 'dataset'
# default output dir; can be overridden via --outdir
DEFAULT_OUTPUT_DIR = DATASET_DIR / 'images_all'
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_filename_from_url(url: str) -> str:
    path = urlparse(url).path
    name = os.path.basename(path)
    name = unquote(name)
    name = re.sub(r"[^0-9A-Za-z._-]", "_", name)
    if not name:
        name = 'image'
    # ensure extension
    if not re.search(r"\.(jpg|jpeg|png|bmp|gif)$", name, flags=re.IGNORECASE):
        name += '.jpg'
    return name


def get_session(retries=3, backoff_factor=0.3):
    s = requests.Session()
    retry = Retry(total=retries, backoff_factor=backoff_factor,
                  status_forcelist=(500, 502, 504))
    adapter = HTTPAdapter(max_retries=retry)
    s.mount('http://', adapter)
    s.mount('https://', adapter)
    return s


def download_one(session, url, folder):
    try:
        filename = safe_filename_from_url(url)
        save_path = folder / filename
        # avoid overwrite by adding suffix if needed
        if save_path.exists():
            base, ext = os.path.splitext(filename)
            i = 1
            while True:
                candidate = folder / f"{base}_{i}{ext}"
                if not candidate.exists():
                    save_path = candidate
                    break
                i += 1

        resp = session.get(url, stream=True, timeout=15)
        resp.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in resp.iter_content(1024 * 16):
                if chunk:
                    f.write(chunk)
        return True, str(save_path)
    except Exception as e:
        return False, f"{url} -> {e}"


def download_many(urls, folder, workers=8):
    session = get_session()
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(download_one, session, u, folder): u for u in urls}
        for fut in as_completed(futures):
            results.append(fut.result())
    return results


def main(limit=None, workers=8, outdir: str = None):
    import pandas as pd

    output_dir = Path(outdir) if outdir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATASET_DIR / 'train.csv', usecols=['image_link'])
    test = pd.read_csv(DATASET_DIR / 'test.csv', usecols=['image_link'])
    links = pd.concat([train['image_link'], test['image_link']], ignore_index=True)
    links = links.dropna().astype(str).str.strip()

    seen = set()
    unique_links = []
    for l in links:
        if l and l not in seen:
            seen.add(l)
            unique_links.append(l)

    if limit:
        unique_links = unique_links[:limit]

    print(f"Will download {len(unique_links)} images to {output_dir} (workers={workers})")
    results = download_many(unique_links, output_dir, workers=workers)
    ok = sum(1 for s, _ in results if s)
    fail = sum(1 for s, _ in results if not s)
    print(f"Done: success={ok} fail={fail}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=2000, help='Number of images to download; 0 means all')
    p.add_argument('--workers', type=int, default=8, help='Number of download worker threads')
    p.add_argument('--outdir', type=str, default=None, help='Output directory for images')
    args = p.parse_args()
    # interpret 0 as no limit
    limit = None if args.limit == 0 else args.limit
    main(limit=limit, workers=args.workers, outdir=args.outdir)
 


