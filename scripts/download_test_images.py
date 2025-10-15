#!/usr/bin/env python3
"""
Download every image referenced in dataset/test.csv, including duplicates, and save them into dataset/test_img.
One file per CSV row. Filenames will be: {row_idx}_{sample_id if present}_{basename}

This intentionally preserves duplicates and will not skip existing files unless they match the generated name.
"""
from pathlib import Path
import csv
import sys
from urllib.parse import urlparse
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import time


def get_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[429,500,502,503,504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def get_basename_from_url(url):
    try:
        p = urlparse(url)
        return Path(p.path).name
    except Exception:
        return Path(url).name


def safe_filename(idx, sample_id, basename):
    parts = [str(idx)]
    if sample_id:
        parts.append(str(sample_id))
    if basename:
        parts.append(basename)
    # replace spaces
    return '_'.join(parts).replace(' ', '_')


def download_batch(rows_iter, out_dir, workers, session, log_writer, timeout=20):
    """Download rows provided by rows_iter (iterable of (idx, sample_id, link)). Writes results via log_writer."""
    results = []
    def download_one(item):
        idx, sample_id, link = item
        if not link:
            return (idx, sample_id, link, 'no_link', '')
        basename = get_basename_from_url(link)
        filename = safe_filename(idx, sample_id, basename)
        out_path = out_dir / filename
        try:
            resp = session.get(link, timeout=timeout)
            if resp.status_code == 200:
                with out_path.open('wb') as w:
                    w.write(resp.content)
                return (idx, sample_id, link, 'ok', str(out_path))
            else:
                return (idx, sample_id, link, f'status_{resp.status_code}', '')
        except Exception as e:
            return (idx, sample_id, link, f'error_{e}', '')

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(download_one, r): r for r in rows_iter}
        for fut in tqdm(as_completed(futures), total=len(futures), desc='batch downloads'):
            res = fut.result()
            log_writer.writerow(res)
            results.append(res)
    return results


def main():
    workspace = Path(__file__).resolve().parents[1]
    input_csv = workspace / 'dataset' / 'test.csv'
    out_dir = workspace / 'dataset' / 'test_img'

    parser = argparse.ArgumentParser(description='Batch downloader for test images')
    parser.add_argument('--batch-size', type=int, default=1000, help='Number of rows per batch')
    parser.add_argument('--start', type=int, default=1, help='1-based start row index to process')
    parser.add_argument('--end', type=int, default=0, help='1-based inclusive end row index (0 = to EOF)')
    parser.add_argument('--workers', type=int, default=16, help='Number of download threads')
    parser.add_argument('--timeout', type=int, default=20, help='HTTP timeout in seconds')
    parser.add_argument('--log-interval', type=int, default=100, help='Flush log every N rows')
    args = parser.parse_args()

    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        print(f"ERROR: test.csv not found at {input_csv}")
        sys.exit(1)

    # detect columns
    with input_csv.open('r', encoding='utf-8', errors='replace', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)

    image_col_idx = None
    id_col_idx = None
    for i, col in enumerate(header):
        name = col.lower()
        if 'image' in name or 'img' in name or 'photo' in name or 'image_link' in name:
            image_col_idx = i
        if 'id' == name or name.endswith('id'):
            if id_col_idx is None:
                id_col_idx = i

    if image_col_idx is None:
        print('Could not detect image column in test.csv. Columns:' , header)
        sys.exit(1)

    session = get_session()

    log_file = out_dir / 'download_log.csv'
    # open log in append mode so we can resume
    first_write = not log_file.exists()
    with log_file.open('a', encoding='utf-8', newline='') as lf:
        log_writer = csv.writer(lf)
        if first_write:
            log_writer.writerow(['row_idx','sample_id','link','status','out_path'])

        # stream through CSV and process in batches
        start = args.start
        end = args.end if args.end > 0 else None
        batch_size = args.batch_size
        workers = args.workers
        timeout = args.timeout

        with input_csv.open('r', encoding='utf-8', errors='replace', newline='') as f:
            reader = csv.reader(f)
            next(reader)  # header
            buffer = []
            current_idx = 0
            for row in reader:
                current_idx += 1
                row_number = current_idx  # 1-based relative to data rows
                global_row_idx = current_idx
                # Only process rows within requested range
                if global_row_idx < start:
                    continue
                if end is not None and global_row_idx > end:
                    break

                if len(row) <= image_col_idx:
                    link = ''
                else:
                    link = row[image_col_idx].strip()
                sample_id = ''
                if id_col_idx is not None and len(row) > id_col_idx:
                    sample_id = row[id_col_idx].strip()

                buffer.append((global_row_idx, sample_id, link))

                if len(buffer) >= batch_size:
                    # process batch
                    download_batch(buffer, out_dir, workers, session, log_writer, timeout=timeout)
                    lf.flush()
                    buffer = []
                    # be polite
                    time.sleep(0.5)

            # final partial batch
            if buffer:
                download_batch(buffer, out_dir, workers, session, log_writer, timeout=timeout)
                lf.flush()

    print(f"Done. Images saved to: {out_dir}. Log: {log_file}")


if __name__ == '__main__':
    main()
