#!/usr/bin/env python3
"""
Create a CSV report mapping image basenames referenced in dataset/combined_filtered.csv
to where they currently reside on disk: images_all_full, images_all_full_unmapped_backup, or missing.

Output: dataset/image_location_report.csv
"""
from pathlib import Path
import csv
import sys
from urllib.parse import urlparse

workspace = Path(__file__).resolve().parents[1]
combined_csv = workspace / 'dataset' / 'combined_filtered.csv'
images_dir = workspace / 'dataset' / 'images_all_full'
backup_dir = workspace / 'dataset' / 'images_all_full_unmapped_backup'
report_csv = workspace / 'dataset' / 'image_location_report.csv'

if not combined_csv.exists():
    print(f"ERROR: combined_filtered.csv not found at {combined_csv}")
    sys.exit(1)

# Find image column by reading header
with combined_csv.open('r', encoding='utf-8', errors='replace', newline='') as f:
    reader = csv.reader(f)
    try:
        header = next(reader)
    except StopIteration:
        print("ERROR: combined_filtered.csv is empty")
        sys.exit(1)

image_col_idx = None
for i, col in enumerate(header):
    name = col.lower()
    if 'image' in name or 'img' in name or 'photo' in name or 'image_link' in name:
        image_col_idx = i
        image_col_name = col
        break

if image_col_idx is None:
    # fallback: common column names
    candidates = ['image_link', 'image', 'img', 'imageurl', 'image_url', 'imagepath']
    for c in candidates:
        if c in [h.lower() for h in header]:
            image_col_idx = [h.lower() for h in header].index(c)
            image_col_name = header[image_col_idx]
            break

if image_col_idx is None:
    print("Could not auto-detect an image column in combined_filtered.csv. Columns:\n", header)
    sys.exit(1)

print(f"Detected image column: '{image_col_name}' at index {image_col_idx}")

# Collect unique basenames referenced in CSV
unique_basenames = set()
count_rows = 0
with combined_csv.open('r', encoding='utf-8', errors='replace', newline='') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        count_rows += 1
        if len(row) <= image_col_idx:
            continue
        val = row[image_col_idx].strip()
        if not val:
            continue
        # parse URL or path to get basename
        try:
            p = urlparse(val)
            basename = Path(p.path).name
        except Exception:
            basename = Path(val).name
        if basename:
            unique_basenames.add(basename)

print(f"Rows scanned: {count_rows}")
print(f"Unique image basenames found: {len(unique_basenames)}")

# Build filename maps for images_dir and backup_dir
images_present = {}
backup_present = {}

if images_dir.exists() and images_dir.is_dir():
    for p in images_dir.iterdir():
        if p.is_file():
            images_present[p.name] = str(p)
else:
    print(f"Warning: images directory not found: {images_dir}")

if backup_dir.exists() and backup_dir.is_dir():
    for p in backup_dir.iterdir():
        if p.is_file():
            backup_present[p.name] = str(p)
else:
    print(f"Warning: backup directory not found: {backup_dir}")

# Build stem maps for more flexible matching (e.g., file_1.jpg matching file.jpg stem)
from collections import defaultdict
images_stem_map = defaultdict(list)
for name, path in images_present.items():
    images_stem_map[Path(name).stem].append(name)

backup_stem_map = defaultdict(list)
for name, path in backup_present.items():
    backup_stem_map[Path(name).stem].append(name)

# Prepare report rows
report_rows = []
missing = 0
found_in_images = 0
found_in_backup = 0

for basename in sorted(unique_basenames):
    stem = Path(basename).stem
    matched = []
    matched_paths = []
    location = 'missing'

    if basename in images_present:
        matched = [basename]
        matched_paths = [images_present[basename]]
        location = 'images_all_full'
        found_in_images += 1
    elif stem in images_stem_map:
        matched = images_stem_map[stem]
        matched_paths = [images_present[n] for n in matched]
        location = 'images_all_full'
        found_in_images += 1
    elif basename in backup_present:
        matched = [basename]
        matched_paths = [backup_present[basename]]
        location = 'images_all_full_unmapped_backup'
        found_in_backup += 1
    elif stem in backup_stem_map:
        matched = backup_stem_map[stem]
        matched_paths = [backup_present[n] for n in matched]
        location = 'images_all_full_unmapped_backup'
        found_in_backup += 1
    else:
        missing += 1

    report_rows.append({
        'image_ref': basename,
        'stem': stem,
        'location': location,
        'matched_filenames': ';'.join(matched),
        'matched_paths': ';'.join(matched_paths),
        'matched_count': len(matched)
    })

# Write report CSV
fieldnames = ['image_ref','stem','location','matched_count','matched_filenames','matched_paths']
with report_csv.open('w', encoding='utf-8', newline='') as out:
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    for r in report_rows:
        writer.writerow(r)

print(f"Report written to: {report_csv}")
print(f"Summary: total unique={len(unique_basenames)}, images_dir={found_in_images}, backup={found_in_backup}, missing={missing}")
