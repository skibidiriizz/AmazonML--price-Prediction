"""
Combine train.csv and test.csv into dataset/combined.csv (keeps originals),
then filter combined to rows whose image exists in dataset/images_all_full.
Writes dataset/combined_filtered.csv and prints a summary.
"""
from pathlib import Path
import pandas as pd
from urllib.parse import urlparse
import os

workspace = Path(__file__).resolve().parents[1]
dataset = workspace / 'dataset'
images_dir = dataset / 'images_all_full'

train_path = dataset / 'train.csv'
test_path = dataset / 'test.csv'
combined_path = dataset / 'combined.csv'
filtered_path = dataset / 'combined_filtered.csv'

print('Reading CSVs...')
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print('Concatenating...')
# add source column to know origin
train['__source'] = 'train'
test['__source'] = 'test'
combined = pd.concat([train, test], ignore_index=True)

print(f'Saving combined to {combined_path} (rows={len(combined)})')
combined.to_csv(combined_path, index=False)

# Build a list of available image stems from folder
print('Scanning image folder:', images_dir)
available = []
if images_dir.exists():
    for f in images_dir.rglob('*'):
        if f.is_file():
            ext = f.suffix.lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                available.append(f.name)
else:
    print('Image folder does not exist:', images_dir)

print('Found', len(available), 'image files in folder')

# create a set of stems for quick lookup; include stems and stems before any _counter suffix
file_stems = set()
for name in available:
    stem = Path(name).stem
    file_stems.add(stem)

# helper to check if a basename from URL exists in folder (allowing _N suffix)
def image_present_for_basename(basename: str) -> bool:
    # basename e.g. '51mo8htwTHL.jpg'
    base_stem = Path(basename).stem
    # direct match
    if base_stem in file_stems:
        return True
    # match by prefix with appended counter like base_1
    prefix = base_stem + '_'
    for s in file_stems:
        if s.startswith(prefix):
            return True
    return False

print('Extracting basenames from combined image_link column and filtering rows...')
# Ensure column exists
if 'image_link' not in combined.columns:
    print('No image_link column in combined CSV; aborting')
else:
    # extract basename
    combined['__img_basename'] = combined['image_link'].fillna('').astype(str).apply(lambda u: Path(urlparse(u).path).name)
    # check presence
    combined['__img_exists'] = combined['__img_basename'].apply(image_present_for_basename)
    kept = combined[combined['__img_exists']].copy()
    print(f'Total combined rows: {len(combined)}')
    print(f'Rows with images present: {len(kept)} (retained)')
    print(f'Rows without images: {len(combined) - len(kept)} (filtered out)')
    kept.to_csv(filtered_path, index=False)
    print(f'Wrote filtered combined CSV to {filtered_path}')

print('Done')
