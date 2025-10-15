"""
Move images from dataset/images_all_full that are NOT referenced in dataset/combined_filtered.csv
to dataset/images_all_full_unmapped_backup/ (backup, not permanent delete).

Usage:
    & ".venv/Scripts/python.exe" scripts/remove_unmapped_images.py
"""
from pathlib import Path
import pandas as pd
import shutil

workspace = Path(__file__).resolve().parents[1]
dataset = workspace / 'dataset'
images_dir = dataset / 'images_all_full'
backup_dir = dataset / 'images_all_full_unmapped_backup'
backup_dir.mkdir(parents=True, exist_ok=True)

filtered_csv = dataset / 'combined_filtered.csv'
if not filtered_csv.exists():
    print('Filtered CSV not found:', filtered_csv)
    raise SystemExit(1)

print('Reading filtered CSV...')
df = pd.read_csv(filtered_csv)
if '__img_basename' not in df.columns:
    # extract basename if missing
    from urllib.parse import urlparse
    df['__img_basename'] = df['image_link'].fillna('').astype(str).apply(lambda u: Path(urlparse(u).path).name)

used = set(df['__img_basename'].dropna().astype(str).tolist())
# also consider stems (to match files with _1 suffix)
used_stems = set([Path(n).stem for n in used])

print('Scanning images folder:', images_dir)
if not images_dir.exists():
    print('Images folder does not exist:', images_dir)
    raise SystemExit(1)

moved = 0
kept = 0
for f in images_dir.rglob('*'):
    if not f.is_file():
        continue
    name = f.name
    stem = f.stem
    if stem in used_stems or name in used:
        kept += 1
    else:
        # move to backup
        dest = backup_dir / name
        shutil.move(str(f), str(dest))
        moved += 1

print(f'Moved {moved} files to {backup_dir}; kept {kept} files in {images_dir}')
