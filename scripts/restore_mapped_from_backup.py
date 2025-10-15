"""
Restore images from backup `images_all_full_unmapped_backup` back to `images_all_full`
if they are referenced in `dataset/combined_filtered.csv`.

Usage:
    & ".venv/Scripts/python.exe" scripts/restore_mapped_from_backup.py
"""
from pathlib import Path
import pandas as pd
import shutil

workspace = Path(__file__).resolve().parents[1]
dataset = workspace / 'dataset'
images_dir = dataset / 'images_all_full'
backup_dir = dataset / 'images_all_full_unmapped_backup'

filtered_csv = dataset / 'combined_filtered.csv'
if not filtered_csv.exists():
    print('Filtered CSV not found:', filtered_csv)
    raise SystemExit(1)

print('Reading filtered CSV...')
df = pd.read_csv(filtered_csv)
if '__img_basename' not in df.columns:
    from urllib.parse import urlparse
    df['__img_basename'] = df['image_link'].fillna('').astype(str).apply(lambda u: Path(urlparse(u).path).name)

used = set(df['__img_basename'].dropna().astype(str).tolist())
used_stems = set([Path(n).stem for n in used])

if not backup_dir.exists():
    print('Backup folder not found:', backup_dir)
    raise SystemExit(1)

print('Scanning backup folder:', backup_dir)
all_backup_files = [f for f in backup_dir.rglob('*') if f.is_file()]
print('Total files in backup:', len(all_backup_files))

moved_back = 0
for f in all_backup_files:
    name = f.name
    stem = f.stem
    if name in used or stem in used_stems:
        dest = images_dir / name
        # ensure destination doesn't already exist
        if dest.exists():
            # if exists, remove backup file
            f.unlink()
        else:
            shutil.move(str(f), str(dest))
            moved_back += 1

# Post-check: how many referenced images are still missing from images_dir
print('Post-restore scan...')
present = set()
for f in images_dir.rglob('*'):
    if f.is_file():
        present.add(f.name)
        present.add(f.stem)

missing = []
for name in used:
    bn = Path(name).name
    stem = Path(name).stem
    if bn not in present and stem not in present:
        missing.append(name)

print(f'Moved back {moved_back} files from backup to {images_dir}')
print(f'Files remaining in backup: {len([f for f in backup_dir.rglob("*") if f.is_file()])}')
print(f'Referenced images missing after restore: {len(missing)}')
if len(missing) > 0:
    print('Sample missing (10):', missing[:10])

print('Done')
