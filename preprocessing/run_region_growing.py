from pathlib import Path
import numpy as np
import pandas as pd
import nrrd
from scipy import ndimage
DATA_DIR = Path('/Users/zhiqiangkong/Desktop/DATA')
CSV_PATH = Path('/Users/zhiqiangkong/Desktop/DATA/region_growing_params.csv')
OUT_MASK_DIR = DATA_DIR / 'region_masks'
OUT_CSV_PATH = DATA_DIR / 'tumor_volumes.csv'
CONNECTIVITY = 1
APPLY_MORPH_CLEAN = True
MIN_REASONABLE_CM3 = 0.1
MAX_REASONABLE_CM3 = 100.0

def get_voxel_spacing(header):
    """
    Extract voxel spacing (mm) from NRRD header.
    """
    if 'space directions' not in header:
        raise ValueError("NRRD header missing 'space directions'.")
    dirs = header['space directions']
    spacing = []
    for vec in dirs:
        arr = np.array(vec, dtype=float)
        spacing.append(float(np.linalg.norm(arr)))
    return tuple(spacing)

def ensure_seed_in_bounds(seed, shape):
    x, y, z = seed
    return 0 <= x < shape[0] and 0 <= y < shape[1] and (0 <= z < shape[2])

def compute_distance_mask(shape, seed, max_radius_vox):
    x, y, z = seed
    xx, yy, zz = np.indices(shape)
    dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2 + (zz - z) ** 2)
    return dist <= max_radius_vox

def region_growing_with_radius(volume, seed, lower, upper, max_radius_vox, connectivity=1, apply_morph_clean=True):
    """
    Steps:
    1. Threshold to candidate mask
    2. Keep connected component containing seed
    3. Apply radius limit
    4. Optional morphology cleaning
    5. Re-keep seed component
    """
    if not ensure_seed_in_bounds(seed, volume.shape):
        raise ValueError(f'Seed {seed} is out of bounds for volume shape {volume.shape}')
    x, y, z = seed
    seed_hu = float(volume[x, y, z])
    if not lower <= seed_hu <= upper:
        raise ValueError(f'Seed point {seed} has HU={seed_hu:.2f}, not inside threshold range [{lower}, {upper}]')
    candidate = (volume >= lower) & (volume <= upper)
    structure = ndimage.generate_binary_structure(rank=3, connectivity=connectivity)
    labeled, num = ndimage.label(candidate, structure=structure)
    if num == 0:
        raise ValueError('No connected components found after thresholding.')
    target_label = labeled[x, y, z]
    if target_label == 0:
        raise ValueError('Seed is not inside any thresholded connected component.')
    region = labeled == target_label
    radius_mask = compute_distance_mask(volume.shape, seed, max_radius_vox)
    region = region & radius_mask
    if apply_morph_clean:
        region = ndimage.binary_closing(region, structure=structure, iterations=1)
        region = ndimage.binary_opening(region, structure=structure, iterations=1)
        region = ndimage.binary_fill_holes(region)
        labeled2, num2 = ndimage.label(region, structure=structure)
        if num2 == 0:
            raise ValueError('No connected component remains after morphology cleaning.')
        target_label2 = labeled2[x, y, z]
        if target_label2 == 0:
            raise ValueError('Seed component disappeared after morphology cleaning.')
        region = labeled2 == target_label2
        region = region & radius_mask
        labeled3, num3 = ndimage.label(region, structure=structure)
        if num3 == 0:
            raise ValueError('No connected component remains after radius recheck.')
        target_label3 = labeled3[x, y, z]
        if target_label3 == 0:
            raise ValueError('Seed component disappeared after radius recheck.')
        region = labeled3 == target_label3
    return (region.astype(np.uint8), seed_hu)

def compute_volume(mask, spacing):
    voxel_count = int(np.sum(mask > 0))
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    volume_mm3 = voxel_count * voxel_volume_mm3
    volume_cm3 = volume_mm3 / 1000.0
    return (voxel_count, voxel_volume_mm3, volume_mm3, volume_cm3)

def save_mask(mask, ref_header, out_path):
    new_header = dict(ref_header)
    new_header['type'] = 'uint8'
    nrrd.write(str(out_path), mask.astype(np.uint8), header=new_header)

def try_parse_scan_to_time(scan_label):
    """
    Fallback relative time if no real date is available.
    CT1 -> 0, CT2 -> 1, CT3 -> 2, ...
    """
    s = str(scan_label).strip().upper()
    if s.startswith('CT'):
        suffix = s[2:]
        if suffix.isdigit():
            return float(int(suffix) - 1)
    return np.nan

def build_ct_path(ct_file_value):
    """
    Support:
    - only filename: 100217_CT1.nrrd
    - relative path
    - absolute path
    """
    ct_file_str = str(ct_file_value).strip()
    p = Path(ct_file_str)
    if p.is_absolute():
        return p
    return DATA_DIR / ct_file_str

def status_from_volume(volume_cm3):
    if np.isnan(volume_cm3):
        return 'error'
    if volume_cm3 > MAX_REASONABLE_CM3:
        return 'volume_too_large'
    if volume_cm3 < MIN_REASONABLE_CM3:
        return 'volume_too_small'
    return 'ok'

def main():
    OUT_MASK_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        raise FileNotFoundError(f'CSV file not found: {CSV_PATH}')
    df = pd.read_csv(CSV_PATH)
    required_cols = ['patient_id', 'scan', 'ct_file', 'seed_x', 'seed_y', 'seed_z', 'lower_HU', 'upper_HU', 'max_radius_vox']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required CSV column: {col}')
    df['patient_id'] = df['patient_id'].astype(str).str.strip()
    df['scan'] = df['scan'].astype(str).str.strip()
    results = []
    for row_idx, row in df.iterrows():
        patient_id = str(row['patient_id'])
        scan = str(row['scan'])
        ct_file = str(row['ct_file']).strip()
        seed = (int(row['seed_x']), int(row['seed_y']), int(row['seed_z']))
        lower = float(row['lower_HU'])
        upper = float(row['upper_HU'])
        max_radius_vox = int(row['max_radius_vox'])
        ct_path = build_ct_path(ct_file)
        mask_filename = f'{patient_id}_{scan}_mask.nrrd'
        mask_path = OUT_MASK_DIR / mask_filename
        print('\n' + '=' * 70)
        print(f'[{row_idx + 1}/{len(df)}] Processing {patient_id} {scan}')
        print(f'CT file: {ct_path}')
        print(f'Seed (x,y,z): {seed}')
        print(f'Range: [{lower}, {upper}]')
        print(f'Max radius (voxels): {max_radius_vox}')
        base_result = {'patient_id': patient_id, 'scan': scan, 'ct_file': ct_file, 'ct_path': str(ct_path), 'seed_x': seed[0], 'seed_y': seed[1], 'seed_z': seed[2], 'lower_HU': lower, 'upper_HU': upper, 'max_radius_vox': max_radius_vox, 't_rel_used': try_parse_scan_to_time(scan), 'seed_HU': np.nan, 'spacing_x_mm': np.nan, 'spacing_y_mm': np.nan, 'spacing_z_mm': np.nan, 'voxel_count': np.nan, 'voxel_volume_mm3': np.nan, 'tumor_volume_mm3': np.nan, 'tumor_volume_cm3': np.nan, 'mask_file': '', 'status': 'error', 'error_message': ''}
        if not ct_path.exists():
            msg = f'file not found -> {ct_path}'
            print('ERROR:', msg)
            rec = dict(base_result)
            rec['status'] = 'file_not_found'
            rec['error_message'] = msg
            results.append(rec)
            continue
        try:
            volume, header = nrrd.read(str(ct_path))
            if volume.ndim != 3:
                raise ValueError(f'Expected 3D CT volume, got shape {volume.shape}')
            spacing = get_voxel_spacing(header)
            print(f'Volume shape: {volume.shape}')
            print(f'Spacing (mm): {spacing}')
            mask, seed_hu = region_growing_with_radius(volume=volume, seed=seed, lower=lower, upper=upper, max_radius_vox=max_radius_vox, connectivity=CONNECTIVITY, apply_morph_clean=APPLY_MORPH_CLEAN)
            voxel_count, voxel_volume_mm3, volume_mm3, volume_cm3 = compute_volume(mask, spacing)
            print(f'Seed HU value: {seed_hu:.2f}')
            print(f'Voxel count: {voxel_count}')
            print(f'Tumor volume: {volume_mm3:.2f} mm^3 = {volume_cm3:.4f} cm^3')
            save_mask(mask, header, mask_path)
            print(f'Saved mask: {mask_path}')
            rec = dict(base_result)
            rec.update({'seed_HU': seed_hu, 'spacing_x_mm': spacing[0], 'spacing_y_mm': spacing[1], 'spacing_z_mm': spacing[2], 'voxel_count': voxel_count, 'voxel_volume_mm3': voxel_volume_mm3, 'tumor_volume_mm3': volume_mm3, 'tumor_volume_cm3': volume_cm3, 'mask_file': str(mask_path), 'status': status_from_volume(volume_cm3), 'error_message': ''})
            results.append(rec)
        except Exception as e:
            msg = str(e)
            print('ERROR:', msg)
            rec = dict(base_result)
            rec['status'] = 'error'
            rec['error_message'] = msg
            results.append(rec)
    out_df = pd.DataFrame(results)
    if 'patient_id' in out_df.columns and 'scan' in out_df.columns:
        out_df = out_df.sort_values(['patient_id', 'scan']).reset_index(drop=True)
    out_df.to_csv(OUT_CSV_PATH, index=False)
    print('\n' + '=' * 70)
    print('Done.')
    print(f'Saved summary CSV: {OUT_CSV_PATH}')
    print(f'Saved masks folder: {OUT_MASK_DIR}')
    print('\nStatus counts:')
    print(out_df['status'].value_counts(dropna=False))
    bad_df = out_df[out_df['status'] != 'ok'].copy()
    if not bad_df.empty:
        bad_path = OUT_CSV_PATH.with_name('tumor_volumes_qc_flags.csv')
        bad_df.to_csv(bad_path, index=False)
        print(f'\nSaved QC review file: {bad_path}')
if __name__ == '__main__':
    main()
