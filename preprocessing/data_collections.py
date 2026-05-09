import os
import numpy as np
import pandas as pd
import pydicom
OLD_MANIFEST_ROOT = '/Users/zhiqiangkong/Desktop/DATA/manifest-1771618465869/NLST'
NEW_MANIFEST_ROOT = '/Users/zhiqiangkong/Desktop/DATA/manifest-1775421119183/NLST'
NEW_METADATA_CSV = '/Users/zhiqiangkong/Desktop/DATA/manifest-1775421119183/metadata.csv'
OUTPUT_FILE = '/Users/zhiqiangkong/Desktop/DATA/scan_level_collection_all30.csv'

def safe_get(ds, attr, default=None):
    return getattr(ds, attr, default)

def find_dicom_folders(root_dir):
    """
    Recursively find folders that directly contain .dcm files.
    """
    dicom_folders = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if any((f.lower().endswith('.dcm') for f in filenames)):
            dicom_folders.append(dirpath)
    return dicom_folders

def try_read_dicom_header(dicom_path):
    """
    Read DICOM header only (faster than loading pixel data).
    """
    return pydicom.dcmread(dicom_path, stop_before_pixels=True, force=True)

def list_dicom_files(dicom_folder):
    files = [os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.lower().endswith('.dcm')]
    return sorted(files)

def get_slice_position(ds):
    """
    For sorting slices by z position if possible.
    """
    ipp = safe_get(ds, 'ImagePositionPatient', None)
    if ipp is not None and len(ipp) >= 3:
        try:
            return float(ipp[2])
        except Exception:
            pass
    sl = safe_get(ds, 'SliceLocation', None)
    if sl is not None:
        try:
            return float(sl)
        except Exception:
            pass
    ins = safe_get(ds, 'InstanceNumber', None)
    if ins is not None:
        try:
            return float(ins)
        except Exception:
            pass
    return 0.0

def parse_patient_id_from_path(dicom_folder, manifest_root):
    """
    Assume patient folder is the first level under manifest_root.
    Example:
      manifest_root/.../NLST/100217/...
    """
    rel_path = os.path.relpath(dicom_folder, manifest_root)
    parts = rel_path.split(os.sep)
    if len(parts) > 0:
        return parts[0]
    return 'UNKNOWN'

def summarize_dicom_series(dicom_folder, manifest_name, manifest_root):
    """
    Build a scan-level / series-level summary from one DICOM folder.
    Header-only read for speed.
    """
    files = list_dicom_files(dicom_folder)
    if len(files) == 0:
        raise ValueError(f'No DICOM files found in folder: {dicom_folder}')
    headers = []
    for f in files:
        try:
            ds = try_read_dicom_header(f)
            headers.append(ds)
        except Exception:
            continue
    if len(headers) == 0:
        raise ValueError(f'All DICOM headers failed to read in folder: {dicom_folder}')
    headers.sort(key=get_slice_position)
    ds0 = headers[0]
    patient_id = parse_patient_id_from_path(dicom_folder, manifest_root)
    pixel_spacing = safe_get(ds0, 'PixelSpacing', [None, None])
    spacing_dx = None
    spacing_dy = None
    if pixel_spacing is not None and len(pixel_spacing) >= 2:
        try:
            spacing_dx = float(pixel_spacing[0])
            spacing_dy = float(pixel_spacing[1])
        except Exception:
            pass
    slice_thickness = safe_get(ds0, 'SliceThickness', None)
    try:
        spacing_dz = float(slice_thickness) if slice_thickness is not None else None
    except Exception:
        spacing_dz = None
    acquisition_date = safe_get(ds0, 'AcquisitionDate', None)
    acquisition_time = safe_get(ds0, 'AcquisitionTime', None)
    study_date = safe_get(ds0, 'StudyDate', None)
    series_date = safe_get(ds0, 'SeriesDate', None)
    study_uid = safe_get(ds0, 'StudyInstanceUID', None)
    series_uid = safe_get(ds0, 'SeriesInstanceUID', None)
    series_number = safe_get(ds0, 'SeriesNumber', None)
    accession_number = safe_get(ds0, 'AccessionNumber', None)
    modality = safe_get(ds0, 'Modality', None)
    manufacturer = safe_get(ds0, 'Manufacturer', None)
    series_description = safe_get(ds0, 'SeriesDescription', None)
    study_description = safe_get(ds0, 'StudyDescription', None)
    convolution_kernel = safe_get(ds0, 'ConvolutionKernel', None)
    rows = safe_get(ds0, 'Rows', None)
    cols = safe_get(ds0, 'Columns', None)
    record = {'manifest_name': manifest_name, 'patient_id': str(patient_id), 'dicom_folder': dicom_folder, 'num_slices': len(headers), 'rows': rows, 'cols': cols, 'spacing_dx_mm': spacing_dx, 'spacing_dy_mm': spacing_dy, 'spacing_dz_mm': spacing_dz, 'modality': modality, 'manufacturer': manufacturer, 'study_date': study_date, 'series_date': series_date, 'acquisition_date': acquisition_date, 'acquisition_time': acquisition_time, 'study_uid': study_uid, 'series_uid': series_uid, 'series_number': series_number, 'accession_number': accession_number, 'study_description': study_description, 'series_description': series_description, 'convolution_kernel': str(convolution_kernel) if convolution_kernel is not None else None}
    return record

def load_metadata_if_exists(csv_path):
    if csv_path is None or not os.path.isfile(csv_path):
        print(f'[INFO] Metadata file not found or skipped: {csv_path}')
        return None
    try:
        meta = pd.read_csv(csv_path)
        print(f'[INFO] Loaded metadata: {csv_path}, shape={meta.shape}')
        print(f'[INFO] Metadata columns: {list(meta.columns)}')
        return meta
    except Exception as e:
        print(f'[WARNING] Failed to read metadata file: {csv_path}')
        print('Error:', repr(e))
        return None

def normalize_date_str(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    if x == '' or x.lower() == 'none':
        return None
    return x

def main():
    manifest_configs = [{'manifest_name': 'old_manifest_5patients', 'root': OLD_MANIFEST_ROOT, 'metadata_csv': None}, {'manifest_name': 'new_manifest_25patients', 'root': NEW_MANIFEST_ROOT, 'metadata_csv': NEW_METADATA_CSV}]
    all_records = []
    all_metadata = []
    for cfg in manifest_configs:
        manifest_name = cfg['manifest_name']
        root_dir = cfg['root']
        metadata_csv = cfg['metadata_csv']
        print('\n' + '=' * 80)
        print(f'Processing manifest: {manifest_name}')
        print(f'Root: {root_dir}')
        print('=' * 80)
        if not os.path.isdir(root_dir):
            print(f'[WARNING] Root does not exist, skipped: {root_dir}')
            continue
        dicom_folders = find_dicom_folders(root_dir)
        print(f'[INFO] Number of DICOM folders detected: {len(dicom_folders)}')
        for i, folder in enumerate(sorted(dicom_folders), start=1):
            try:
                rec = summarize_dicom_series(folder, manifest_name, root_dir)
                all_records.append(rec)
                print(f"[{i}/{len(dicom_folders)}] patient={rec['patient_id']} | slices={rec['num_slices']} | date={rec['acquisition_date'] or rec['study_date']} | series_uid={(str(rec['series_uid'])[-8:] if rec['series_uid'] else None)}")
            except Exception as e:
                print(f'[WARNING] Failed folder: {folder}')
                print('Error:', repr(e))
        meta = load_metadata_if_exists(metadata_csv)
        if meta is not None:
            meta['manifest_name'] = manifest_name
            all_metadata.append(meta)
    if len(all_records) == 0:
        print('[ERROR] No valid DICOM series found.')
        return
    df = pd.DataFrame(all_records)
    if 'modality' in df.columns:
        before_ct = len(df)
        df = df[df['modality'].astype(str).str.upper() == 'CT'].copy()
        print(f'\n[INFO] CT filter: {before_ct} -> {len(df)} rows')
    for c in ['patient_id', 'study_date', 'series_date', 'acquisition_date', 'acquisition_time']:
        if c in df.columns:
            df[c] = df[c].apply(normalize_date_str)
    sort_cols = [c for c in ['patient_id', 'acquisition_date', 'acquisition_time', 'study_date', 'series_number'] if c in df.columns]
    if len(sort_cols) > 0:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    df['scan_index_within_patient'] = df.groupby('patient_id').cumcount() + 1
    if len(all_metadata) > 0:
        meta_all = pd.concat(all_metadata, ignore_index=True)
        candidate_patient_cols = [c for c in meta_all.columns if 'patient' in c.lower() and 'id' in c.lower()]
        if len(candidate_patient_cols) > 0:
            meta_pid_col = candidate_patient_cols[0]
            meta_all[meta_pid_col] = meta_all[meta_pid_col].astype(str)
            rename_map = {}
            for c in meta_all.columns:
                if c in df.columns and c not in ['manifest_name']:
                    rename_map[c] = f'meta_{c}'
            meta_all = meta_all.rename(columns=rename_map)
            left_on = ['manifest_name', 'patient_id']
            right_patient_col = f'meta_{meta_pid_col}' if meta_pid_col in rename_map else meta_pid_col
            right_on = ['manifest_name', right_patient_col]
            try:
                df = df.merge(meta_all, how='left', left_on=left_on, right_on=right_on)
                print(f'[INFO] Metadata merged by patient id column: {meta_pid_col}')
            except Exception as e:
                print('[WARNING] Metadata merge failed.')
                print('Error:', repr(e))
        else:
            print('[WARNING] Could not find patient_id-like column in metadata; skipped merge.')
    patient_summary = df.groupby('patient_id').agg(manifest_name=('manifest_name', 'first'), num_series=('series_uid', 'count'), min_date=('acquisition_date', 'min'), max_date=('acquisition_date', 'max')).reset_index()
    print('\n' + '=' * 80)
    print('[INFO] Final summary')
    print('=' * 80)
    print(f'Total rows (series-level): {len(df)}')
    print(f"Total unique patients: {df['patient_id'].nunique()}")
    print('\nPatient counts by manifest:')
    print(df.groupby('manifest_name')['patient_id'].nunique())
    print('\nFirst few rows:')
    print(df.head())
    df.to_csv(OUTPUT_FILE, index=False)
    print(f'\n[INFO] Saved scan-level summary to:\n{OUTPUT_FILE}')
    patient_summary_file = OUTPUT_FILE.replace('.csv', '_patient_summary.csv')
    patient_summary.to_csv(patient_summary_file, index=False)
    print(f'[INFO] Saved patient summary to:\n{patient_summary_file}')
if __name__ == '__main__':
    main()
