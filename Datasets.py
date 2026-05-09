import os
import numpy as np
import pandas as pd
import torch

def load_tumor_volume_table(csv_path: str, keep_only_ok: bool=True):
    """
    Load the full tumor_volumes.csv table.

    Required columns:
        - patient_id
        - scan
        - tumor_volume_cm3

    Optional columns:
        - status
        - t_rel_used
        - t_rel / time_days / days_from_baseline / time_months
        - scan_date
        - ct_file / mask_file / seed_x / seed_y / seed_z ...
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'File not found: {csv_path}')
    df = pd.read_csv(csv_path)
    required_cols = {'patient_id', 'scan', 'tumor_volume_cm3'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f'CSV must contain columns {required_cols}, but found {df.columns.tolist()}')
    if keep_only_ok and 'status' in df.columns:
        df = df[df['status'].astype(str).str.lower().str.strip() == 'ok'].copy()
    if df.empty:
        raise ValueError('No valid rows found in CSV after filtering.')
    df['patient_id'] = df['patient_id'].astype(str).str.strip()
    df['scan'] = df['scan'].astype(str).str.strip()
    df['tumor_volume_cm3'] = pd.to_numeric(df['tumor_volume_cm3'], errors='coerce')
    if df['tumor_volume_cm3'].isna().any():
        bad_rows = df[df['tumor_volume_cm3'].isna()]
        raise ValueError(f'Found invalid tumor_volume_cm3 values in rows:\n{bad_rows}')
    if 'scan_date' in df.columns:
        df['scan_date'] = pd.to_datetime(df['scan_date'], errors='coerce')
    return df.reset_index(drop=True)

def get_available_patient_ids(csv_path: str, keep_only_ok: bool=True):
    df = load_tumor_volume_table(csv_path, keep_only_ok=keep_only_ok)
    return sorted(df['patient_id'].unique().tolist())

def _assign_time_column(sub: pd.DataFrame):
    """
    Priority:
    1. Use existing real-time column if available
    2. Use scan_date if available
    3. Otherwise map CT1/CT2/CT3/... -> 0/1/2/...
    """
    sub = sub.copy()
    for col in ['t_rel_used', 't_rel', 'time_days', 'days_from_baseline', 'time_months']:
        if col in sub.columns:
            sub['t_rel_used'] = pd.to_numeric(sub[col], errors='coerce')
            if sub['t_rel_used'].isna().any():
                raise ValueError(f'Column {col} contains invalid time values.')
            return sub
    if 'scan_date' in sub.columns and sub['scan_date'].notna().all():
        sub = sub.sort_values('scan_date').copy()
        baseline_date = sub['scan_date'].iloc[0]
        sub['t_rel_used'] = (sub['scan_date'] - baseline_date).dt.days.astype(np.float32)
        return sub

    def scan_to_num(scan_label):
        s = str(scan_label).strip().upper()
        if s.startswith('CT'):
            suffix = s[2:]
            if suffix.isdigit():
                return float(int(suffix) - 1)
        return np.nan
    sub['t_rel_used'] = sub['scan'].apply(scan_to_num)
    if sub['t_rel_used'].isna().any():
        bad_scans = sorted(sub.loc[sub['t_rel_used'].isna(), 'scan'].astype(str).unique())
        raise ValueError(f'Unexpected scan labels: {bad_scans}. Please provide a real time column such as t_rel_used or scan_date.')
    return sub

def load_patient_timeseries(csv_path: str, patient_id=None, keep_only_ok: bool=True):
    """
    Load one patient's longitudinal tumor volume time series.

    Returns:
        patient_id : str
        t_tensor   : torch.FloatTensor of shape (n, 1)
        y_tensor   : torch.FloatTensor of shape (n, 1), where y = log(volume)
        sub        : processed dataframe for that patient
    """
    df = load_tumor_volume_table(csv_path, keep_only_ok=keep_only_ok)
    if patient_id is None:
        patient_id = df['patient_id'].iloc[0]
    else:
        patient_id = str(patient_id)
    sub = df[df['patient_id'] == patient_id].copy()
    if sub.empty:
        raise ValueError(f'No data found for patient_id={patient_id}')
    sub = _assign_time_column(sub)
    sub['V_obs'] = sub['tumor_volume_cm3'].astype(np.float32)
    if (sub['V_obs'] <= 0).any():
        show_cols = ['patient_id', 'V_obs']
        if 'scan' in sub.columns:
            show_cols.insert(1, 'scan')
        bad_vals = sub.loc[sub['V_obs'] <= 0, show_cols]
        raise ValueError(f'Non-positive tumor volumes found for patient_id={patient_id}:\n{bad_vals}')
    sub = sub.sort_values('t_rel_used').reset_index(drop=True)
    if sub['t_rel_used'].duplicated().any():
        raise ValueError(f'Duplicate time points found for patient_id={patient_id}. Please check t_rel_used / scan_date.')
    if len(sub) < 3:
        print(f'Warning: patient_id={patient_id} has only {len(sub)} valid scan(s). Current study expects at least 3.')
    t = sub['t_rel_used'].to_numpy(dtype=np.float32)
    v = sub['V_obs'].to_numpy(dtype=np.float32)
    y = np.log(v)
    t_tensor = torch.tensor(t, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    return (patient_id, t_tensor, y_tensor, sub)

def patient_df_to_tensors(sub: pd.DataFrame):
    """
    Convert one processed patient dataframe into tensors.
    Expects:
      - t_rel_used
      - V_obs
    """
    if 't_rel_used' not in sub.columns or 'V_obs' not in sub.columns:
        raise ValueError('Dataframe must contain t_rel_used and V_obs columns.')
    t = sub['t_rel_used'].to_numpy(dtype=np.float32)
    v = sub['V_obs'].to_numpy(dtype=np.float32)
    y = np.log(v)
    t_tensor = torch.tensor(t, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    return (t_tensor, y_tensor)

def split_patient_train_holdout(csv_path: str, patient_id=None, keep_only_ok: bool=True, holdout_rule: str='last_timepoint'):
    """
    Split one patient's longitudinal series into:
      - training subset
      - holdout final observation

    Current supported rule:
      - last_timepoint
    """
    patient_id, _, _, sub = load_patient_timeseries(csv_path, patient_id=patient_id, keep_only_ok=keep_only_ok)
    if holdout_rule != 'last_timepoint':
        raise ValueError(f'Unsupported holdout_rule: {holdout_rule}')
    if len(sub) < 2:
        raise ValueError(f'Need at least 2 observations for train/holdout split, got {len(sub)}')
    train_df = sub.iloc[:-1].copy().reset_index(drop=True)
    holdout_df = sub.iloc[[-1]].copy().reset_index(drop=True)
    t_train, y_train = patient_df_to_tensors(train_df)
    t_holdout, y_holdout = patient_df_to_tensors(holdout_df)
    return {'patient_id': str(patient_id), 'full_df': sub.copy(), 'train_df': train_df, 'holdout_df': holdout_df, 't_train': t_train, 'y_train': y_train, 't_holdout': t_holdout, 'y_holdout': y_holdout, 'holdout_t': float(t_holdout.view(-1)[0].item()), 'holdout_y': float(y_holdout.view(-1)[0].item()), 'holdout_V': float(holdout_df['V_obs'].iloc[0]), 'holdout_scan': str(holdout_df['scan'].iloc[0]) if 'scan' in holdout_df.columns else ''}

def load_all_patient_timeseries(csv_path: str, keep_only_ok: bool=True, min_scans: int=3):
    """
    Load all patients into a dictionary.

    Returns:
        cohort_dict[patient_id] = {
            "t_tensor": ...,
            "y_tensor": ...,
            "sub_df": ...
        }
        skipped = list of (patient_id, reason)
    """
    patient_ids = get_available_patient_ids(csv_path, keep_only_ok=keep_only_ok)
    cohort_dict = {}
    skipped = []
    for pid in patient_ids:
        try:
            patient_id, t_tensor, y_tensor, sub = load_patient_timeseries(csv_path, patient_id=pid, keep_only_ok=keep_only_ok)
            if len(sub) < min_scans:
                skipped.append((patient_id, f'only {len(sub)} scans'))
                continue
            cohort_dict[patient_id] = {'t_tensor': t_tensor, 'y_tensor': y_tensor, 'sub_df': sub}
        except Exception as e:
            skipped.append((pid, str(e)))
    return (cohort_dict, skipped)
if __name__ == '__main__':
    csv_path = '/Users/zhiqiangkong/Desktop/DATA/tumor_volumes.csv'
    patient_ids = get_available_patient_ids(csv_path)
    print('Available patient IDs:', patient_ids)
    print('Number of patients:', len(patient_ids))
    cohort_dict, skipped = load_all_patient_timeseries(csv_path, keep_only_ok=True, min_scans=3)
    print('\nLoaded patients:', len(cohort_dict))
    print('Skipped patients:', len(skipped))
    if skipped:
        print('Skipped details:')
        for item in skipped:
            print('  ', item)
    for pid, obj in cohort_dict.items():
        sub_df = obj['sub_df']
        print('\n' + '=' * 60)
        print('Patient ID:', pid)
        print('Number of samples:', len(sub_df))
        cols_to_show = ['patient_id', 'scan', 't_rel_used', 'V_obs']
        if 'scan_date' in sub_df.columns:
            cols_to_show.insert(2, 'scan_date')
        print(sub_df[cols_to_show])
        print('t_tensor shape:', tuple(obj['t_tensor'].shape))
        print('y_tensor shape:', tuple(obj['y_tensor'].shape))
