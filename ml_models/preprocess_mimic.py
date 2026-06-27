"""
preprocess_mimic.py
MIMIC-IV data preprocessing - extracts Heart Rate, SpO2, Temperature
Dynamically looks up itemids from d_items.csv
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "mimic-iv-clinical-database-demo-2.2"
ICU_DIR = DATA_DIR / "icu"
OUTPUT_DIR = Path(__file__).resolve().parent

CHARTEVENTS_PATH = ICU_DIR / "chartevents.csv"
D_ITEMS_PATH = ICU_DIR / "d_items.csv"
OUTPUT_PATH = OUTPUT_DIR / "processed_vitals.csv"

# Medical thresholds for labeling
THRESHOLDS = {
    'heart_rate': {'min': 60, 'max': 100},
    'spo2': {'min': 95, 'max': 100},
    'temperature': {'min': 36.1, 'max': 37.8}
}


def load_d_items():
    """Load d_items.csv and extract relevant itemids dynamically."""
    print(f"[PREPROCESS] Loading d_items from {D_ITEMS_PATH}")
    
    if not D_ITEMS_PATH.exists():
        raise FileNotFoundError(f"d_items.csv not found at {D_ITEMS_PATH}")
    
    d_items = pd.read_csv(D_ITEMS_PATH)
    print(f"[PREPROCESS] Loaded {len(d_items)} items from d_items.csv")
    print(f"[PREPROCESS] Columns: {list(d_items.columns)}")
    
    # Convert label to lowercase for matching
    d_items['label_lower'] = d_items['label'].astype(str).str.lower()
    
    # Dynamic itemid lookup
    itemids = {
        'heart_rate': [],
        'spo2': [],
        'temperature': []
    }
    
    # Heart Rate patterns
    hr_patterns = ['heart rate', 'hr ', 'pulse rate']
    for pattern in hr_patterns:
        matches = d_items[d_items['label_lower'].str.contains(pattern, na=False, regex=False)]
        itemids['heart_rate'].extend(matches['itemid'].tolist())
    
    # SpO2 patterns
    spo2_patterns = ['spo2', 'oxygen saturation', 'o2 sat', 'pulse oximetry']
    for pattern in spo2_patterns:
        matches = d_items[d_items['label_lower'].str.contains(pattern, na=False, regex=False)]
        itemids['spo2'].extend(matches['itemid'].tolist())
    
    # Temperature patterns (exclude non-body temperatures)
    temp_matches = d_items[d_items['label_lower'].str.contains('temperature', na=False, regex=False)]
    for _, row in temp_matches.iterrows():
        label = row['label_lower']
        if 'room' not in label and 'ambient' not in label and 'skin' not in label:
            itemids['temperature'].append(row['itemid'])
    
    # Remove duplicates
    for key in itemids:
        itemids[key] = list(set(itemids[key]))
    
    print(f"[PREPROCESS] Found itemids:")
    print(f"  Heart Rate: {itemids['heart_rate']}")
    print(f"  SpO2: {itemids['spo2']}")
    print(f"  Temperature: {itemids['temperature']}")
    
    return itemids, d_items


def load_and_filter_chartevents(itemids):
    """Load chartevents.csv and filter for relevant vital signs."""
    print(f"[PREPROCESS] Loading chartevents from {CHARTEVENTS_PATH}")
    
    if not CHARTEVENTS_PATH.exists():
        raise FileNotFoundError(f"chartevents.csv not found at {CHARTEVENTS_PATH}")
    
    # Combine all itemids
    all_itemids = set()
    for ids in itemids.values():
        all_itemids.update(ids)
    
    print(f"[PREPROCESS] Filtering for {len(all_itemids)} itemids")
    
    # Read chartevents in chunks for memory efficiency
    chunks = []
    chunk_size = 100000
    total_filtered = 0
    
    for chunk in pd.read_csv(CHARTEVENTS_PATH, chunksize=chunk_size, low_memory=False):
        filtered = chunk[chunk['itemid'].isin(all_itemids)]
        if len(filtered) > 0:
            chunks.append(filtered)
            total_filtered += len(filtered)
        print(f"  Processed chunk, total filtered rows: {total_filtered}", end='\r')
    
    print(f"\n[PREPROCESS] Total filtered rows: {total_filtered}")
    
    if not chunks:
        raise ValueError("No matching vital signs found in chartevents.csv")
    
    df = pd.concat(chunks, ignore_index=True)
    return df


def process_vitals(df, itemids):
    """Process and structure vital signs data."""
    print("[PREPROCESS] Processing vital signs...")
    
    # Convert valuenum to numeric
    df['valuenum'] = pd.to_numeric(df['valuenum'], errors='coerce')
    
    # Drop rows with missing values
    df = df.dropna(subset=['valuenum'])
    
    # Add vital type column
    def get_vital_type(itemid):
        for vital_type, ids in itemids.items():
            if itemid in ids:
                return vital_type
        return None
    
    df['vital_type'] = df['itemid'].apply(get_vital_type)
    df = df.dropna(subset=['vital_type'])
    
    # Convert charttime to datetime
    df['charttime'] = pd.to_datetime(df['charttime'], errors='coerce')
    df = df.dropna(subset=['charttime'])
    
    # Filter physiologically valid values
    valid_mask = (
        ((df['vital_type'] == 'heart_rate') & (df['valuenum'] >= 20) & (df['valuenum'] <= 250)) |
        ((df['vital_type'] == 'spo2') & (df['valuenum'] >= 50) & (df['valuenum'] <= 100)) |
        ((df['vital_type'] == 'temperature') & (df['valuenum'] >= 30) & (df['valuenum'] <= 45))
    )
    df = df[valid_mask]
    
    print(f"[PREPROCESS] Valid records: {len(df)}")
    
    # Pivot to wide format
    pivot_df = df.pivot_table(
        index=['subject_id', 'charttime'],
        columns='vital_type',
        values='valuenum',
        aggfunc='mean'
    ).reset_index()
    
    # Flatten column names
    pivot_df.columns = ['subject_id', 'charttime'] + [col for col in pivot_df.columns[2:]]
    
    # Ensure all required columns exist
    for col in ['heart_rate', 'spo2', 'temperature']:
        if col not in pivot_df.columns:
            pivot_df[col] = np.nan
    
    # Sort by subject_id and charttime
    pivot_df = pivot_df.sort_values(['subject_id', 'charttime']).reset_index(drop=True)
    
    print(f"[PREPROCESS] Pivoted data shape: {pivot_df.shape}")
    
    return pivot_df


def handle_missing_values(df):
    """Handle missing values using forward fill and interpolation."""
    print("[PREPROCESS] Handling missing values...")
    
    # Count missing before
    missing_before = df[['heart_rate', 'spo2', 'temperature']].isna().sum()
    print(f"  Missing before: HR={missing_before['heart_rate']}, SpO2={missing_before['spo2']}, Temp={missing_before['temperature']}")
    
    # Forward fill within each subject
    df = df.copy()
    for col in ['heart_rate', 'spo2', 'temperature']:
        df[col] = df.groupby('subject_id')[col].transform(
            lambda x: x.fillna(method='ffill').fillna(method='bfill')
        )
    
    # Fill remaining with column median
    for col in ['heart_rate', 'spo2', 'temperature']:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    
    # Count missing after
    missing_after = df[['heart_rate', 'spo2', 'temperature']].isna().sum()
    print(f"  Missing after: HR={missing_after['heart_rate']}, SpO2={missing_after['spo2']}, Temp={missing_after['temperature']}")
    
    return df


def create_labels(df):
    """Create abnormal/normal labels based on medical thresholds."""
    print("[PREPROCESS] Creating labels...")
    
    df = df.copy()
    
    # Check each vital against thresholds
    hr_abnormal = (df['heart_rate'] < THRESHOLDS['heart_rate']['min']) | \
                  (df['heart_rate'] > THRESHOLDS['heart_rate']['max'])
    
    spo2_abnormal = df['spo2'] < THRESHOLDS['spo2']['min']
    
    temp_abnormal = (df['temperature'] < THRESHOLDS['temperature']['min']) | \
                    (df['temperature'] > THRESHOLDS['temperature']['max'])
    
    # Label as abnormal (1) if any vital is abnormal
    df['label'] = ((hr_abnormal) | (spo2_abnormal) | (temp_abnormal)).astype(int)
    
    # Statistics
    n_normal = (df['label'] == 0).sum()
    n_abnormal = (df['label'] == 1).sum()
    print(f"  Normal: {n_normal} ({100*n_normal/len(df):.1f}%)")
    print(f"  Abnormal: {n_abnormal} ({100*n_abnormal/len(df):.1f}%)")
    
    return df


def main():
    """Main preprocessing pipeline."""
    print("=" * 60)
    print("MIMIC-IV Data Preprocessing")
    print("=" * 60)
    
    # Step 1: Load d_items and get itemids
    itemids, d_items = load_d_items()
    
    # Step 2: Load and filter chartevents
    chartevents = load_and_filter_chartevents(itemids)
    
    # Step 3: Process vitals
    processed = process_vitals(chartevents, itemids)
    
    # Step 4: Handle missing values
    processed = handle_missing_values(processed)
    
    # Step 5: Create labels
    processed = create_labels(processed)
    
    # Step 6: Save processed data
    print(f"[PREPROCESS] Saving to {OUTPUT_PATH}")
    processed.to_csv(OUTPUT_PATH, index=False)
    
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Shape: {processed.shape}")
    print("=" * 60)
    
    # Print summary statistics
    print("\nData Statistics:")
    for col in ['heart_rate', 'spo2', 'temperature']:
        print(f"  {col}: mean={processed[col].mean():.2f}, std={processed[col].std():.2f}, "
              f"min={processed[col].min():.2f}, max={processed[col].max():.2f}")


if __name__ == "__main__":
    main()