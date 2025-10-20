import os
import pandas as pd
from sklearn.model_selection import train_test_split

# =============================================================
# CONFIGURATION
# =============================================================
base_dir = '/Users/nehajacob/Desktop/YourFolder/sem5/proj/ddm/AlloyBERT/data/datapackets'
data_csv = os.path.join(base_dir, 'ys_text_ready_noleak.csv')

# Identify columns
text_col = 'text'
label_candidates = ['YS_Class', 'label', 'class', 'target']
label_col = None

# =============================================================
# LOAD DATA
# =============================================================
print(f"📂 Reading dataset: {data_csv}")
df = pd.read_csv(data_csv)

# Auto-detect label column
for c in label_candidates:
    if c in df.columns:
        label_col = c
        break

if label_col is None:
    raise ValueError(f"No label column found. Expected one of {label_candidates}")

if text_col not in df.columns:
    raise ValueError(f"No 'text' column found in {data_csv}")

print(f"✅ Using text column: '{text_col}', label column: '{label_col}'")

# =============================================================
# SPLITTING LOGIC
# =============================================================
# 70% train, 10% validation, 20% test
train_val, test = train_test_split(
    df, test_size=0.20, random_state=42, shuffle=True, stratify=df[label_col]
)

val_fraction = 10 / 80  # 10% of total, inside remaining 80%
train, val = train_test_split(
    train_val, test_size=val_fraction, random_state=42, shuffle=True, stratify=train_val[label_col]
)

# =============================================================
# SAVE OUTPUTS
# =============================================================
for name, split in [('train', train), ('val', val), ('test', test)]:
    csv_path = os.path.join(base_dir, f'{name}.csv')
    pkl_path = os.path.join(base_dir, f'{name}.pkl')
    split.to_csv(csv_path, index=False)
    split.to_pickle(pkl_path)
    print(f"💾 Saved: {csv_path}  ({len(split)} rows)")

print(f"""
✅ Splitting complete!

Folder: {base_dir}
Created files:
  - train.csv, val.csv, test.csv
  - train.pkl, val.pkl, test.pkl
""")
