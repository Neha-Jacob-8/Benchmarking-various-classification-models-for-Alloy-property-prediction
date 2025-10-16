import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------- CONFIG ----------------------
dataset = 'MPEA'  # folder name under ./data/
data_csv = f'./data/{dataset}/{dataset}.csv'

# target column logic (as per your code)
# NOTE: your example used 'PROPERTY: Calculated Young modulus (GPa)' as default
if dataset == 'ys_clean':
    targetCol = 'YS'
else:
    targetCol = 'PROPERTY: Calculated Young modulus (GPa)'

# split params
TEST_SIZE = 0.20           # 20% test
VAL_FRAC_WITHIN_TRAIN = 0.10  # 10% of the 80% train -> 8% overall
RANDOM_STATE = 42
# ---------------------------------------------------

# load
data = pd.read_csv(data_csv)

# normalize target to [0, 1] by max (same as your code)
data[targetCol] = data[targetCol] / data[targetCol].max()

# first split: train+val vs test
train_val, te = train_test_split(
    data,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True
)

# second split: train vs val (within the 80% chunk)
val_size = VAL_FRAC_WITHIN_TRAIN  # fraction of current train_val
tr, vl = train_test_split(
    train_val,
    test_size=val_size,
    random_state=RANDOM_STATE,
    shuffle=True
)

# save raw splits as CSV
out_dir = f'./data/{dataset}'
os.makedirs(out_dir, exist_ok=True)
tr.to_csv(f'{out_dir}/tr.csv', index=False)
vl.to_csv(f'{out_dir}/vl.csv', index=False)
te.to_csv(f'{out_dir}/te.csv', index=False)

print(f"Split sizes: tr={len(tr)} ({len(tr)/len(data):.2%}), "
      f"vl={len(vl)} ({len(vl)/len(data):.2%}), "
      f"te={len(te)} ({len(te)/len(data):.2%})")

# build text columns in the same format you used: "k: v. "
def rows_to_text(df, target_col):
    X = df.drop(columns=[target_col])
    texts = []
    for _, row in X.iterrows():
        s = ''
        for k, v in row.items():
            s += f'{k}: {v}. '
        texts.append(s)
    return texts

# create (text, target) DataFrames
df_train = pd.DataFrame({
    'text': rows_to_text(tr, targetCol),
    'target': tr[targetCol].tolist()
})
df_val = pd.DataFrame({
    'text': rows_to_text(vl, targetCol),
    'target': vl[targetCol].tolist()
})
df_test = pd.DataFrame({
    'text': rows_to_text(te, targetCol),
    'target': te[targetCol].tolist()
})

# save PKL to match your training/validation style
df_train.to_pickle(f'{out_dir}/tr.pkl')
df_val.to_pickle(f'{out_dir}/vl.pkl')
df_test.to_pickle(f'{out_dir}/te.pkl')

print(f"Saved:\n  {out_dir}/tr.csv, {out_dir}/vl.csv, {out_dir}/te.csv\n  "
      f"{out_dir}/tr.pkl, {out_dir}/vl.pkl, {out_dir}/te.pkl")
