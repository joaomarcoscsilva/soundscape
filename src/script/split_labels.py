import pandas as pd
import os

from sklearn.model_selection import train_test_split

from soundscape.lib.settings import settings

df = pd.read_csv(os.path.join(settings["data"]["data_dir"], "labels.csv"))

test_frac = settings["data"]["split"]["test_frac"]
val_frac = settings["data"]["split"]["val_frac"]
train_frac = 1 - test_frac - val_frac

test_files = int(test_frac * len(df))
val_files = int(val_frac * len(df))
train_files = len(df) - test_files - val_files

idx = df['index']
y = df['class']

idx_train, idx_test, y_train, y_test = train_test_split(idx, y, test_size=test_files, random_state=0, stratify=y)
idx_train, idx_val, y_train, y_val = train_test_split(idx_train, y_train, test_size=val_files, random_state=1, stratify=y_train)

train_df = df.iloc[idx_train]
val_df = df.iloc[idx_val]
test_df = df.iloc[idx_test]

train_df.to_csv(
    os.path.join(settings["data"]["data_dir"], "train_labels.csv"),
    index=False,
)
val_df.to_csv(
    os.path.join(settings["data"]["data_dir"], "val_labels.csv"),
    index=False,
)
test_df.to_csv(
    os.path.join(settings["data"]["data_dir"], "test_labels.csv"),
    index=False,
)

print('Split')