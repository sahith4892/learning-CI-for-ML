import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW = Path("data/raw/iris_tiny.csv")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW)
X = df.drop(columns=["target"]).to_numpy(dtype=np.float32)
y = df["target"].to_numpy(dtype=np.int64)

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y
)

np.save(OUT / "X_train.npy", Xtr)
np.save(OUT / "y_train.npy", ytr)
np.save(OUT / "X_test.npy", Xte)
np.save(OUT / "y_test.npy", yte)

print(f"Saved processed arrays to {OUT}")
