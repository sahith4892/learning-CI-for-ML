import json
from pathlib import Path
import numpy as np
from joblib import load
from sklearn.metrics import accuracy_score, classification_report

DATA = Path("data/processed")
MODEL_PATH = Path("models/model.joblib")

Xte = np.load(DATA / "X_test.npy")
yte = np.load(DATA / "y_test.npy")

model = load(MODEL_PATH)
pred = model.predict(Xte)

acc = float(accuracy_score(yte, pred))
report = classification_report(yte, pred, output_dict=True)

with open("eval.json", "w") as f:
    json.dump({"accuracy": acc, "report": report}, f, indent=2)

print("Test accuracy:", acc)
