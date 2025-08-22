import json
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump

DATA = Path("data/processed")
MODELS = Path("models")
MODELS.mkdir(parents=True, exist_ok=True)

Xtr = np.load(DATA / "X_train.npy")
ytr = np.load(DATA / "y_train.npy")
Xte = np.load(DATA / "X_test.npy")
yte = np.load(DATA / "y_test.npy")

pipe = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, multi_class="auto")),
    ]
)
pipe.fit(Xtr, ytr)

train_acc = float(accuracy_score(ytr, pipe.predict(Xtr)))
test_acc = float(accuracy_score(yte, pipe.predict(Xte)))

dump(pipe, MODELS / "model.joblib")

with open("metrics.json", "w") as f:
    json.dump({"train_accuracy": train_acc, "test_accuracy": test_acc}, f, indent=2)

print("Model saved to models/model.joblib")
print("Metrics:", {"train_accuracy": train_acc, "test_accuracy": test_acc})
