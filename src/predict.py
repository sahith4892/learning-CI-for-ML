import argparse, json
import pandas as pd
from joblib import load

p = argparse.ArgumentParser()
p.add_argument("--input", required=True, help="CSV with feature columns")
p.add_argument("--model", default="models/model.joblib")
p.add_argument("--out", default="predictions.json")
args = p.parse_args()

df = pd.read_csv(args.input)
X = df.to_numpy(dtype=float)

model = load(args.model)
pred = model.predict(X).tolist()

with open(args.out, "w") as f:
    json.dump({"predictions": pred}, f, indent=2)

print(f"Predictions saved to {args.out}")
