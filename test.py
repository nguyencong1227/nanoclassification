import pandas as pd
from pathlib import Path

df = pd.read_csv("evaluation_results_ALL.csv")

def make_model_name(row):
    p = Path(row["model_file"])
    stem = p.stem
    parent = p.parent.name

    base = f"{parent}/{stem}" if parent.lower() != "model" else stem
    feat = str(row.get("feature_used", ""))

    if feat and feat != "Baseline34":
        return f"{base} ({feat})"
    return base

summary = (
    df.assign(model_name=df.apply(make_model_name, axis=1))
      [["model_name", "precision_macro", "recall_macro", "f1_macro", "accuracy"]]
      .rename(columns={
          "precision_macro": "precision",
          "recall_macro": "recall",
          "f1_macro": "f1-score",
          "accuracy": "accurancy",
      })
      .sort_values("accurancy", ascending=False)
      .reset_index(drop=True)
      .round(3)
)

print(summary)

summary.to_csv("summary_metrics.csv", index=False)
print("Saved: summary_metrics.csv")
