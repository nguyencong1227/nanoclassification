import pandas as pd
from pathlib import Path

# Đọc kết quả evaluate
df = pd.read_csv("evaluation_results_ALL.csv")

# Tạo model_name (kèm feature_used để tránh trùng tên và nhìn rõ model chạy trên feature nào)
def make_model_name(row):
    p = Path(row["model_file"])
    stem = p.stem  # tên file không có .pkl
    parent = p.parent.name  # model / PCA / AutoEncoder

    base = f"{parent}/{stem}" if parent.lower() != "model" else stem
    feat = str(row.get("feature_used", ""))

    # nếu không phải Baseline34 thì thêm (feature_used) để phân biệt
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
          "accuracy": "accurancy",   # đúng theo cột bạn yêu cầu (dù chính tả chuẩn là accuracy)
      })
      .sort_values("accurancy", ascending=False)
      .reset_index(drop=True)
      .round(3)
)

print(summary)

# nếu muốn lưu ra csv
summary.to_csv("summary_metrics.csv", index=False)
print("Saved: summary_metrics.csv")
