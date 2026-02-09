from pathlib import Path
import numpy as np
import pandas as pd
import pickle, warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

from utils import *
from extract_feature_PCA import *
from extract_feature_AE import *

np.random.seed(42)

def load_pkl(path: Path):
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

def infer_n_features(model):
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    raise ValueError("Model không có n_features_in_. Hãy bọc pipeline khi save để khỏi mất thông tin.")

X, y = make_data(paths, 34)
X = Norm(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, stratify=y, shuffle=True, random_state=42
)
y_true = np.array(y_test).ravel()

X_mean = np.mean(X)

_pca_cache = {}
_ae_cache = {}

def get_pca(k: int):
    if k not in _pca_cache:
        Xtr = extract_PCA(X_train, X_mean, n_components=k)
        Xte = extract_PCA(X_test,  X_mean, n_components=k)
        _pca_cache[k] = (Xtr, Xte)
    return _pca_cache[k]

def get_ae(k: int):
    if k not in _ae_cache:
        ae_model = modelAE(X_train, out_features=k, num_epochs=20, learning_rate=0.01)
        Xtr = extractAE(ae_model, X_train)
        Xte = extractAE(ae_model, X_test)
        _ae_cache[k] = (Xtr, Xte)
    return _ae_cache[k]

def metrics_row(y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    row = {
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
    }
    if y_proba is not None:
        try:
            row["log_loss"] = log_loss(y_true, y_proba)
        except Exception as e:
            row["log_loss"] = f"ERR: {e}"
    return row

def try_predict(model, X_eval):
    y_pred = np.array(model.predict(X_eval)).ravel()
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_eval)
        except Exception:
            y_proba = None
    return y_pred, y_proba

os.makedirs("plots/confusion_matrix", exist_ok=True)

def evaluate_model(model_path: Path):
    model = load_pkl(model_path)
    k = infer_n_features(model)

    folder_parts = set(model_path.parts)
    candidates = []

    if "PCA" in folder_parts:
        _, Xte = get_pca(k)
        candidates = [("PCA", Xte)]
    elif "AutoEncoder" in folder_parts:
        _, Xte = get_ae(k)
        candidates = [("AutoEncoder", Xte)]
    else:
        if k == X_test.shape[1]:
            candidates = [("Baseline34", X_test)]
        else:
            if 1 <= k <= X_test.shape[1]:
                candidates.append((f"PCA_{k}", get_pca(k)[1]))
                candidates.append((f"AE_{k}", get_ae(k)[1]))
                candidates.append((f"Slice34_first{k}", X_test[:, :k]))

    tried = []
    best = None

    for name, Xte in candidates:
        try:
            y_pred, y_proba = try_predict(model, Xte)
            rowm = metrics_row(y_pred, y_proba)
            tried.append((name, rowm["accuracy"]))
            if (best is None) or (rowm["accuracy"] > best["accuracy"]):
                best = {"feature_used": name, "y_pred": y_pred, **rowm}
        except Exception as e:
            tried.append((name, f"ERR: {e}"))

    if best is None:
        raise ValueError(f"Không predict được với candidates={tried}")

    try:
        y_pred_best = best["y_pred"]
        cm = confusion_matrix(y_true, y_pred_best)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title(f"Confusion Matrix: {model_path.name} ({best['feature_used']})")
        
        save_name = f"{model_path.stem}_{best['feature_used']}.png"
        save_path = Path("plots/confusion_matrix") / save_name
        plt.savefig(save_path)
        plt.close(fig)
    except Exception as e:
        print(f"Failed to plot CM for {model_path}: {e}")

    res = {
        "model_file": str(model_path),
        "n_in": k,
        "feature_used": best["feature_used"],
        **{k: v for k, v in best.items() if k not in ["feature_used", "y_pred"]},
        "tried": str(tried),
    }
    return res

model_paths = sorted(Path("./model").rglob("*.pkl"))
rows, errs = [], []

for p in model_paths:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rows.append(evaluate_model(p))
    except Exception as e:
        errs.append({"model_file": str(p), "error": repr(e)})

df = pd.DataFrame(rows).sort_values(["accuracy"], ascending=False).reset_index(drop=True)
print(df)
df.to_csv("evaluation_results_ALL.csv", index=False)
print("Saved: evaluation_results_ALL.csv")

if errs:
    err_df = pd.DataFrame(errs)
    print(err_df)
    err_df.to_csv("evaluation_errors_ALL.csv", index=False)
    print("Saved: evaluation_errors_ALL.csv")
