import pandas as pd
import numpy as np
import warnings
import time
import os
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.exceptions import FitFailedWarning, DataConversionWarning

import utils

# Suppress warnings
warnings.filterwarnings('ignore')

def perform_grid_search(X, y, cv=2):
    """
    Performs grid search to find best hyperparameters for base models.
    Adapted from utils.Grid_search_model but ensures SVC has probability=True.
    """
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'SVM': SVC(probability=True, random_state=42), # Important for Soft Voting
        'KNN': KNeighborsClassifier(),
        'Random Forest': RandomForestClassifier(random_state=42),
    }

    # Parameters from utils.py (simplified or full)
    parameters = {
        'Logistic Regression': {'C': [0.1, 1.0, 10.0], 'solver': ['liblinear', 'lbfgs']},
        'SVM': {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']},
        'KNN': {'n_neighbors': [3, 4, 5, 7], 'weights': ['uniform', 'distance'], 'metric': ["euclidean", "manhattan", "correlation"]},
        'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
    }

    best_models = {}
    
    print(f"Grid Search Progress:")
    for name in models:
        if name not in parameters: continue
        
        print(f"  Tuning {name}...", end=" ", flush=True)
        start_time = time.time()
        model = models[name]
        params = parameters[name]
        
        clf = GridSearchCV(model, params, cv=cv, n_jobs=-1, scoring='accuracy')
        clf.fit(X, y)
        
        end_time = time.time()
        print(f"Done ({end_time - start_time:.2f}s). Best score: {clf.best_score_:.4f}")
        # print(f"    Params: {clf.best_params_}")
        best_models[name] = clf.best_estimator_
            
    return best_models

def main():
    # 1. Load Data
    print("Loading and processing data...")
    # Checking utils.make_data signature: def make_data(paths_data = paths, num_features=None):
    # It uses global 'paths' and 'file_names' in utils.py. 
    # We should assume utils is set up correctly or pass arguments if needed.
    # utils.paths and utils.file_names are defined at module level in utils.py.
    
    X, y = utils.make_data(num_features=34) 
    
    # Normalize
    X = utils.Norm(X)
    
    # Split
    # Split with stratify to ensure all classes are in test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    print(f"Dataset: Train={X_train.shape}, Test={X_test.shape}")
    
    # Save Test Set for inspection
    test_df = pd.DataFrame(X_test)
    test_df['label_index'] = y_test
    # Map label index to name if possible, utils.labels is global in utils but we need access
    # We can use utils.labels
    test_df['label_name'] = [utils.labels[int(i)] for i in y_test]
    test_df.to_csv("test_dataset.csv", index=False)
    print("Saved test dataset to test_dataset.csv")

    # 2. Grid Search
    print("\n--- Hyperparameter Tuning ---")
    best_models = perform_grid_search(X_train, y_train)

    # --- 3. Evaluate Individual Models ---
    print("\n--- Evaluating Individual Models ---")
    models_to_eval = []
    
    for name, model in best_models.items():
        print(f"\nEvaluating {name}...")
        utils.model_predict(X_test, y_test, model)
        models_to_eval.append((name, model))

    # --- 4. Ensemble (Voting) ---
    print("\n--- Building Ensembles ---")
    
    # Retrieve best models
    clf_log = best_models.get('Logistic Regression')
    clf_svm = best_models.get('SVM')
    clf_knn = best_models.get('KNN')
    clf_rf = best_models.get('Random Forest')
    
    if not all([clf_log, clf_svm, clf_knn, clf_rf]):
        print("Error: Could not find all base models from Grid Search.")
        return

    # Separate estimators for Boosting (No KNN) and Full Voting (With KNN)
    # KNN does not support sample_weight, which AdaBoost requires.
    
    voting_estimators_full = [
        ('logistic', clf_log),
        ('svm', clf_svm),
        ('knn', clf_knn),
        ('random', clf_rf)
    ]
    
    voting_estimators_boost = [
        ('logistic', clf_log),
        ('svm', clf_svm),
        ('random', clf_rf)
    ]
    
    # --- 5. Standalone Voting Classifier (Full) ---
    print("\n[VotingClassifier (Full) Results]")
    ensemble_clf_full = VotingClassifier(
        estimators=voting_estimators_full,
        voting='soft'
    )
    ensemble_clf_full.fit(X_train, y_train)
    utils.model_predict(X_test, y_test, ensemble_clf_full)
    models_to_eval.append(("VotingClassifier_Full", ensemble_clf_full))

    # --- 6. Boosting with Voting Base (No KNN) ---
    print("\n[AdaBoostClassifier (Base=Voting without KNN) Results]")
    ensemble_clf_boost = VotingClassifier(
        estimators=voting_estimators_boost,
        voting='soft'
    )
    # We don't necessarily need to pre-fit the base estimator for AdaBoost, 
    # but let's define it correctly.

    boosting_clf = AdaBoostClassifier(
        estimator=ensemble_clf_boost,
        n_estimators=10,
        learning_rate=1.0,
        random_state=42
    )
    
    model_start = time.time()
    boosting_clf.fit(X_train, y_train)
    print(f"Time taken to train AdaBoost: {time.time() - model_start:.2f}s")
    
    utils.model_predict(X_test, y_test, boosting_clf)
    models_to_eval.append(("AdaBoostClassifier", boosting_clf))
    
    # --- 7. Save Results ---
    
    # Note: Confusion Matrices are automatically plotted and saved to './plots' 
    # by utils.model_predict -> utils.evaluate_model
    
    # Calculate and Save Metrics to CSV
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    
    results_data = []
    
    for name, model in models_to_eval:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        
        results_data.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        })
        
    results_df = pd.DataFrame(results_data)
    results_df.to_csv("evaluation_results.csv", index=False)
    print("\nSaved evaluation results to evaluation_results.csv")
    print(results_df)

if __name__ == "__main__":
    main()
