import pandas as pd
import numpy as np
import warnings
import os
import utils
import extract_feature_PCA
try:
    import extract_feature_AE
except ImportError:
    print("Warning: extract_feature_AE (Torch) not found. Skipping AutoEncoder.")
    extract_feature_AE = None
from sklearn.model_selection import train_test_split
from nano_ensemble import run_experiment

# Suppress warnings
warnings.filterwarnings('ignore')

def train_group(group_name, specific_labels):
    print(f"\n{'='*20} Training {group_name} {'='*20}")
    print(f"Labels: {specific_labels}")
    
    # 1. Load Data for specific group
    print(f"[{group_name}] Loading Data...")
    # Using 40 features as per default in nano_ensemble.main
    X, y = utils.make_data(num_features=40, specific_labels=specific_labels)
    X = utils.Norm(X)
    
    print(f"[{group_name}] Data Shape: {X.shape}, Labels Shape: {y.shape}")
    
    # Check Stratified Split
    # y is (N, 1), ravel it
    y = y.ravel()
    
    if len(np.unique(y)) < 2:
        print(f"[{group_name}] Error: Not enough classes ({len(np.unique(y))}) to train.")
        return []

    X_train_orig, X_test_orig, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"[{group_name}] Train size: {X_train_orig.shape}, Test size: {X_test_orig.shape}")
    
    group_results = []
    
    # --- Experiment 1: Original ---
    # We pass group_name as prefix to feature_set_name to identify results later
    run_experiment(X_train_orig, y_train, X_test_orig, y_test, f"{group_name}_Original", group_results, specific_labels=specific_labels)
    
    # --- Experiment 2: PCA ---
    print(f"\n[{group_name}] Extracting PCA Features...")
    X_mean = np.mean(X_train_orig, axis=0)
    # Get projection matrix U using Training data
    # Standard n_components=15 from nano_ensemble
    try:
        U = extract_feature_PCA.U_for_pca(X_train_orig, X_mean, n_components=15)
        X_train_pca = extract_feature_PCA.pca(X_train_orig, X_mean, U)
        X_test_pca = extract_feature_PCA.pca(X_test_orig, X_mean, U)
        
        run_experiment(X_train_pca, y_train, X_test_pca, y_test, f"{group_name}_PCA", group_results, specific_labels=specific_labels)
    except Exception as e:
        print(f"[{group_name}] PCA Error: {e}")
    
    # --- Experiment 3: AutoEncoder ---
    if extract_feature_AE is not None:
        print(f"\n[{group_name}] Extracting AutoEncoder Features...")
        try:
            # Reduced epochs for speed, match nano_ensemble or slightly less? Using 50 as in nano_ensemble
            ae_model = extract_feature_AE.modelAE(X_train_orig, out_features=15, num_epochs=50)
            X_train_ae = extract_feature_AE.extractAE(ae_model, X_train_orig)
            X_test_ae = extract_feature_AE.extractAE(ae_model, X_test_orig)
            
            run_experiment(X_train_ae, y_train, X_test_ae, y_test, f"{group_name}_AutoEncoder", group_results, specific_labels=specific_labels)
        except Exception as e:
            print(f"[{group_name}] AutoEncoder Error: {e}")
    else:
        print(f"[{group_name}] AutoEncoder skipped (module not loaded).")

    return group_results

def main():
    # Define groups from utils
    groups = [
        ("Group1", utils.GROUP_1),
        ("Group2", utils.GROUP_2),
        ("Group3", utils.GROUP_3)
    ]
    
    all_groups_results = []
    
    for group_name, labels_list in groups:
        results = train_group(group_name, labels_list)
        all_groups_results.extend(results)
        
    # Save consolidated results
    if all_groups_results:
        results_df = pd.DataFrame(all_groups_results)
        results_df.to_csv("evaluation_results_groups.csv", index=False)
        print("\nAll group experiments completed. Results saved to evaluation_results_groups.csv")
        print(results_df)
    else:
        print("\nNo results generated.")

if __name__ == "__main__":
    main()
