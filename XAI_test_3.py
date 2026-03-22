import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import math
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

# ==========================================
# 1. BIOLOGICAL DATA GENERATION (CLASSIFICATION)
# ==========================================

def generate_categorical_features(num_samples: int, feature_names: list[str]) -> pd.DataFrame:
    """Generates categorical features (0, 1, 2) mimicking gene distributions."""
    num_features = len(feature_names)
    P = np.random.uniform(0.01, 0.49, num_features)
    
    data = np.zeros((num_samples, num_features))
    for i in range(num_features):
        p1 = P[i]
        p2 = P[i]**2
        p0 = 1.0 - p1 - p2
        data[:, i] = np.random.choice([0, 1, 2], size=num_samples, p=[p0, p1, p2])
        
    return pd.DataFrame(data, columns=feature_names)

def apply_genetic_main_effect(x_series: pd.Series, weight: float, mode: str) -> np.ndarray:
    if mode == 'additive':
        return weight * x_series.values
    elif mode == 'dominant':
        return weight * np.where(x_series.values > 0, 1, 0)
    elif mode == 'recessive':
        return weight * np.where(x_series.values == 2, 1, 0)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def apply_epistatic_interaction(x1_series: pd.Series, x2_series: pd.Series, weight: float, mode: str) -> np.ndarray:
    x1, x2 = x1_series.values, x2_series.values
    if mode == 'multiplicative':
        return weight * (x1 * x2)
    elif mode == 'dominant_epistasis': 
        return weight * np.where((x1 > 0) & (x2 > 0), 1, 0)
    elif mode == 'recessive_epistasis':
        return weight * np.where((x1 == 2) & (x2 == 2), 1, 0)
    elif mode == 'xor_interference':
        return weight * np.where((x1 > 0) ^ (x2 > 0), 1, 0)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def generate_synthetic_classification_data(
    num_inputs: int = 10, 
    num_samples: int = 5000, 
    num_contributing_features: tuple[int, int] = (3, 5), 
    num_hidden_features: tuple[int, int] = (1, 2), 
    weight_range: tuple[float, float] = (-1.5, 1.5), 
    num_interactions: tuple[int, int] = (1, 2), 
    interaction_weight_range: tuple[float, float] = (-1.5, 1.5), 
    noise_std: float = 0.5, 
    hidden_in_linear: bool = True, 
    hidden_in_interactions: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str], list[str]]:
    """Generates categorical features with a BINARY categorical output."""
    m_inputs = np.random.randint(num_contributing_features[0], num_contributing_features[1] + 1)
    n_hidden = np.random.randint(num_hidden_features[0], num_hidden_features[1] + 1)
    total_features = num_inputs + n_hidden
    
    def get_feat_name(idx):
        return f"Feature_{idx}" if idx < num_inputs else f"Hidden_{idx - num_inputs}"

    visible_indices = np.random.choice(num_inputs, m_inputs, replace=False)
    hidden_indices = np.arange(num_inputs, total_features) 
    
    all_contributing_indices = np.concatenate([visible_indices, hidden_indices]) if hidden_in_linear else visible_indices
        
    linear_weights = np.random.uniform(weight_range[0], weight_range[1], len(all_contributing_indices))
    linear_modes = np.random.choice(['additive', 'dominant', 'recessive'], size=len(all_contributing_indices))
    linear_components = sorted(list(zip(all_contributing_indices, linear_weights, linear_modes)), key=lambda x: abs(x[1]), reverse=True)

    n_interactions = np.random.randint(num_interactions[0], num_interactions[1] + 1)
    interact_modes = np.random.choice(['multiplicative', 'dominant_epistasis', 'recessive_epistasis', 'xor_interference'], size=n_interactions)
    
    interactions = []
    for i, mode in enumerate(interact_modes):
        if hidden_in_interactions and i == 0 and n_hidden > 0:
            h_feat = np.random.choice(hidden_indices)
            other_feat = np.random.choice([x for x in range(total_features) if x != h_feat])
            pair = sorted([h_feat, other_feat])
        elif hidden_in_interactions:
            pair = sorted(np.random.choice(total_features, 2, replace=False))
        else:
            pair = sorted(np.random.choice(num_inputs, 2, replace=False))
            
        weight = np.random.uniform(interaction_weight_range[0], interaction_weight_range[1])
        interactions.append((pair[0], pair[1], weight, mode))

    sorted_interactions = sorted(interactions, key=lambda x: abs(x[2]), reverse=True)

    print("--- Generative Model Info (Log-Odds Formulation) ---")
    equation_parts = [f"({w:.2f} * {get_feat_name(idx)} [{mode}])" for idx, w, mode in linear_components]
    equation_parts += [f"({w:.2f} * ({get_feat_name(i)}, {get_feat_name(j)}) [{mode}])" for i, j, w, mode in sorted_interactions]
    print("Log-Odds (z) = \n  " + " + \n  ".join(equation_parts) + f" \n  + Noise(0, {noise_std})")
    print("Probability (P) = 1 / (1 + e^-z)")
    print("Class Output = Binomial(1, P)\n")

    full_feature_names = [get_feat_name(i) for i in range(total_features)]
    X_full = generate_categorical_features(num_samples, full_feature_names)

    z = np.zeros(num_samples)
    for idx, w, mode in linear_components:
        z += apply_genetic_main_effect(X_full[get_feat_name(idx)], w, mode)
    for i, j, w, mode in interactions:
        z += apply_epistatic_interaction(X_full[get_feat_name(i)], X_full[get_feat_name(j)], w, mode)
    
    z += np.random.normal(0, noise_std, num_samples)
    
    probabilities = 1 / (1 + np.exp(-z))
    y_binary = np.random.binomial(1, probabilities)
    
    visible_feature_names = [f"Feature_{i}" for i in range(num_inputs)]
    hidden_feature_names = [f"Hidden_{i}" for i in range(n_hidden)]
    
    return X_full[visible_feature_names].copy(), X_full[hidden_feature_names].copy(), y_binary, visible_feature_names, hidden_feature_names

# ==========================================
# 2. MODELING & EVALUATION FUNCTIONS
# ==========================================

def perform_traditional_chi_square(X_train: pd.DataFrame, y_train: np.ndarray):
    """Performs Standard Bivariate Pearson Chi-Square Tests."""
    print("\n--- Traditional Baseline 1: Chi-Square Independence Tests ---")
    p_values = {}
    
    for feature in X_train.columns:
        # Create a contingency table (cross-tabulation)
        contingency_table = pd.crosstab(X_train[feature], y_train)
        
        # Run Chi-Square test
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        p_values[feature] = p
        
    # Sort features by significance (lowest p-value first)
    sorted_p_values = sorted(p_values.items(), key=lambda x: x[1])
    
    print("Significant Features (p < 0.05) [Bivariate Test]:")
    significant_found = False
    for feat, p in sorted_p_values:
        if p < 0.05:
            print(f"  - {feat}: p-value = {p:.4e}")
            significant_found = True
    if not significant_found:
        print("  None detected at p < 0.05")
        
    # Plot -log10(p-values)
    feats = [x[0] for x in sorted_p_values]
    # Add a tiny number to prevent log10(0) if a p-value is extremely small
    log_ps = [-np.log10(x[1] + 1e-300) for x in sorted_p_values] 
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=feats, y=log_ps, color='skyblue')
    plt.axhline(-np.log10(0.05), color='red', linestyle='--', label='Significance Threshold (p=0.05)')
    plt.title("Chi-Square Results: -log10(p-value) by Feature")
    plt.ylabel("-log10(p-value) [Higher is more significant]")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def perform_traditional_logistic_regression(X_train: pd.DataFrame, y_train: np.ndarray):
    """Performs traditional Multivariate Logistic Regression (statsmodels)."""
    print("\n--- Traditional Baseline 2: Multivariate Logistic Regression ---")
    X_train_sm = sm.add_constant(X_train.astype(float))
    
    try:
        logit_model = sm.Logit(y_train, X_train_sm).fit(disp=False)
        print(logit_model.summary())
        
        coefs = logit_model.params.drop('const')
        pvalues = logit_model.pvalues.drop('const')
        conf_int = logit_model.conf_int().drop('const')
        
        significant_features = pvalues[pvalues < 0.05].index.tolist()
        print("\n" + "="*50)
        print(f"Statistically Significant Features (p < 0.05) [Multivariate Test]:")
        if significant_features:
            for feat in significant_features:
                print(f"  - {feat} (p-value: {pvalues[feat]:.4e})")
        else:
            print("  None detected at p < 0.05")
        print("="*50 + "\n")
        
        lower_errors = coefs - conf_int[0]
        upper_errors = conf_int[1] - coefs
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(coefs.index, coefs, yerr=[lower_errors, upper_errors], 
                     fmt='o', color='#D81B60', ecolor='lightgray', elinewidth=3, capsize=5)
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.title("Logistic Regression Coefficients (Log-Odds Impact with 95% CI)")
        plt.xlabel("Features")
        plt.ylabel("Coefficient Value (Log-Odds)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        return logit_model
    except Exception as e:
        print(f"Logistic Regression failed (often due to perfect separation): {e}")
        return None

def evaluate_xgb_classifier(model, X_test: pd.DataFrame, y_test: np.ndarray):
    """Evaluates the XGBoost Classifier."""
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    print(f"\n--- XGBoost Classifier Performance (Testing) ---")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"ROC AUC:  {roc_auc_score(y_test, probs):.4f}")
    print(f"Log Loss: {log_loss(y_test, probs):.4f}\n")

# ==========================================
# 3. PLOTTING & SHAP EXPLAINABILITY FUNCTIONS
# ==========================================

def plot_all_xgb_importances(model):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    xgb.plot_importance(model, importance_type='weight', ax=axes[0], title="Importance: Weight (Frequency)", values_format="{v:.2f}")
    xgb.plot_importance(model, importance_type='gain', ax=axes[1], title="Importance: Gain (Accuracy)", values_format="{v:.2f}")
    xgb.plot_importance(model, importance_type='cover', ax=axes[2], title="Importance: Cover (Coverage)", values_format="{v:.2f}")
    plt.tight_layout()
    plt.show()

def plot_all_individual_shap_violins(shap_values, X_test: pd.DataFrame, feature_names: list[str]):
    num_features = len(feature_names)
    cols = 4 
    rows = math.ceil(num_features / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5), sharey=True)
    if rows == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()
    
    for i, feature_name in enumerate(feature_names):
        feature_idx = X_test.columns.get_loc(feature_name)
        feature_shap_vals = shap_values.values[:, feature_idx]
        feature_actual_vals = X_test[feature_name].values
        
        plot_df = pd.DataFrame({'Genotype': feature_actual_vals, 'SHAP': feature_shap_vals})
        
        sns.violinplot(
            x='Genotype', y='SHAP', data=plot_df, hue='Genotype',
            palette="muted", inner="quartile", legend=False, ax=axes_flat[i]
        )
        axes_flat[i].set_title(f"{feature_name}")
        axes_flat[i].axhline(0, color='black', linestyle='--', alpha=0.5)
        
        if i % cols == 0:
            axes_flat[i].set_ylabel('SHAP Value (Log-Odds)')
        else:
            axes_flat[i].set_ylabel('')
        axes_flat[i].set_xlabel('Genotype')
    
    for j in range(num_features, len(axes_flat)):
        fig.delaxes(axes_flat[j])
        
    plt.suptitle("Individual SHAP Effect Distributions by Genotype (Log-Odds Scale)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.90) 
    plt.show()

def analyze_shap_interactions(model, X_test: pd.DataFrame):
    print("\n--- Computing SHAP Interaction Values ---")
    explainer = shap.TreeExplainer(model)
    X_test_numeric = X_test.astype(float)
    feature_names = X_test_numeric.columns.tolist()
    
    shap_interaction_values = explainer.shap_interaction_values(X_test_numeric)
    
    shap.summary_plot(shap_interaction_values, X_test_numeric, max_display=20, show=False)
    plt.show()
    
    mean_abs_interactions = np.abs(shap_interaction_values).mean(axis=0)
    np.fill_diagonal(mean_abs_interactions, 0)
    
    idx_i, idx_j = np.unravel_index(np.argmax(mean_abs_interactions), mean_abs_interactions.shape)
    feat_i, feat_j = feature_names[idx_i], feature_names[idx_j]
    
    print(f"\nStrongest Interaction found by SHAP: {feat_i} & {feat_j}")
    
    shap.dependence_plot(
        (idx_i, idx_j), shap_interaction_values, X_test_numeric, feature_names=feature_names, show=False
    )
    plt.title(f"Pure SHAP Interaction Effect: {feat_i} * {feat_j} (Log-Odds)")
    plt.tight_layout()
    plt.show()

def print_feature_importance(shap_values: shap.Explanation) -> list[str]:
    global_importances = np.abs(shap_values.values).mean(0)
    feature_importance_dict = dict(zip(shap_values.feature_names, global_importances))
    sorted_importance = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))
    
    print("\n--- SHAP Feature Importance Ranking (Impact on Log-Odds) ---")
    for feat, imp in sorted_importance.items():
        print(f"  {feat}: {imp:.4f}")
    return list(sorted_importance.keys())

# ==========================================
# 4. MAIN PIPELINE
# ==========================================

def main():
    # 1. Generate Binary Data
    X, X_hidden, y_binary, visible_features, hidden_features = generate_synthetic_classification_data(
        num_inputs=10, 
        num_samples=5000, 
        num_contributing_features=(3, 5), 
        num_hidden_features=(1, 2), 
        weight_range=(-1.5, 1.5), 
        num_interactions=(2, 3), 
        interaction_weight_range=(-1.5, 1.5), 
        noise_std=0.5, 
        hidden_in_linear=True,           
        hidden_in_interactions=True
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)
    
    # 2a. Traditional Baseline 1 (Chi-Square Test)
    perform_traditional_chi_square(X_train, y_train)

    # 2b. Traditional Baseline 2 (Logistic Regression)
    perform_traditional_logistic_regression(X_train, y_train)
    
    # 3. Categorical Conversion for XGBoost
    X_train_cat = X_train.astype('category')
    X_test_cat = X_test.astype('category')

    # 4. Machine Learning Modeling (XGBoost Classifier)
    model = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.1,
        enable_categorical=True,
        tree_method='hist',
        objective='binary:logistic',
        random_state=42
    )
    model.fit(X_train_cat, y_train)
    
    # 5. Evaluation & XGBoost Importances
    evaluate_xgb_classifier(model, X_test_cat, y_test)
    plot_all_xgb_importances(model)
    
    # 6. SHAP Explainability 
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test_cat)
    
    # Global SHAP Violin Summary
    plt.title("Global SHAP Summary (Violin - Log-Odds Scale)")
    shap.summary_plot(shap_values.values, X_test_cat.astype(float), plot_type="violin", show=False)
    plt.show()
    
    # 7. Individual Feature Violin Plots 
    print("Generating SHAP violin grid for all features...")
    plot_all_individual_shap_violins(shap_values, X_test_cat, visible_features)
    
    # 8. Deep Dive: Interactions
    analyze_shap_interactions(model, X_test_cat)
    
    # Print numerical ranking
    print_feature_importance(shap_values)

if __name__ == "__main__":
    main()