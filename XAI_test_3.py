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

def generate_categorical_features(num_samples: int, probability_range: tuple[float, float], feature_names: list[str]) -> pd.DataFrame:
    """Generates categorical features (0, 1, 2) mimicking gene distributions."""
    num_features = len(feature_names)
    P = np.random.uniform(probability_range[0], probability_range[1], num_features)
    
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
    probability_range: tuple[float, float] = (0.01, 0.49),
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
    X_full = generate_categorical_features(num_samples, probability_range, full_feature_names)

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

def perform_traditional_chi_square(X_train: pd.DataFrame, y_train: np.ndarray, pvalue_threshold: float = 0.05) -> dict[str, float]:
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
    
    print(f"Significant Features (p < {pvalue_threshold}) [Bivariate Test]:")
    significant_found = False
    for feat, p in sorted_p_values:
        if p < pvalue_threshold:
            print(f"  - {feat}: p-value = {p:.4e}")
            significant_found = True
    if not significant_found:
        print(f"  None detected at p < {pvalue_threshold}")
        
    # Plot -log10(p-values)
    feats = [x[0] for x in sorted_p_values]
    # Add a tiny number to prevent log10(0) if a p-value is extremely small
    log_ps = [-np.log10(x[1] + 1e-300) for x in sorted_p_values] 
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=feats, y=log_ps, color='skyblue')
    plt.axhline(-np.log10(pvalue_threshold), color='red', linestyle='--', label=f'Significance Threshold (p={pvalue_threshold})')
    plt.title("Chi-Square Results: -log10(p-value) by Feature")
    plt.ylabel("-log10(p-value) [Higher is more significant]")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return sorted_p_values

def perform_traditional_logistic_regression(X_train: pd.DataFrame, y_train: np.ndarray, pvalue_threshold: float = 0.05) -> tuple[sm.Logit, list[str]]:
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
            print(f"  None detected at p < {pvalue_threshold}")
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
        return logit_model, significant_features
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

def plot_all_xgb_importances(model) -> tuple[dict, dict, dict]:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Fixed formatter to {v:.2f}
    weight_importance = model.get_booster().get_score(importance_type='weight')
    gain_importance = model.get_booster().get_score(importance_type='gain')
    cover_importance = model.get_booster().get_score(importance_type='cover')

    xgb.plot_importance(model, importance_type='weight', ax=axes[0], title="Importance: Weight (Frequency)", values_format="{v:.2f}")
    xgb.plot_importance(model, importance_type='gain', ax=axes[1], title="Importance: Gain (Accuracy)", values_format="{v:.2f}")
    xgb.plot_importance(model, importance_type='cover', ax=axes[2], title="Importance: Cover (Coverage)", values_format="{v:.2f}")
    plt.tight_layout()
    plt.show()

    return weight_importance, gain_importance, cover_importance

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

def analyze_shap_interactions(model: xgb.XGBRegressor, X_test: pd.DataFrame, feature_names: list[str]) -> dict[str, float]:
    """Computes True SHAP Interaction Values, plots the strongest pair, and returns a dictionary of global interaction importances."""
    print("\n--- Computing SHAP Interaction Values ---")
    explainer = shap.TreeExplainer(model)
    
    # THE FIX: Cast the dataframe to float to bypass the SHAP DMatrix categorical bug
    X_test_numeric = X_test.astype(float)
    
    # Output shape is (n_samples, n_features, n_features)
    # We pass the numeric version here!
    interaction_values = explainer.shap_interaction_values(X_test_numeric)
    
    # Global Interaction Summary
    shap.summary_plot(interaction_values, X_test_numeric, feature_names=feature_names, max_display=len(feature_names), show=False)
    plt.show()
    
    # Build the Global Interaction Dictionary
    mean_abs_interactions = np.abs(interaction_values).mean(axis=0)
    interaction_dict = {}
    n_features = len(feature_names)
    
    for i in range(n_features):
        for j in range(i + 1, n_features):
            total_interaction = mean_abs_interactions[i, j] * 2
            pair_name = f"{feature_names[i]} * {feature_names[j]}"
            interaction_dict[pair_name] = total_interaction
            
    # Sort from most important interaction to least important
    sorted_interactions = dict(sorted(interaction_dict.items(), key=lambda item: item[1], reverse=True))
    
    # Find the strongest interacting pair
    strongest_pair = list(sorted_interactions.keys())[0]
    feat_i, feat_j = strongest_pair.split(" * ")
    
    print(f"\nStrongest Interaction found by SHAP: {feat_i} & {feat_j}")
    print(f"Plotting pure interaction effect for {feat_i} and {feat_j}...")
    
    idx_i = feature_names.index(feat_i)
    idx_j = feature_names.index(feat_j)
    
    shap.dependence_plot(
        (idx_i, idx_j), 
        interaction_values, 
        X_test_numeric, 
        feature_names=feature_names,
        show=False
    )
    plt.show()

    # Print the ranked interaction dictionary
    print("\n--- SHAP Interaction Feature Importance Ranking ---")
    print("Top 10 Strongest Interactions:")
    for pair, imp in list(sorted_interactions.items())[:10]: 
        print(f"  {pair}: {imp:.6f}")
    print("-" * 49 + "\n")

    return sorted_interactions

def print_feature_importance(shap_values: shap.Explanation, X_test: pd.DataFrame) -> pd.DataFrame:
    """Prints a ranking based on SHAP value and returns distribution stats per feature option."""
    feature_names = shap_values.feature_names
    
    # List to collect the distribution statistics
    stats_list = []
    
    for i, feature in enumerate(feature_names):
        # Extract actual categorical data for this feature
        feature_data = X_test[feature].values
        # Extract SHAP values for this feature
        feature_shap = shap_values.values[:, i]
        
        # Calculate global mean absolute SHAP for this feature to maintain a ranking system
        global_mean_abs = np.abs(feature_shap).mean()
        
        # Iterate through unique options/genotypes present for this feature (e.g., 0, 1, 2)
        unique_options = np.unique(feature_data)
        for option in unique_options:
            # Mask for the current option
            mask = (feature_data == option)
            option_shap = feature_shap[mask]
            
            # Calculate descriptive stats using pandas
            stats = pd.Series(option_shap).describe()
            
            stats_list.append({
                'Feature': feature,
                'Global_Mean_Abs_SHAP': global_mean_abs,
                'Option': option,
                'Count': int(stats['count']),
                'Mean_SHAP': stats['mean'],
                'Std_SHAP': stats['std'] if not pd.isna(stats['std']) else 0.0,
                'Min_SHAP': stats['min'],
                '25%_SHAP': stats['25%'],
                'Median_SHAP': stats['50%'],
                '75%_SHAP': stats['75%'],
                'Max_SHAP': stats['max']
            })
            
    # Create the detailed DataFrame
    importance_df = pd.DataFrame(stats_list)
    
    # Sort first by the global importance of the feature, then by the specific option
    importance_df.sort_values(
        by=['Global_Mean_Abs_SHAP', 'Feature', 'Option'], 
        ascending=[False, True, True], 
        inplace=True
    )
    
    print("--- SHAP Feature Distribution Statistics by Genotype ---")
    # Print the top 15 rows to preview the top few features and their options
    print(importance_df.head(15).to_string(index=False))
    print("------------------------------------------------------\n")
    
    return importance_df


# ==========================================
# 4. p-like values for SHAP importances
# ==========================================

def compute_shap_pvalues(X_train: pd.DataFrame, y_train: np.ndarray, 
                         X_test: pd.DataFrame, true_shap_values: shap.Explanation, 
                         feature_names: list[str], n_permutations: int = 50) -> dict[str, float]:
    """Calculates empirical p-values for SHAP feature importances via permutation."""
    print(f"\n--- Calculating Empirical SHAP p-values (Permutations: {n_permutations}) ---")
    
    true_importances = np.abs(true_shap_values.values).mean(axis=0)
    null_importances = np.zeros((n_permutations, len(feature_names)))
    
    for i in range(n_permutations):
        y_train_shuffled = np.random.permutation(y_train)
        
        # Categorical-enabled CLASSIFIER null model
        null_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1, 
            enable_categorical=True, tree_method='hist', 
            objective='binary:logistic', n_jobs=-1, random_state=i
        )
        null_model.fit(X_train, y_train_shuffled)
        
        null_explainer = shap.TreeExplainer(null_model)
        null_shap_vals = null_explainer(X_test)
        null_importances[i, :] = np.abs(null_shap_vals.values).mean(axis=0)
        
    p_values = {}
    for idx, feat in enumerate(feature_names):
        count_extreme = np.sum(null_importances[:, idx] >= true_importances[idx])
        p_values[feat] = (count_extreme + 1) / (n_permutations + 1)
        
    sorted_pvalues = dict(sorted(p_values.items(), key=lambda item: item[1]))

    print("\nSHAP Empirical P-values (< 0.05 is statistically significant):")
    for feat, pval in sorted_pvalues.items():
        significance = "***" if pval < 0.01 else ("*" if pval < 0.05 else "")
        print(f"  - {feat}: {pval:.4f} {significance}")
    print("-" * 52 + "\n")
        
    return sorted_pvalues


def compute_shap_shadow_features(X_train: pd.DataFrame, y_train: np.ndarray, 
                                 X_test: pd.DataFrame, feature_names: list[str]) -> dict[str, bool]:
    """Uses shadow features to determine SHAP statistical significance."""
    print("\n--- Running SHAP Shadow Feature Analysis ---")
    
    X_train_shadow = X_train.copy()
    X_test_shadow = X_test.copy()
    
    shadow_names = [f"Shadow_{feat}" for feat in feature_names]
    X_train_shadow.columns = shadow_names
    X_test_shadow.columns = shadow_names
    
    for col in shadow_names:
        X_train_shadow[col] = np.random.permutation(X_train_shadow[col].values)
        X_test_shadow[col] = np.random.permutation(X_test_shadow[col].values)
    
    # Ensure shadows are categorical for XAI_test_3.py
    X_train_shadow = X_train_shadow.astype('category')
    X_test_shadow = X_test_shadow.astype('category')
        
    X_train_extended = pd.concat([X_train, X_train_shadow], axis=1)
    X_test_extended = pd.concat([X_test, X_test_shadow], axis=1)
    
    # Categorical-enabled CLASSIFIER shadow model
    shadow_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1, 
        enable_categorical=True, tree_method='hist', 
        objective='binary:logistic', n_jobs=-1, random_state=42
    )
    shadow_model.fit(X_train_extended, y_train)
    
    explainer = shap.TreeExplainer(shadow_model)
    shap_vals_extended = explainer(X_test_extended)
    
    importances = np.abs(shap_vals_extended.values).mean(axis=0)
    importance_dict = dict(zip(X_train_extended.columns, importances))
    
    max_shadow_importance = max([importance_dict[name] for name in shadow_names])
    print(f"Maximum Shadow Feature Importance (Noise Threshold): {max_shadow_importance:.4f}")
    
    results = {}
    print("\nShadow Feature Significance Results:")
    for feat in feature_names:
        feat_importance = importance_dict[feat]
        is_significant = feat_importance > max_shadow_importance
        results[feat] = is_significant
        status = "PASSED (Significant)" if is_significant else "FAILED (Noise)"
        print(f"  - {feat}: {feat_importance:.4f} -> {status}")
        
    print("-" * 46 + "\n")
    return results


def compute_shap_bootstrapping(X_train: pd.DataFrame, y_train: np.ndarray, 
                               X_test: pd.DataFrame, feature_names: list[str], 
                               n_bootstraps: int = 50) -> dict[str, tuple[float, float, float]]:
    """Calculates 95% Confidence Intervals for SHAP feature importances."""
    print(f"\n--- Running SHAP Bootstrapping Analysis (Iterations: {n_bootstraps}) ---")
    
    bootstrap_importances = np.zeros((n_bootstraps, len(feature_names)))
    
    for i in range(n_bootstraps):
        indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_train_boot = X_train.iloc[indices]
        y_train_boot = y_train[indices] if isinstance(y_train, np.ndarray) else y_train.iloc[indices]
        
        # Categorical-enabled CLASSIFIER bootstrap model
        boot_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1, 
            enable_categorical=True, tree_method='hist', 
            objective='binary:logistic', n_jobs=-1, random_state=i
        )
        boot_model.fit(X_train_boot, y_train_boot)
        
        boot_explainer = shap.TreeExplainer(boot_model)
        boot_shap_vals = boot_explainer(X_test)
        bootstrap_importances[i, :] = np.abs(boot_shap_vals.values).mean(axis=0)
        
    results = {}
    print("\nSHAP Bootstrapped 95% Confidence Intervals:")
    
    mean_importances = np.mean(bootstrap_importances, axis=0)
    sorted_indices = np.argsort(mean_importances)[::-1]
    
    for idx in sorted_indices:
        feat = feature_names[idx]
        mean_val = mean_importances[idx]
        lower_bound = np.percentile(bootstrap_importances[:, idx], 2.5)
        upper_bound = np.percentile(bootstrap_importances[:, idx], 97.5)
        
        results[feat] = (mean_val, lower_bound, upper_bound)
        print(f"  - {feat}: {mean_val:.4f} (95% CI: [{lower_bound:.4f}, {upper_bound:.4f}])")
        
    print("-" * 55 + "\n")
    return results

# ==========================================
# 5. MAIN PIPELINE
# ==========================================

def main():
    # 1. Generate Binary Data
    X, X_hidden, y_binary, visible_features, hidden_features = generate_synthetic_classification_data(
        num_inputs=10, 
        num_samples=5000, 
        probability_range=(0.01, 0.49),
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
    chi_square_features = perform_traditional_chi_square(X_train, y_train)

    # 2b. Traditional Baseline 2 (Logistic Regression)
    logistic_model, logistic_features = perform_traditional_logistic_regression(X_train, y_train)
    
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
    weight_importance, gain_importance, cover_importance = plot_all_xgb_importances(model)
    
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
    sorted_interactions = analyze_shap_interactions(model, X_test_cat, visible_features)
    
    # 9. Print numerical ranking
    importance_df = print_feature_importance(shap_values, X_test_cat)

    # ---------------------------------------------------------
    # NEW: SHAP STATISTICAL VALIDATION (Classification)
    # ---------------------------------------------------------
    
    # 10. SHAP Empirical P-values (Permutation Test)
    shap_pvalues = compute_shap_pvalues(
        X_train=X_train_cat, 
        y_train=y_train, 
        X_test=X_test_cat, 
        true_shap_values=shap_values, 
        feature_names=visible_features, 
        n_permutations=50
    )

    # 11. SHAP Shadow Feature Analysis (Boruta Method)
    shadow_results = compute_shap_shadow_features(
        X_train=X_train_cat, 
        y_train=y_train, 
        X_test=X_test_cat, 
        feature_names=visible_features
    )

    # 12. SHAP Bootstrapping (Confidence Intervals)
    bootstrap_results = compute_shap_bootstrapping(
        X_train=X_train_cat, 
        y_train=y_train, 
        X_test=X_test_cat, 
        feature_names=visible_features,
        n_bootstraps=30 
    )

if __name__ == "__main__":
    main()