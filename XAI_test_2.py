import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. BIOLOGICAL DATA GENERATION FUNCTIONS
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
    """Applies realistic biological main effects for genotypes (0, 1, 2)."""
    if mode == 'additive':
        return weight * x_series.values
    elif mode == 'dominant':
        return weight * np.where(x_series.values > 0, 1, 0)
    elif mode == 'recessive':
        return weight * np.where(x_series.values == 2, 1, 0)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def apply_epistatic_interaction(x1_series: pd.Series, x2_series: pd.Series, weight: float, mode: str) -> np.ndarray:
    """Applies biological epistatic (gene-gene) interactions."""
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

def generate_synthetic_data_with_hidden_features(
    num_inputs: int = 10, 
    num_samples: int = 5000, 
    num_contributing_features: tuple[int, int] = (2, 5), 
    num_hidden_features: tuple[int, int] = (1, 2), 
    weight_range: tuple[float, float] = (-3, 3), 
    num_interactions: tuple[int, int] = (1, 2), 
    interaction_weight_range: tuple[float, float] = (-3, 3), 
    noise_std: float = 0.05, 
    hidden_in_linear: bool = True, 
    hidden_in_interactions: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str], list[str]]:
    """Generates synthetic data where hidden features impact 'y' using biological logic."""
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

    # Print Generative Equation
    equation_parts = [f"({w:.2f} * {get_feat_name(idx)} [{mode}])" for idx, w, mode in linear_components]
    equation_parts += [f"({w:.2f} * ({get_feat_name(i)}, {get_feat_name(j)}) [{mode}])" for i, j, w, mode in sorted_interactions]
    print("--- Generative Model Info (With HIDDEN Features) ---")
    print("y = \n  " + " + \n  ".join(equation_parts) + f" \n  + Noise(0, {noise_std})\n")

    full_feature_names = [get_feat_name(i) for i in range(total_features)]
    X_full = generate_categorical_features(num_samples, full_feature_names)

    y = np.zeros(num_samples)
    for idx, w, mode in linear_components:
        y += apply_genetic_main_effect(X_full[get_feat_name(idx)], w, mode)
    for i, j, w, mode in interactions:
        y += apply_epistatic_interaction(X_full[get_feat_name(i)], X_full[get_feat_name(j)], w, mode)
    y += np.random.normal(0, noise_std, num_samples)
    
    visible_feature_names = [f"Feature_{i}" for i in range(num_inputs)]
    hidden_feature_names = [f"Hidden_{i}" for i in range(n_hidden)]
    
    return X_full[visible_feature_names].copy(), X_full[hidden_feature_names].copy(), y, visible_feature_names, hidden_feature_names

# ==========================================
# 2. MODELING & EVALUATION FUNCTIONS
# ==========================================

def perform_traditional_regression(X_train: pd.DataFrame, y_train: np.ndarray) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Performs traditional OLS regression to find significant features and their coefficients."""
    print("\n--- Traditional Regression Analysis (OLS) ---")
    
    # Add a constant term for the intercept (statsmodels requires this explicitly)
    # Cast to float to ensure statsmodels compatibility with integer/categorical arrays
    X_train_sm = sm.add_constant(X_train.astype(float))
    
    # Fit the Ordinary Least Squares (OLS) model
    ols_model = sm.OLS(y_train, X_train_sm).fit()
    
    # Print the traditional statistical summary
    print(ols_model.summary())
    
    # Extract coefficients and confidence intervals (excluding the intercept 'const')
    coefs = ols_model.params.drop('const')
    pvalues = ols_model.pvalues.drop('const')
    conf_int = ols_model.conf_int().drop('const')
    
    # Identify statistically significant features (p < 0.05)
    significant_features = pvalues[pvalues < 0.05].index.tolist()
    print("\n" + "="*50)
    print(f"Statistically Significant Features (p < 0.05):")
    if significant_features:
        for feat in significant_features:
            print(f"  - {feat} (p-value: {pvalues[feat]:.4e})")
    else:
        print("  None detected at p < 0.05")
    print("="*50 + "\n")
    
    # Calculate error margins for the plot
    lower_errors = coefs - conf_int[0]
    upper_errors = conf_int[1] - coefs
    
    # Plotting Coefficients with 95% Confidence Intervals
    plt.figure(figsize=(10, 6))
    plt.errorbar(coefs.index, coefs, yerr=[lower_errors, upper_errors], 
                 fmt='o', color='#D81B60', ecolor='lightgray', elinewidth=3, capsize=5)
    
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title("Traditional OLS Regression Coefficients (95% CI)")
    plt.xlabel("Features")
    plt.ylabel("Coefficient Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return ols_model

def evaluate_xgb_model(model, X_test: pd.DataFrame, y_test: np.ndarray):
    """Evaluates the XGBoost model."""
    preds = model.predict(X_test)
    print(f"--- XGBoost Performance (Testing) ---")
    print(f"R2 Score: {r2_score(y_test, preds):.4f}")
    print(f"MSE: {mean_squared_error(y_test, preds):.4f}\n")

# ==========================================
# 3. PLOTTING & SHAP EXPLAINABILITY FUNCTIONS
# ==========================================

def plot_all_xgb_importances(model):
    """Plots Weight, Gain, and Cover XGBoost feature importances."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Fixed formatter to {v:.2f}
    xgb.plot_importance(model, importance_type='weight', ax=axes[0], title="Importance: Weight (Frequency)", values_format="{v:.2f}")
    xgb.plot_importance(model, importance_type='gain', ax=axes[1], title="Importance: Gain (Accuracy)", values_format="{v:.2f}")
    xgb.plot_importance(model, importance_type='cover', ax=axes[2], title="Importance: Cover (Coverage)", values_format="{v:.2f}")
    
    plt.tight_layout()
    plt.show()

def print_feature_importance(shap_values):
    """Prints a ranking based on mean absolute SHAP value."""
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature': shap_values.feature_names,
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values(by='Mean_Abs_SHAP', ascending=False)
    
    print("--- SHAP Feature Importance Ranking ---")
    print(importance_df.head(10).to_string(index=False))
    print("---------------------------------------\n")
    return importance_df

def plot_shap_summary_violin(shap_values, X_test: pd.DataFrame):
    """Generates the main SHAP summary plot as a violin plot."""
    plt.title("Global SHAP Summary (Violin)")
    shap.summary_plot(shap_values.values, X_test.astype(float), plot_type="violin", show=False)
    plt.show()

def plot_all_individual_shap_violins(shap_values, X_test: pd.DataFrame, feature_names: list[str]):
    """Plots violin plots for all features' SHAP values in a single grid figure with a shared y-axis."""
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
        
        plot_df = pd.DataFrame({
            'Genotype': feature_actual_vals,
            'SHAP': feature_shap_vals
        })
        
        sns.violinplot(
            x='Genotype', 
            y='SHAP', 
            data=plot_df, 
            hue='Genotype',
            palette="muted",
            inner="quartile",
            legend=False,
            ax=axes_flat[i]
        )
        
        axes_flat[i].set_title(f"{feature_name}")
        axes_flat[i].axhline(0, color='black', linestyle='--', alpha=0.5)
        
        if i % cols == 0:
            axes_flat[i].set_ylabel('SHAP Value (Impact)')
        else:
            axes_flat[i].set_ylabel('')
            
        axes_flat[i].set_xlabel('Genotype')
    
    for j in range(num_features, len(axes_flat)):
        fig.delaxes(axes_flat[j])
        
    plt.suptitle("Individual SHAP Effect Distributions by Genotype (Shared Y-Axis)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.90) 
    plt.show()

def analyze_shap_interactions(model, X_test: pd.DataFrame):
    """Calculates SHAP interaction values, plots the summary, and isolates the strongest interaction."""
    print("\n--- Computing SHAP Interaction Values (this may take a moment) ---")
    explainer = shap.TreeExplainer(model)
    
    # FIX: Cast to float to bypass SHAP's internal categorical bug
    X_test_numeric = X_test.astype(float)
    feature_names = X_test_numeric.columns.tolist()
    
    # Output shape is (n_samples, n_features, n_features)
    shap_interaction_values = explainer.shap_interaction_values(X_test_numeric)
    
    # 1. Global Interaction Summary
    # plt.title("SHAP Top Interactions Summary")
    shap.summary_plot(shap_interaction_values, X_test_numeric, max_display=20, show=False)
    plt.show()
    
    # 2. Find the strongest interacting pair
    # Take mean absolute value across all samples
    mean_abs_interactions = np.abs(shap_interaction_values).mean(axis=0)
    
    # Zero out the diagonal (which represents main effects, not interactions)
    np.fill_diagonal(mean_abs_interactions, 0)
    
    # Find indices of the maximum interaction
    idx_i, idx_j = np.unravel_index(np.argmax(mean_abs_interactions), mean_abs_interactions.shape)
    
    feat_i, feat_j = feature_names[idx_i], feature_names[idx_j]
    print(f"\nStrongest Interaction found by SHAP: {feat_i} & {feat_j}")
    
    # 3. Plot specific dependence for the strongest interaction
    print(f"Plotting pure interaction effect for {feat_i} and {feat_j}...")
    
    shap.dependence_plot(
        (idx_i, idx_j), 
        shap_interaction_values, 
        X_test_numeric, 
        feature_names=feature_names,
        show=False
    )
    plt.title(f"Pure SHAP Interaction Effect: {feat_i} * {feat_j}")
    plt.tight_layout()
    plt.show()

def print_feature_importance(shap_values: shap.Explanation) -> list[str]:
    """Calculates and prints global feature importance based on mean absolute SHAP values."""
    global_importances = np.abs(shap_values.values).mean(0)
    feature_importance_dict = dict(zip(shap_values.feature_names, global_importances))

    sorted_importance = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))
    ranked_features = list(sorted_importance.keys())

    print("\n--- SHAP Feature Importance Ranking ---")
    print("SHAP Global Importance Dictionary:")
    for feat, imp in sorted_importance.items():
        print(f"  {feat}: {imp:.4f}")

    print("\nRanked Features (Most to Least Important):")
    print(ranked_features)
    print("-" * 39 + "\n")
    
    return ranked_features

# ==========================================
# 4. MAIN PIPELINE
# ==========================================

def main():
    # 1. Generate Data
    X, X_hidden, y, visible_features, hidden_features = generate_synthetic_data_with_hidden_features(
        num_inputs=10, 
        num_samples=5000, 
        num_contributing_features=(3, 5), 
        num_hidden_features=(1, 2), 
        weight_range=(-3, 3), 
        num_interactions=(2, 3), 
        interaction_weight_range=(-3, 3), 
        noise_std=0.1, 
        hidden_in_linear=True,           
        hidden_in_interactions=True
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Traditional Baseline (OLS)
    ols_model = perform_traditional_regression(X_train, y_train)
    
    # 3. Categorical Conversion for XGBoost
    X_train_cat = X_train.astype('category')
    X_test_cat = X_test.astype('category')

    # 4. Machine Learning Modeling (XGBoost)
    model = xgb.XGBRegressor(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.1,
        enable_categorical=True,
        tree_method='hist',
        random_state=42
    )
    model.fit(X_train_cat, y_train)
    
    # 5. Evaluation & XGBoost Importances
    evaluate_xgb_model(model, X_test_cat, y_test)
    plot_all_xgb_importances(model)
    
    # 6. SHAP Explainability 
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test_cat)
    
    # Global SHAP Violin Summary
    plot_shap_summary_violin(shap_values, X_test_cat)
    
    # 7. Individual Feature Violin Plots (All in one grid)
    print("Generating SHAP violin grid for all features...")
    plot_all_individual_shap_violins(shap_values, X_test_cat, visible_features)
    
    # 8. Deep Dive: Interactions
    analyze_shap_interactions(model, X_test_cat)
    
    # Print numerical ranking and save the list!
    ranked_features = print_feature_importance(shap_values)

if __name__ == "__main__":
    main()