import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

def generate_synthetic_data(num_inputs: int = 10, num_samples: int = 5000, 
                            num_contributing_features: tuple[int, int] = (2, 5), 
                            weight_range: tuple[float, float] = (-3, 3), 
                            noise_std: float = 0.05, 
                            ) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Generates synthetic data with a random linear combination of features."""
    m_inputs = np.random.randint(num_contributing_features[0], num_contributing_features[1] + 1)
    input_indices = np.random.choice(num_inputs, m_inputs, replace=False)
    weights = np.random.uniform(weight_range[0], weight_range[1], m_inputs)

    # Sort features by absolute weight (contribution)
    sorted_features = sorted(zip(input_indices, weights), key=lambda x: abs(x[1]), reverse=True)

    print("--- Generative Model Info ---")
    print(f"Number of contributing variables (M): {m_inputs}")
    for idx, w in sorted_features:
        print(f"Feature_{idx}: weight = {w:.4f}")
    print("-----------------------------\n")

    feature_names = [f"Feature_{i}" for i in range(num_inputs)]
    X = pd.DataFrame(np.random.rand(num_samples, num_inputs), columns=feature_names)

    y = np.zeros(num_samples)
    for idx, w in zip(input_indices, weights):
        y += w * X[f"Feature_{idx}"]

    # Add noise
    y += np.random.normal(0, noise_std, num_samples)
    
    return X, y, feature_names

def generate_synthetic_data_with_interactions(num_inputs: int = 10, num_samples: int = 5000, 
                            num_contributing_features: tuple[int, int] = (2, 5), 
                            weight_range: tuple[float, float] = (-3, 3), 
                            num_interactions: tuple[int, int] = (1, 2), 
                            interaction_weight_range: tuple[float, float] = (-3, 3), 
                            noise_std: float = 0.05, 
                            ) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Generates synthetic data with linear combinations AND 1-2 feature interactions."""
    # 1. Linear components
    m_inputs = np.random.randint(num_contributing_features[0], num_contributing_features[1] + 1)
    linear_indices = np.random.choice(num_inputs, m_inputs, replace=False)
    linear_weights = np.random.uniform(weight_range[0], weight_range[1], m_inputs)
    sorted_linear = sorted(zip(linear_indices, linear_weights), key=lambda x: abs(x[1]), reverse=True)

    # 2. Interaction components (1 to 2 interactions)
    num_interactions = np.random.randint(num_interactions[0], num_interactions[1] + 1)
    interactions = []
    for _ in range(num_interactions):
        # Pick 2 distinct features for the interaction pair
        pair = np.random.choice(num_inputs, 2, replace=False)
        weight = np.random.uniform(interaction_weight_range[0], interaction_weight_range[1], 1)[0]
        # Sort the pair indices just so "Feature_2 * Feature_1" prints as "Feature_1 * Feature_2"
        pair = sorted(pair) 
        interactions.append((pair[0], pair[1], weight))

    # Print Truth
    print("--- Generative Model Info (With Interactions) ---")
    print(f"Number of linear variables (M): {m_inputs}")
    for idx, w in sorted_linear:
        print(f"  Feature_{idx}: weight = {w:.4f}")
        
    print(f"\nNumber of interaction terms: {num_interactions}")
    # Sort interactions by absolute weight
    sorted_interactions = sorted(interactions, key=lambda x: abs(x[2]), reverse=True)
    for i, j, w in sorted_interactions:
        print(f"  Feature_{i} * Feature_{j}: weight = {w:.4f}")
    print("-------------------------------------------------\n")

    # Generate X
    feature_names = [f"Feature_{i}" for i in range(num_inputs)]
    X = pd.DataFrame(np.random.rand(num_samples, num_inputs), columns=feature_names)

    # 3. Calculate y
    y = np.zeros(num_samples)
    
    # Add linear terms
    for idx, w in zip(linear_indices, linear_weights):
        y += w * X[f"Feature_{idx}"]
        
    # Add interaction terms (w * Xi * Xj)
    for i, j, w in interactions:
        y += w * (X[f"Feature_{i}"] * X[f"Feature_{j}"])

    # Add noise
    y += np.random.normal(0, noise_std, num_samples)
    
    return X, y, feature_names

def perform_traditional_regression(X_train: pd.DataFrame, y_train: np.ndarray, 
                                   ) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Performs traditional OLS regression to find significant features and their coefficients."""
    print("\n--- Traditional Regression Analysis (OLS) ---")
    
    # Add a constant term for the intercept (statsmodels requires this explicitly)
    X_train_sm = sm.add_constant(X_train)
    
    # Fit the Ordinary Least Squares (OLS) model
    ols_model = sm.OLS(y_train, X_train_sm).fit()
    
    # Print the traditional statistical summary (this is what you'd see in R or SAS)
    print(ols_model.summary())
    
    # Extract coefficients and confidence intervals (excluding the intercept 'const')
    coefs = ols_model.params.drop('const')
    pvalues = ols_model.pvalues.drop('const')
    conf_int = ols_model.conf_int().drop('const')
    
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
    
    # Identify statistically significant features (p < 0.05)
    significant_features = pvalues[pvalues < 0.05].index.tolist()
    print(f"\nStatistically Significant Features (p < 0.05):")
    print(significant_features)
    
    return ols_model

def train_xgb_model(X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2, random_state: int = 4,
                    ) -> tuple[xgb.XGBRegressor, pd.DataFrame, pd.DataFrame]:
    """Splits data and trains a basic XGBoost Regressor."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
    model.fit(X_train, y_train)
    
    return model, X_train, X_test

def plot_all_xgb_importances(model: xgb.XGBRegressor, feature_names: list[str]) -> None:
    """Plots Weight, Gain, and Cover importance metrics side-by-side."""
    booster = model.get_booster()
    metrics = ['weight', 'gain', 'cover']
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
    fig.suptitle("XGBoost Built-in Feature Importances", fontsize=16)
    
    for i, metric in enumerate(metrics):
        score_dict = booster.get_score(importance_type=metric)
        full_scores = {feat: score_dict.get(feat, 0.0) for feat in feature_names}
        sorted_scores = dict(sorted(full_scores.items(), key=lambda item: item[1]))
        
        features = list(sorted_scores.keys())
        scores = list(sorted_scores.values())

        axes[i].barh(features, scores, color='#43B02A')
        axes[i].set_title(metric.capitalize())
        axes[i].set_xlabel(f"Average {metric.capitalize()} Score")
        
    plt.tight_layout()
    plt.show()

def compute_shap_values(model: xgb.XGBRegressor, X_train: pd.DataFrame, X_test: pd.DataFrame, max_background_samples: int = 100) -> shap.Explanation:
    """Computes SHAP values using the training set as a background distribution."""
    # masker = shap.maskers.Independent(X_train, max_samples=max_background_samples)
    # explainer = shap.Explainer(model.predict, masker)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    
    return shap_values

def plot_shap_analysis(shap_values: shap.Explanation, feature_names: list[str]) -> None:
    """Generates the beeswarm plot and detailed feature dependence plots."""
    # 1. Beeswarm Plot
    plt.figure(figsize=(10, 6))
    plt.title("SHAP Beeswarm Plot")
    shap.plots.beeswarm(shap_values)
    plt.show()

    # 2. Feature Dependence Plots
    num_features = len(feature_names)
    cols = 5
    rows = (num_features + cols - 1) // cols 

    fig1, axes1 = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 4 * rows))
    fig2, axes2 = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 4 * rows))
    fig3, axes3 = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 4 * rows))

    axes1, axes2, axes3 = axes1.flatten(), axes2.flatten(), axes3.flatten()

    for i, feature in enumerate(feature_names):
        x_vals = shap_values.data[:, i]
        y_vals = shap_values.values[:, i]
        
        y_vals_normalize = y_vals / (x_vals - x_vals.mean() + 1e-3) 
        
        sort_idx = np.argsort(x_vals)
        x_sorted, y_sorted = x_vals[sort_idx], y_vals[sort_idx]
        y_vals_normalize2 = (y_sorted[1:] - y_sorted[:-1]) / (x_sorted[1:] - x_sorted[:-1] + 1e-8) 

        axes1[i].scatter(x_vals, y_vals, alpha=0.5, s=15, color='#1E88E5')
        axes1[i].set(title=f"SHAP vs {feature}", xlabel="Feature", ylabel="SHAP")

        axes2[i].scatter(x_vals, y_vals_normalize, alpha=0.5, s=15, color='#1E88E5')
        axes2[i].set(title=f"SHAP/Feature vs {feature}", xlabel="Feature", ylabel="SHAP/Feature")

        axes3[i].scatter(x_sorted[:-1], y_vals_normalize2, alpha=0.5, s=15, color='#1E88E5')
        axes3[i].set(title=f"d(SHAP)/d(Feature) vs {feature}", xlabel="Feature", ylabel="Derivative")
        
        for ax in [axes1[i], axes2[i], axes3[i]]:
            ax.axhline(0, color='gray', linestyle='--', linewidth=1)

    for j in range(num_features, len(axes1)):
        fig1.delaxes(axes1[j])
        fig2.delaxes(axes2[j])
        fig3.delaxes(axes3[j])

    for fig in [fig1, fig2, fig3]:
        fig.tight_layout()
    plt.show()

def print_feature_importance(shap_values: shap.Explanation) -> list[str]:
    """Calculates and prints global feature importance based on mean absolute SHAP values."""
    global_importances = np.abs(shap_values.values).mean(0)
    feature_importance_dict = dict(zip(shap_values.feature_names, global_importances))

    sorted_importance = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))
    ranked_features = list(sorted_importance.keys())

    print("\nSHAP Global Importance Dictionary:")
    for feat, imp in sorted_importance.items():
        print(f"  {feat}: {imp:.4f}")

    print("\nRanked Features (Most to Least Important):")
    print(ranked_features)
    
    return ranked_features

def analyze_shap_interactions(model: xgb.XGBRegressor, X_test: pd.DataFrame, feature_names: list[str]) -> None:
    """Computes True SHAP Interaction Values and plots the strongest pair."""
    print("\n--- Computing SHAP Interaction Values ---")
    # TreeExplainer calculates interactions directly from the tree structure
    explainer = shap.TreeExplainer(model)
    
    # Output shape is (n_samples, n_features, n_features)
    interaction_values = explainer.shap_interaction_values(X_test)
    
    # 1. Global Interaction Summary
    # Pass show=False so we can attach a title before it renders
    shap.summary_plot(interaction_values, X_test, feature_names=feature_names, max_display=len(feature_names), show=False)
    # plt.title("SHAP Interaction Summary (Global View)")
    # plt.tight_layout()
    plt.show()
    
    # 2. Find the strongest interacting pair
    # Take mean absolute value across all samples
    mean_abs_interactions = np.abs(interaction_values).mean(axis=0)
    
    # Zero out the diagonal (which represents main effects, not interactions)
    np.fill_diagonal(mean_abs_interactions, 0)
    
    # Find indices of the maximum interaction
    idx_i, idx_j = np.unravel_index(np.argmax(mean_abs_interactions), mean_abs_interactions.shape)
    
    feat_i, feat_j = feature_names[idx_i], feature_names[idx_j]
    print(f"\nStrongest Interaction found by SHAP: {feat_i} & {feat_j}")
    
    # 3. Plot specific dependence for the strongest interaction
    print(f"Plotting pure interaction effect for {feat_i} and {feat_j}...")
    
    # Remove the invalid 'title=' argument and use show=False
    shap.dependence_plot(
        (idx_i, idx_j), 
        interaction_values, 
        X_test, 
        feature_names=feature_names,
        show=False
    )
    # Apply the title correctly via Matplotlib
    # plt.title(f"Pure SHAP Interaction Effect: {feat_i} * {feat_j}")
    # plt.tight_layout()
    plt.show()

def main():
    """Main execution pipeline."""
    # 1. Setup & Data
    # X, y, feature_names = generate_synthetic_data(
    #     num_inputs=10, 
    #     num_samples=5000, 
    #     num_contributing_features=(2, 5), 
    #     weight_range=(-3, 3),
    #     noise_std=0.05,
    # )
    X, y, feature_names = generate_synthetic_data_with_interactions(
        num_inputs=10, 
        num_samples=5000, 
        num_contributing_features=(2, 5), 
        weight_range=(-3, 3),
        num_interactions=(1, 2),
        interaction_weight_range=(-3, 3),
        noise_std=0.05,
    )
    
    # Split data here so both OLS and XGBoost train on the exact same subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Traditional Baseline (OLS)
    perform_traditional_regression(X_train, y_train)
    
    # 3. Machine Learning Modeling (XGBoost)
    # We pass the pre-split data directly to ensure apples-to-apples comparison
    model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
    model.fit(X_train, y_train)
    
    # 4. XGBoost Native Importances
    plot_all_xgb_importances(model, feature_names)
    
    # 5. Explainability (SHAP)
    shap_values = compute_shap_values(model, X_train, X_test)
    
    # 6. Visualization (SHAP)
    plot_shap_analysis(shap_values, feature_names)
    
    # 7. Deep Dive: SHAP Interaction Analysis
    # Note: X_test must be a DataFrame or numpy array. We use X_test directly.
    analyze_shap_interactions(model, X_test, feature_names)

    # 8. Summary (SHAP)
    ranked_features = print_feature_importance(shap_values)

if __name__ == "__main__":
    main()