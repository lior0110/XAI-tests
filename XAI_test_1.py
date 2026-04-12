import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================
# 1. SYNTHETIC DATA GENERATION FUNCTIONS
# ==========================================

def generate_linear_synthetic_data(num_inputs: int = 10, num_samples: int = 5000, 
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
    
    # Build Equation String
    equation_parts = []
    for idx, w in sorted_features:
        equation_parts.append(f"({w:.4f} * Feature_{idx})")
    equation_str = "y = " + " + ".join(equation_parts) + f" + Noise(0, {noise_std})"

    print("--- Generative Model Info ---")
    print(f"Number of contributing variables (M): {m_inputs}")
    print("\n--- True Mathematical Equation ---")
    print(equation_str)
    print("-" * 39 + "\n")

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

    sorted_interactions = sorted(interactions, key=lambda x: abs(x[2]), reverse=True)

    # 3. Build Equation String
    equation_parts = []
    for idx, w in sorted_linear:
        equation_parts.append(f"({w:.4f} * Feature_{idx})")
    for i, j, w in sorted_interactions:
        equation_parts.append(f"({w:.4f} * Feature_{i} * Feature_{j})")
        
    equation_str = "y = " + " + ".join(equation_parts) + f" + Noise(0, {noise_std})"

    # Print Truth
    print("--- Generative Model Info (With Interactions) ---")
    print(f"Number of linear variables (M): {m_inputs}")
    print(f"Number of interaction terms: {num_interactions}")
    print("\n--- True Mathematical Equation ---")
    print(equation_str)
    print("-" * 49 + "\n")

    # Generate X
    feature_names = [f"Feature_{i}" for i in range(num_inputs)]
    X = pd.DataFrame(np.random.rand(num_samples, num_inputs), columns=feature_names)

    # 4. Calculate y
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


def generate_synthetic_data_with_hidden_features(
    num_inputs: int = 10, 
    num_samples: int = 5000, 
    num_contributing_features: tuple[int, int] = (2, 5), 
    num_hidden_features: tuple[int, int] = (1, 2), 
    weight_range: tuple[float, float] = (-3, 3), 
    num_interactions: tuple[int, int] = (1, 2), 
    interaction_weight_range: tuple[float, float] = (-3, 3), 
    noise_std: float = 0.05, 
    hidden_in_linear: bool = True,           # NEW: Should hidden features have a linear impact?
    hidden_in_interactions: bool = True      # NEW: Should hidden features be part of interactions?
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str], list[str]]:
    """Generates synthetic data where hidden features impact 'y' based on user demand."""
    
    if not hidden_in_linear and not hidden_in_interactions:
        print("WARNING: Hidden features are excluded from both linear and interaction parts. They will not impact 'y'.")

    # 1. Determine counts
    m_inputs = np.random.randint(num_contributing_features[0], num_contributing_features[1] + 1)
    n_hidden = np.random.randint(num_hidden_features[0], num_hidden_features[1] + 1)
    total_features = num_inputs + n_hidden
    
    def get_feat_name(idx):
        return f"Feature_{idx}" if idx < num_inputs else f"Hidden_{idx - num_inputs}"

    # 2. Linear components
    visible_indices = np.random.choice(num_inputs, m_inputs, replace=False)
    hidden_indices = np.arange(num_inputs, total_features) 
    
    # Apply hidden features to linear terms if requested
    if hidden_in_linear:
        all_contributing_indices = np.concatenate([visible_indices, hidden_indices])
    else:
        all_contributing_indices = visible_indices
        
    linear_weights = np.random.uniform(weight_range[0], weight_range[1], len(all_contributing_indices))
    sorted_linear = sorted(zip(all_contributing_indices, linear_weights), key=lambda x: abs(x[1]), reverse=True)

    # 3. Interaction components
    n_interactions = np.random.randint(num_interactions[0], num_interactions[1] + 1)
    interactions = []
    
    for i in range(n_interactions):
        if hidden_in_interactions and i == 0 and n_hidden > 0:
            # Force the first interaction to explicitly include a hidden feature
            h_feat = np.random.choice(hidden_indices)
            other_pool = list(range(total_features))
            other_pool.remove(h_feat)
            other_feat = np.random.choice(other_pool)
            pair = sorted([h_feat, other_feat])
        elif hidden_in_interactions:
            # Allow any combination (Visible*Visible, Visible*Hidden, Hidden*Hidden)
            pair = sorted(np.random.choice(total_features, 2, replace=False))
        else:
            # Strictly Visible*Visible interactions
            pair = sorted(np.random.choice(num_inputs, 2, replace=False))
            
        weight = np.random.uniform(interaction_weight_range[0], interaction_weight_range[1])
        interactions.append((pair[0], pair[1], weight))

    # 4. Build the True Equation String
    equation_parts = []
    for idx, w in sorted_linear:
        equation_parts.append(f"({w:.4f} * {get_feat_name(idx)})")
        
    sorted_interactions = sorted(interactions, key=lambda x: abs(x[2]), reverse=True)
    for i, j, w in sorted_interactions:
        equation_parts.append(f"({w:.4f} * {get_feat_name(i)} * {get_feat_name(j)})")
        
    equation_str = "y = " + " + ".join(equation_parts) + f" + Noise(0, {noise_std})"

    # Print Truth
    print("--- Generative Model Info (With HIDDEN Features) ---")
    print(f"Number of linear variables: {len(all_contributing_indices)} ({m_inputs} visible, {n_hidden if hidden_in_linear else 0} hidden)")
    print(f"Number of interaction terms: {n_interactions}")
    print("\n--- True Mathematical Equation ---")
    print(equation_str)
    print("-" * 52 + "\n")

    # 5. Generate the full dataset
    full_feature_names = [get_feat_name(i) for i in range(total_features)]
    X_full = pd.DataFrame(np.random.rand(num_samples, total_features), columns=full_feature_names)

    # 6. Calculate y
    y = np.zeros(num_samples)
    
    for idx, w in zip(all_contributing_indices, linear_weights):
        y += w * X_full[get_feat_name(idx)]
        
    for i, j, w in interactions:
        y += w * (X_full[get_feat_name(i)] * X_full[get_feat_name(j)])

    y += np.random.normal(0, noise_std, num_samples)
    
    # 7. Split X into visible and hidden DataFrames
    visible_feature_names = [f"Feature_{i}" for i in range(num_inputs)]
    hidden_feature_names = [f"Hidden_{i}" for i in range(n_hidden)]
    
    X_visible = X_full[visible_feature_names].copy()
    X_hidden = X_full[hidden_feature_names].copy()
    
    return X_visible, X_hidden, y, visible_feature_names, hidden_feature_names


# ==========================================
# 2. MODELING & EVALUATION FUNCTIONS
# ==========================================

def perform_traditional_regression(X_train: pd.DataFrame, y_train: np.ndarray, pvalue_threshold: float = 0.05
                                   ) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, dict[str, dict[str, float]]]:
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

    # Identify statistically significant features (p < pvalue_threshold)
    sig_mask = pvalues < pvalue_threshold
    sig_pvalues = pvalues[sig_mask]
    sig_coefs = coefs[sig_mask]
    
    # Create the dictionary of significant features
    unsorted_sig_features = {}
    for feat in sig_pvalues.index:
        unsorted_sig_features[feat] = {
            'pvalue': sig_pvalues[feat],
            'coefficient': sig_coefs[feat]
        }
        
    # Sort the dictionary by p-value (ascending: most significant first)
    significant_features = dict(sorted(unsorted_sig_features.items(), key=lambda item: item[1]['pvalue']))

    print("\n" + "="*50)
    print(f"Statistically Significant Features (p < {pvalue_threshold}):")
    if significant_features:
        for feat, stats in significant_features.items():
            print(f"  - {feat} (p-value: {stats['pvalue']:.4e}, coef: {stats['coefficient']:.4f})")
    else:
        print(f"  None detected at p < {pvalue_threshold}")
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
    
    return ols_model, significant_features


def train_xgb_model(X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2, random_state: int = 4,
                    ) -> tuple[xgb.XGBRegressor, pd.DataFrame, pd.DataFrame]:
    """Splits data and trains a basic XGBoost Regressor."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
    model.fit(X_train, y_train)
    
    return model, X_train, X_test


def evaluate_xgb_model(model: xgb.XGBRegressor, X_test: pd.DataFrame, y_test: np.ndarray) -> None:
    """Evaluates the XGBoost model and prints regression metrics."""
    print("\n--- XGBoost Performance (Testing) ---")
    
    # Generate predictions on the unseen test data
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print results
    print(f"R-squared Score:                {r2:.4f}")
    print(f"Mean Squared Error (MSE):       {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE):      {mae:.4f}")
    print("-" * 37 + "\n")


# ==========================================
# 3. PLOTTING & SHAP EXPLAINABILITY FUNCTIONS
# ==========================================

def plot_all_xgb_importances(model) -> tuple[dict, dict, dict]:
    """Plots and outputs Weight, Gain, and Cover XGBoost feature importances."""
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

def compute_shap_values(model: xgb.XGBRegressor, X_train: pd.DataFrame, X_test: pd.DataFrame, max_background_samples: int = 100) -> shap.Explanation:
    """Computes SHAP values using the training set as a background distribution."""
    # masker = shap.maskers.Independent(X_train, max_samples=max_background_samples)
    # explainer = shap.Explainer(model.predict, masker)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    
    return shap_values

def plot_shap_analysis(shap_values: shap.Explanation, feature_names: list[str]) -> None:
    """Generates the beeswarm plot and detailed feature dependence plots with improvements."""
    # 1. Beeswarm Plot
    plt.figure(figsize=(10, 6))
    plt.title("SHAP Beeswarm Plot")
    shap.plots.beeswarm(shap_values)
    plt.show()

    # 2. Feature Dependence Plots
    num_features = len(feature_names)
    cols = 5
    rows = (num_features + cols - 1) // cols 

    # Using sharey=True to standardize vertical scales across each row
    fig1, axes1 = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 4 * rows), sharey=True)
    fig2, axes2 = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 4 * rows), sharey=True)
    fig3, axes3 = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 4 * rows), sharey=True)

    axes1, axes2, axes3 = axes1.flatten(), axes2.flatten(), axes3.flatten()

    for i, feature in enumerate(feature_names):
        x_vals = shap_values.data[:, i]
        y_vals = shap_values.values[:, i]
        
        # Adaptive Epsilon based on standard deviation
        x_std = np.std(x_vals)
        epsilon = 1e-3 * x_std if x_std > 0 else 1e-3
        
        y_vals_normalize = y_vals / (x_vals - x_vals.mean() + epsilon) 
        
        sort_idx = np.argsort(x_vals)
        x_sorted, y_sorted = x_vals[sort_idx], y_vals[sort_idx]
        
        # Clipping the derivative to fix divide-by-zero spikes
        dx = x_sorted[1:] - x_sorted[:-1] + epsilon
        dy = y_sorted[1:] - y_sorted[:-1]
        raw_derivative = dy / dx
        
        # Clip between 1st and 99th percentiles to keep sharey=True from exploding
        lower_bound = np.percentile(raw_derivative, 1)
        upper_bound = np.percentile(raw_derivative, 99)
        y_vals_normalize2 = np.clip(raw_derivative, lower_bound, upper_bound)

        # Color by a highly correlated interaction proxy
        correlations = []
        for j in range(num_features):
            if j == i or np.std(shap_values.data[:, j]) == 0:
                correlations.append(0)
            else:
                corr = np.corrcoef(y_vals, shap_values.data[:, j])[0, 1]
                correlations.append(corr if not np.isnan(corr) else 0)
        
        interact_idx = np.argmax(np.abs(correlations))
        c_vals = shap_values.data[:, interact_idx]
        interact_name = feature_names[interact_idx]

        # FIXED: Sort the color array to match x_sorted, and drop the last element for the derivative
        c_sorted = c_vals[sort_idx]

        plot_kwargs_main = {'alpha': 0.4, 's': 10, 'c': c_vals, 'cmap': 'coolwarm'}
        plot_kwargs_deriv = {'alpha': 0.4, 's': 10, 'c': c_sorted[:-1], 'cmap': 'coolwarm'}

        # Plot 1: Standard SHAP
        axes1[i].scatter(x_vals, y_vals, **plot_kwargs_main)
        axes1[i].set(title=f"SHAP vs {feature}", xlabel="Feature")
        if i % cols == 0: axes1[i].set_ylabel("SHAP")

        # Plot 2: Normalized SHAP
        axes2[i].scatter(x_vals, y_vals_normalize, **plot_kwargs_main)
        axes2[i].set(title=f"SHAP/Feature vs {feature}", xlabel="Feature")
        if i % cols == 0: axes2[i].set_ylabel("SHAP / Feature")

        # Plot 3: Derivative (Uses the sorted, N-1 color array)
        axes3[i].scatter(x_sorted[:-1], y_vals_normalize2, **plot_kwargs_deriv)
        axes3[i].set(title=f"d(SHAP)/d(Feature)", xlabel="Feature")
        if i % cols == 0: axes3[i].set_ylabel("Derivative")
        
        # Add a zero reference line and a text box showing the interaction color variable
        for ax in [axes1[i], axes2[i], axes3[i]]:
            ax.axhline(0, color='gray', linestyle='--', linewidth=1)
            ax.text(0.05, 0.95, f"Color: {interact_name}", transform=ax.transAxes, 
                    fontsize=8, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='lightgray'))

    # Clean up empty subplots
    for j in range(num_features, len(axes1)):
        fig1.delaxes(axes1[j])
        fig2.delaxes(axes2[j])
        fig3.delaxes(axes3[j])

    for fig in [fig1, fig2, fig3]:
        fig.tight_layout()
        
    plt.show()

def print_feature_importance(shap_values: shap.Explanation) -> dict[str, float]:
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
    
    return sorted_importance


def analyze_shap_interactions(model: xgb.XGBRegressor, X_test: pd.DataFrame, feature_names: list[str]) -> dict[str, float]:
    """Computes True SHAP Interaction Values, plots the strongest pair, and returns a dictionary of global interaction importances."""
    print("\n--- Computing SHAP Interaction Values ---")
    # TreeExplainer calculates interactions directly from the tree structure
    explainer = shap.TreeExplainer(model)
    
    # Output shape is (n_samples, n_features, n_features)
    interaction_values = explainer.shap_interaction_values(X_test)
    
    # 1. Global Interaction Summary
    shap.summary_plot(interaction_values, X_test, feature_names=feature_names, max_display=len(feature_names), show=False)
    plt.show()
    
    # 2. Build the Global Interaction Dictionary
    # Take mean absolute value across all samples
    mean_abs_interactions = np.abs(interaction_values).mean(axis=0)
    
    interaction_dict = {}
    n_features = len(feature_names)
    
    for i in range(n_features):
        # We only iterate the upper triangle (j > i) to avoid duplicate pairs 
        # and to exclude the diagonal (which represents main effects, not interactions)
        for j in range(i + 1, n_features):
            # SHAP interaction matrices are symmetric, splitting the effect equally between [i, j] and [j, i].
            # We multiply by 2 to capture the total interaction effect for the pair.
            total_interaction = mean_abs_interactions[i, j] * 2
            pair_name = f"{feature_names[i]} * {feature_names[j]}"
            interaction_dict[pair_name] = total_interaction
            
    # Sort from most important interaction to least important
    sorted_interactions = dict(sorted(interaction_dict.items(), key=lambda item: item[1], reverse=True))
    
    # 3. Find the strongest interacting pair
    strongest_pair = list(sorted_interactions.keys())[0]
    feat_i, feat_j = strongest_pair.split(" * ")
    
    print(f"\nStrongest Interaction found by SHAP: {feat_i} & {feat_j}")
    
    # 4. Plot specific dependence for the strongest interaction
    print(f"Plotting pure interaction effect for {feat_i} and {feat_j}...")
    
    idx_i = feature_names.index(feat_i)
    idx_j = feature_names.index(feat_j)
    
    shap.dependence_plot(
        (idx_i, idx_j), 
        interaction_values, 
        X_test, 
        feature_names=feature_names,
        show=False
    )
    plt.show()

    # 5. Print the ranked interaction dictionary
    print("\n--- SHAP Interaction Feature Importance Ranking ---")
    print("Top 10 Strongest Interactions:")
    # Slice top 10 so we don't flood the console for datasets with many features
    for pair, imp in list(sorted_interactions.items())[:10]: 
        print(f"  {pair}: {imp:.4f}")
    print("-" * 49 + "\n")

    return sorted_interactions

# ==========================================
# 4. MAIN PIPELINE
# ==========================================

def main():
    """Main execution pipeline."""
    # 1. Setup & Data Generation
    
    # Normal Linear Data (Commented out as in original)
    # X, y, feature_names = generate_linear_synthetic_data(
    #     num_inputs=10, 
    #     num_samples=5000, 
    #     num_contributing_features=(2, 5), 
    #     weight_range=(-3, 3),
    #     noise_std=0.05,
    # )
    
    # With Interactions Data (Commented out as in original)
    # X, y, feature_names = generate_synthetic_data_with_interactions(
    #     num_inputs=10, 
    #     num_samples=5000, 
    #     num_contributing_features=(2, 5), 
    #     weight_range=(-3, 3),
    #     num_interactions=(1, 2),
    #     interaction_weight_range=(-3, 3),
    #     noise_std=0.05,
    # )
    
    # With Hidden Features
    X, hidden_features, y, feature_names, hidden_feature_names = generate_synthetic_data_with_hidden_features(
        num_inputs = 10, 
        num_samples = 5000, 
        num_contributing_features = (2, 5), 
        num_hidden_features = (1, 2), 
        weight_range = (-3, 3), 
        num_interactions = (1, 2), 
        interaction_weight_range = (-3, 3), 
        noise_std = 0.05, 
        hidden_in_linear = True,           # Should hidden features have a linear impact?
        hidden_in_interactions = True,     # Should hidden features be part of interactions?
    )

    # Split data here so both OLS and XGBoost train on the exact same subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Traditional Baseline (OLS)
    ols_model, significant_features = perform_traditional_regression(X_train, y_train)
    
    # 3. Machine Learning Modeling (XGBoost)
    # We pass the pre-split data directly to ensure apples-to-apples comparison
    model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
    model.fit(X_train, y_train)
    
    # 4. Evaluation & XGBoost Native Importances
    evaluate_xgb_model(model, X_test, y_test)
    weight_importance, gain_importance, cover_importance = plot_all_xgb_importances(model)
    
    # 5. SHAP Explainability & Visualization
    print("Computing SHAP values...")
    shap_values = compute_shap_values(model, X_train, X_test)
    plot_shap_analysis(shap_values, feature_names)
    
    # 6. Deep Dive: SHAP Interaction Analysis
    sorted_interactions = analyze_shap_interactions(model, X_test, feature_names)

    # 7. SHAP Feature Summary
    ranked_features = print_feature_importance(shap_values)

if __name__ == "__main__":
    main()