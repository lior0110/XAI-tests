import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
import gc

# ==========================================
# 1. SYNTHETIC DATA GENERATION FUNCTIONS
# ==========================================

def generate_linear_synthetic_data(num_inputs: int = 10, num_samples: int = 5000, 
                            num_contributing_features: tuple[int, int] = (2, 5), 
                            input_range: tuple[float, float] = (-1.0, 1.0),
                            weight_range: tuple[float, float] = (-3, 3), 
                            noise_std: float = 0.05,
                            ) -> tuple[pd.DataFrame, np.ndarray, list[str], dict[str, float]]:
    """Generates synthetic data with a random linear combination of features."""
    m_inputs = np.random.randint(num_contributing_features[0], num_contributing_features[1] + 1)
    input_indices = np.random.choice(num_inputs, m_inputs, replace=False)
    weights = np.random.uniform(weight_range[0], weight_range[1], m_inputs)

    sorted_features = sorted(zip(input_indices, weights), key=lambda x: abs(x[1]), reverse=True)
    
    equation_parts = []
    true_linear_weights = {}
    
    for idx, w in sorted_features:
        feat_name = f"Feature_{idx}"
        equation_parts.append(f"({w:.4f} * {feat_name})")
        true_linear_weights[feat_name] = float(w)

    equation_str = "y = " + " + ".join(equation_parts) + f" + Noise(0, {noise_std})"

    print("--- Generative Model Info ---")
    print(f"Number of contributing variables (M): {m_inputs}")
    print("\n--- True Mathematical Equation ---")
    print(equation_str)
    print("-" * 39 + "\n")

    feature_names = [f"Feature_{i}" for i in range(num_inputs)]
    X = pd.DataFrame(np.random.uniform(input_range[0], input_range[1], (num_samples, num_inputs)), columns=feature_names)

    y = np.zeros(num_samples)
    for idx, w in zip(input_indices, weights):
        y += w * X[f"Feature_{idx}"]

    y += np.random.normal(0, noise_std, num_samples)
    
    return X, y, feature_names, true_linear_weights


def generate_synthetic_data_with_interactions(num_inputs: int = 10, num_samples: int = 5000, 
                            num_contributing_features: tuple[int, int] = (2, 5), 
                            input_range: tuple[float, float] = (-1.0, 1.0),
                            weight_range: tuple[float, float] = (-3, 3), 
                            num_interactions: tuple[int, int] = (1, 2), 
                            interaction_weight_range: tuple[float, float] = (-3, 3), 
                            noise_std: float = 0.05,
                            ) -> tuple[pd.DataFrame, np.ndarray, list[str], dict[str, float], dict[str, float | list[float]]]:
    """Generates synthetic data with linear combinations AND 1-2 feature interactions."""
    m_inputs = np.random.randint(num_contributing_features[0], num_contributing_features[1] + 1)
    linear_indices = np.random.choice(num_inputs, m_inputs, replace=False)
    linear_weights = np.random.uniform(weight_range[0], weight_range[1], m_inputs)
    sorted_linear = sorted(zip(linear_indices, linear_weights), key=lambda x: abs(x[1]), reverse=True)

    num_interactions = np.random.randint(num_interactions[0], num_interactions[1] + 1)
    interactions = []
    for _ in range(num_interactions):
        pair = np.random.choice(num_inputs, 2, replace=False)
        weight = np.random.uniform(interaction_weight_range[0], interaction_weight_range[1], 1)[0]
        pair = sorted(pair) 
        interactions.append((pair[0], pair[1], weight))

    sorted_interactions = sorted(interactions, key=lambda x: abs(x[2]), reverse=True)

    equation_parts = []
    true_linear_weights = {}
    true_interaction_weights = {}

    def add_interaction_weight(feat, w):
        if feat in true_interaction_weights:
            if isinstance(true_interaction_weights[feat], list):
                true_interaction_weights[feat].append(w)
            else:
                true_interaction_weights[feat] = [true_interaction_weights[feat], w]
        else:
            true_interaction_weights[feat] = w

    for idx, w in sorted_linear:
        feat_name = f"Feature_{idx}"
        equation_parts.append(f"({w:.4f} * {feat_name})")
        true_linear_weights[feat_name] = float(w)
        
    for i, j, w in sorted_interactions:
        feat_i = f"Feature_{i}"
        feat_j = f"Feature_{j}"
        pair_name = f"{feat_i} * {feat_j}"
        equation_parts.append(f"({w:.4f} * {pair_name})")
        
        # Add the weight to both individual features in the interaction dictionary
        add_interaction_weight(feat_i, float(w))
        add_interaction_weight(feat_j, float(w))
        
    equation_str = "y = " + " + ".join(equation_parts) + f" + Noise(0, {noise_std})"

    print("--- Generative Model Info (With Interactions) ---")
    print(f"Number of linear variables (M): {m_inputs}")
    print(f"Number of interaction terms: {num_interactions}")
    print("\n--- True Mathematical Equation ---")
    print(equation_str)
    print("-" * 49 + "\n")

    feature_names = [f"Feature_{i}" for i in range(num_inputs)]
    X = pd.DataFrame(np.random.uniform(input_range[0], input_range[1], (num_samples, num_inputs)), columns=feature_names)

    y = np.zeros(num_samples)
    for idx, w in zip(linear_indices, linear_weights):
        y += w * X[f"Feature_{idx}"]
        
    for i, j, w in interactions:
        y += w * (X[f"Feature_{i}"] * X[f"Feature_{j}"])

    y += np.random.normal(0, noise_std, num_samples)
    
    return X, y, feature_names, true_linear_weights, true_interaction_weights


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
    hidden_in_interactions: bool = True,
    input_range: tuple[float, float] = (-1.0, 1.0),
    missing_pct: float = 0.0,
    error_pct: float = 0.0
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str], list[str], dict[str, float], dict[str, float | list[float]]]:
    """Generates synthetic data where hidden features impact 'y' based on user demand, with optional missing/error noise."""
    if not hidden_in_linear and not hidden_in_interactions:
        print("WARNING: Hidden features are excluded from both linear and interaction parts. They will not impact 'y'.")

    # 1. Determine Total Counts
    total_linear_features = np.random.randint(num_contributing_features[0], num_contributing_features[1] + 1)
    n_hidden_total = np.random.randint(num_hidden_features[0], num_hidden_features[1] + 1)
    total_features_dataset = num_inputs + n_hidden_total
    
    def get_feat_name(idx):
        return f"Feature_{idx}" if idx < num_inputs else f"Hidden_{idx - num_inputs}"

    # 2. Linear components (Split total between visible and hidden)
    if hidden_in_linear:
        m_visible_linear = np.random.randint(max(1, total_linear_features - n_hidden_total), total_linear_features)
        n_hidden_linear = total_linear_features - m_visible_linear
        
        visible_indices = np.random.choice(num_inputs, m_visible_linear, replace=False)
        hidden_pool = np.arange(num_inputs, total_features_dataset)
        hidden_indices = np.random.choice(hidden_pool, n_hidden_linear, replace=False)
        
        all_contributing_indices = np.concatenate([visible_indices, hidden_indices])
    else:
        m_visible_linear = total_linear_features
        n_hidden_linear = 0
        all_contributing_indices = np.random.choice(num_inputs, m_visible_linear, replace=False)
        
    linear_weights = np.random.uniform(weight_range[0], weight_range[1], len(all_contributing_indices))
    sorted_linear = sorted(zip(all_contributing_indices, linear_weights), key=lambda x: abs(x[1]), reverse=True)

    # 3. Interaction components
    n_interactions = np.random.randint(num_interactions[0], num_interactions[1] + 1)
    interactions = []
    hidden_interaction_count = 0 
    
    for i in range(n_interactions):
        if hidden_in_interactions and i == 0 and n_hidden_total > 0:
            h_feat = np.random.choice(np.arange(num_inputs, total_features_dataset))
            other_pool = list(range(total_features_dataset))
            other_pool.remove(h_feat)
            other_feat = np.random.choice(other_pool)
            pair = sorted([h_feat, other_feat])
            hidden_interaction_count += 1
        elif hidden_in_interactions:
            pair = sorted(np.random.choice(total_features_dataset, 2, replace=False))
            if pair[0] >= num_inputs or pair[1] >= num_inputs:
                hidden_interaction_count += 1
        else:
            pair = sorted(np.random.choice(num_inputs, 2, replace=False))
            
        weight = np.random.uniform(interaction_weight_range[0], interaction_weight_range[1])
        interactions.append((pair[0], pair[1], weight))

    # 4. Build Equation and Ground Truth Dictionaries
    equation_parts = []
    true_linear_weights = {}
    true_interaction_weights = {}

    def add_interaction_weight(feat, w):
        if feat in true_interaction_weights:
            if isinstance(true_interaction_weights[feat], list):
                true_interaction_weights[feat].append(w)
            else:
                true_interaction_weights[feat] = [true_interaction_weights[feat], w]
        else:
            true_interaction_weights[feat] = w

    for idx, w in sorted_linear:
        feat_name = get_feat_name(idx)
        equation_parts.append(f"({w:.4f} * {feat_name})")
        true_linear_weights[feat_name] = float(w)
        
    sorted_interactions = sorted(interactions, key=lambda x: abs(x[2]), reverse=True)
    for i, j, w in sorted_interactions:
        feat_i = get_feat_name(i)
        feat_j = get_feat_name(j)
        pair_name = f"{feat_i} * {feat_j}"
        equation_parts.append(f"({w:.4f} * {pair_name})")
        
        add_interaction_weight(feat_i, float(w))
        add_interaction_weight(feat_j, float(w))
        
    equation_str = "y = " + " + ".join(equation_parts) + f" + Noise(0, {noise_std})"

    print("--- Generative Model Info (With HIDDEN Features) ---")
    print(f"Total linear variables:  {total_linear_features} ({m_visible_linear} visible, {n_hidden_linear} hidden)")
    print(f"Total interaction terms: {n_interactions} ({n_interactions - hidden_interaction_count} pure visible, {hidden_interaction_count} involving hidden)")
    print("\n--- True Mathematical Equation ---")
    print(equation_str)
    print("-" * 52 + "\n")

    # 5. Generate clean data
    full_feature_names = [get_feat_name(i) for i in range(total_features_dataset)]
    X_full = pd.DataFrame(np.random.uniform(input_range[0], input_range[1], (num_samples, total_features_dataset)), columns=full_feature_names)

    # 6. Calculate y perfectly based on clean data
    y = np.zeros(num_samples)
    for idx, w in zip(all_contributing_indices, linear_weights):
        y += w * X_full[get_feat_name(idx)]
        
    for i, j, w in interactions:
        y += w * (X_full[get_feat_name(i)] * X_full[get_feat_name(j)])

    y += np.random.normal(0, noise_std, num_samples)
    
    # 7. Apply Missing and Error rates to the input features (X) AFTER calculating y
    total_cells = num_samples * total_features_dataset
    num_missing = int(total_cells * missing_pct)
    num_error = int(total_cells * error_pct)
    
    if num_missing > 0 or num_error > 0:
        X_arr = X_full.to_numpy()
        
        # Pick random distinct locations to corrupt
        corrupt_indices = np.random.choice(total_cells, num_missing + num_error, replace=False)
        missing_indices = corrupt_indices[:num_missing]
        error_indices = corrupt_indices[num_missing:]
        
        # Inject NaN values
        if num_missing > 0:
            X_arr.flat[missing_indices] = np.nan
            
        # Inject Error values (Resampled from the same distribution as the input data)
        if num_error > 0:
            X_arr.flat[error_indices] = np.random.uniform(input_range[0], input_range[1], size=num_error)

        # Re-assign back to DataFrame
        X_full = pd.DataFrame(X_arr, columns=full_feature_names)

    # 8. Split X into visible and hidden DataFrames
    visible_feature_names = [f"Feature_{i}" for i in range(num_inputs)]
    hidden_feature_names = [f"Hidden_{i}" for i in range(n_hidden_total)]
    
    X_visible = X_full[visible_feature_names].copy()
    X_hidden = X_full[hidden_feature_names].copy()
    
    return X_visible, X_hidden, y, visible_feature_names, hidden_feature_names, true_linear_weights, true_interaction_weights

# ==========================================
# 2. MODELING & EVALUATION FUNCTIONS
# ==========================================

def perform_traditional_regression(X: pd.DataFrame, y: np.ndarray, pvalue_threshold: float = 0.05, use_bonferroni: bool = True, nan_strategy: str = "impute_mean"):
    """
    Performs traditional Ordinary Least Squares (OLS) regression using statsmodels.
    Handles NaN values and extracts statistically significant features.
    """
    print(f"\n--- Traditional OLS Regression (NaN Strategy: {nan_strategy}) ---")
    
    if use_bonferroni:
        pvalue_threshold = pvalue_threshold / X.shape[1]
        print(f"Bonferroni correction applied: Adjusted p-value threshold = {pvalue_threshold:.4e}")

    # 1. Handle Missing Values
    if X.isna().sum().sum() > 0:
        print(f"Warning: {X.isna().sum().sum()} missing values detected in X.")
        
        if nan_strategy == "drop":
            valid_rows_mask = ~X.isna().any(axis=1)
            X_clean = X[valid_rows_mask].copy()
            y_clean = y[valid_rows_mask].copy() if isinstance(y, np.ndarray) else y[valid_rows_mask].copy()
            print(f"Dropped {len(X) - len(X_clean)} rows containing NaNs.")
            
        elif nan_strategy == "impute_mean":
            X_clean = X.fillna(X.mean())
            y_clean = y.copy()
            print("Imputed missing values with column means.")
            
        else:
            raise ValueError("nan_strategy must be either 'drop' or 'impute_mean'")
    else:
        X_clean = X.copy()
        y_clean = y.copy()
        
    # 2. Add a constant (intercept) to the model
    X_clean_sm = sm.add_constant(X_clean)
    
    # 3. Fit the OLS model
    ols_model = sm.OLS(y_clean, X_clean_sm).fit()
    print(ols_model.summary())
    
    # 4. Extract coefficients and calculate significant features
    coefs = ols_model.params.drop('const')
    pvalues = ols_model.pvalues.drop('const')
    conf_int = ols_model.conf_int().drop('const')

    sig_mask = pvalues < pvalue_threshold
    sig_pvalues = pvalues[sig_mask]
    sig_coefs = coefs[sig_mask]
    
    unsorted_sig_features = {}
    for feat in sig_pvalues.index:
        unsorted_sig_features[feat] = {
            'pvalue': sig_pvalues[feat],
            'coefficient': sig_coefs[feat]
        }
        
    significant_features = dict(sorted(unsorted_sig_features.items(), key=lambda item: item[1]['pvalue']))

    print("\n" + "="*50)
    print(f"Statistically Significant Features (p < {pvalue_threshold}):")
    if significant_features:
        for feat, stats in significant_features.items():
            print(f"  - {feat} (p-value: {stats['pvalue']:.4e}, coef: {stats['coefficient']:.4f})")
    else:
        print(f"  None detected at p < {pvalue_threshold}")
    print("="*50 + "\n")
    
    # 5. Plotting Coefficients with 95% Confidence Intervals
    lower_errors = coefs - conf_int[0]
    upper_errors = conf_int[1] - coefs
    
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

def train_xgb_model(
    X: pd.DataFrame, 
    y: np.ndarray, 
    test_size: float = 0.2, 
    random_state: int = None,
    **xgb_kwargs
) -> tuple[xgb.XGBRegressor, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Splits data and trains an XGBoost Regressor.
    
    Args:
        X: Input features.
        y: Target variable.
        test_size: Proportion of data to use for testing.
        random_state: Random seed for splitting.
        **xgb_kwargs: Any valid parameters for xgb.XGBRegressor 
                      (e.g., n_estimators, max_depth, reg_alpha, etc.)
    """
    print("\n--- Training XGBoost Model ---")
    
    # 1. Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 2. Define default parameters
    # We explicitly set missing=np.nan to leverage XGBoost's native missing value handling
    params = {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.1,
        'missing': np.nan, 
        'random_state': random_state
    }
    
    # 3. Overwrite/add any parameters passed in via **xgb_kwargs
    params.update(xgb_kwargs)
    
    # 4. Initialize and fit the model
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    print(f"Model trained with parameters: {params}")
    
    return model, X_train, X_test, y_train, y_test

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

def evaluate_feature_discovery(
    significant_features: dict | list | str, 
    true_linear_weights: dict[str, float], 
    true_interaction_weights: dict[str, float | list[float]]
) -> dict:
    """
    Evaluates how well the ML model discovered the true generative features.
    Provides separate recall metrics for linear vs. interaction features.
    """
    def _extract_score(val):
        try:
            if isinstance(val, dict):
                for k in ['importance', 'score', 'shap_value', 'weight', 'mean_abs_shap']:
                    if k in val: return float(val[k])
                return float(list(val.values())[0])
            elif isinstance(val, (list, tuple, np.ndarray)):
                return float(val[0])
            return float(val)
        except Exception:
            return 1.0

    # 1. Standardize Model Output
    if isinstance(significant_features, str):
        model_features = [f.strip() for f in significant_features.split(',')]
        model_importances = {f: 1.0 for f in model_features}
    elif isinstance(significant_features, list):
        model_features = significant_features
        model_importances = {f: 1.0 for f in model_features}
    elif isinstance(significant_features, dict):
        model_features = list(significant_features.keys())
        model_importances = significant_features
    else:
        raise ValueError("significant_features must be a dict, list, or string.")
        
    # 2. Extract Ground Truth Sets (Excluding Hidden Features)
    true_linear_set = {f for f in true_linear_weights.keys() if not f.startswith("Hidden")}
    true_interaction_set = {f for f in true_interaction_weights.keys() if not f.startswith("Hidden")}
    
    true_features_set = true_linear_set.union(true_interaction_set)
    model_features_set = set(model_features)
    
    # 3. Calculate Global Metrics
    true_positives = true_features_set.intersection(model_features_set)
    false_positives = model_features_set - true_features_set
    false_negatives = true_features_set - model_features_set
    
    tp_count = len(true_positives)
    fp_count = len(false_positives)
    fn_count = len(false_negatives)
    
    precision = tp_count / len(model_features_set) if len(model_features_set) > 0 else 0.0
    recall_overall = tp_count / len(true_features_set) if len(true_features_set) > 0 else 0.0
    f1_score = 2 * (precision * recall_overall) / (precision + recall_overall) if (precision + recall_overall) > 0 else 0.0
    
    # 4. Calculate Specific Recalls (Linear vs Interaction)
    linear_hits = true_linear_set.intersection(model_features_set)
    interaction_hits = true_interaction_set.intersection(model_features_set)
    
    # If there are no linear/interaction features in the ground truth, we consider recall to be N/A (1.0)
    recall_linear = len(linear_hits) / len(true_linear_set) if len(true_linear_set) > 0 else 1.0
    recall_interaction = len(interaction_hits) / len(true_interaction_set) if len(true_interaction_set) > 0 else 1.0

    # 5. Calculate Aggregate True Weights (For Ranking Analysis)
    true_aggregate_importance = {}
    for f in true_features_set:
        imp = 0.0
        if f in true_linear_weights:
            imp += abs(true_linear_weights[f])
        if f in true_interaction_weights:
            w_int = true_interaction_weights[f]
            if isinstance(w_int, list):
                imp += sum(abs(x) for x in w_int)
            else:
                imp += abs(w_int)
        true_aggregate_importance[f] = imp
        
    # 6. Rank Correlation (Spearman)
    correlation, p_value = None, None
    if isinstance(significant_features, dict) and len(true_positives) > 1:
        shared_features = list(true_positives)
        true_scores = [true_aggregate_importance[f] for f in shared_features]
        model_scores = [abs(_extract_score(model_importances[f])) for f in shared_features]
        
        if len(set(model_scores)) == 1:
             model_scores = [s + np.random.normal(0, 1e-10) for s in model_scores]

        correlation, p_value = spearmanr(true_scores, model_scores)

    # 7. Print Summary
    print("\n" + "="*50)
    print("           FEATURE DISCOVERY EVALUATION")
    print("="*50)
    print(f"Total True (Visible) Features: {len(true_features_set)}")
    print(f"  ├─ Purely Linear / Main:     {len(true_linear_set)}")
    print(f"  └─ Involved in Interactions: {len(true_interaction_set)}")
    print(f"Total Model Features Found:    {len(model_features_set)}")
    print("-" * 50)
    print(f"True Positives (Correct): {tp_count} -> {list(true_positives)}")
    print(f"False Positives (Noise):  {fp_count} -> {list(false_positives)}")
    print(f"False Negatives (Missed): {fn_count} -> {list(false_negatives)}")
    print("-" * 50)
    print(f"Precision:            {precision:.4f}")
    print(f"Overall Recall:       {recall_overall:.4f}")
    print(f"  ├─ Linear Recall:   {recall_linear:.4f}  (Found {len(linear_hits)}/{len(true_linear_set)})")
    print(f"  └─ Interact Recall: {recall_interaction:.4f}  (Found {len(interaction_hits)}/{len(true_interaction_set)})")
    print(f"F1-Score:             {f1_score:.4f}")
    
    if correlation is not None:
        print(f"Rank Correlation:     {correlation:.4f} (p-val: {p_value:.4f})")
    print("="*50 + "\n")

    return {
        "precision": precision,
        "recall_overall": recall_overall,
        "recall_linear": recall_linear,
        "recall_interaction": recall_interaction,
        "f1_score": f1_score,
        "true_positives": list(true_positives),
        "false_positives": list(false_positives),
        "false_negatives": list(false_negatives),
        "spearman_correlation": correlation
    }

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

def plot_shap_analysis(shap_values: shap.Explanation, feature_names: list[str], top_n: int = 5) -> None:
    """
    Generates beeswarm and detailed dependence plots for the top N most important features.
    """
    # 1. Identify Top N Features based on mean absolute SHAP
    global_importances = np.abs(shap_values.values).mean(0)
    # Get indices of top_n features, sorted descending
    top_indices = np.argsort(global_importances)[-top_n:][::-1]
    
    # Update count in case top_n is larger than available features
    num_to_plot = len(top_indices)
    plot_feature_names = [feature_names[i] for i in top_indices]

    print(f"Plotting SHAP analysis for top {num_to_plot} features: {plot_feature_names}")

    # 2. Beeswarm Plot (automatically handles top_n via max_display)
    plt.figure(figsize=(10, 6))
    plt.title(f"SHAP Beeswarm Plot (Top {num_to_plot})")
    shap.plots.beeswarm(shap_values, max_display=num_to_plot + 1)
    plt.show()

    # 3. Feature Dependence Plots Setup
    cols = 5
    rows = (num_to_plot + cols - 1) // cols 

    fig1, axes1 = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 4 * rows), squeeze=False, sharey=True)
    fig2, axes2 = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 4 * rows), squeeze=False, sharey=True)
    fig3, axes3 = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 4 * rows), squeeze=False, sharey=True)

    axes1, axes2, axes3 = axes1.flatten(), axes2.flatten(), axes3.flatten()

    # 4. Iterate only through the top indices
    for plot_idx, original_feat_idx in enumerate(top_indices):
        feature = feature_names[original_feat_idx]
        x_vals = shap_values.data[:, original_feat_idx]
        y_vals = shap_values.values[:, original_feat_idx]
        
        # Adaptive Epsilon for normalization
        x_std = np.std(x_vals)
        epsilon = 1e-3 * x_std if x_std > 0 else 1e-3
        
        y_vals_normalize = y_vals / (x_vals - x_vals.mean() + epsilon) 
        
        sort_idx = np.argsort(x_vals)
        x_sorted, y_sorted = x_vals[sort_idx], y_vals[sort_idx]
        
        # Derivative Calculation
        dx = x_sorted[1:] - x_sorted[:-1] + epsilon
        dy = y_sorted[1:] - y_sorted[:-1]
        raw_derivative = dy / dx
        
        lower_bound = np.percentile(raw_derivative, 1)
        upper_bound = np.percentile(raw_derivative, 99)
        y_vals_normalize2 = np.clip(raw_derivative, lower_bound, upper_bound)

        # Interaction Proxy (Calculated against all features for best context)
        correlations = []
        for j in range(len(feature_names)):
            if j == original_feat_idx or np.std(shap_values.data[:, j]) == 0:
                correlations.append(0)
            else:
                corr = np.corrcoef(y_vals, shap_values.data[:, j])[0, 1]
                correlations.append(corr if not np.isnan(corr) else 0)
        
        interact_idx = np.argmax(np.abs(correlations))
        c_vals = shap_values.data[:, interact_idx]
        interact_name = feature_names[interact_idx]
        c_sorted = c_vals[sort_idx]

        plot_kwargs_main = {'alpha': 0.4, 's': 10, 'c': c_vals, 'cmap': 'coolwarm'}
        plot_kwargs_deriv = {'alpha': 0.4, 's': 10, 'c': c_sorted[:-1], 'cmap': 'coolwarm'}

        # Plotting
        axes1[plot_idx].scatter(x_vals, y_vals, **plot_kwargs_main)
        axes1[plot_idx].set(title=f"SHAP vs {feature}", xlabel="Value")
        
        axes2[plot_idx].scatter(x_vals, y_vals_normalize, **plot_kwargs_main)
        axes2[plot_idx].set(title=f"SHAP/Feat vs {feature}", xlabel="Value")

        axes3[plot_idx].scatter(x_sorted[:-1], y_vals_normalize2, **plot_kwargs_deriv)
        axes3[plot_idx].set(title=f"d(SHAP)/d({feature})", xlabel="Value")
        
        for ax in [axes1[plot_idx], axes2[plot_idx], axes3[plot_idx]]:
            ax.axhline(0, color='gray', linestyle='--', linewidth=1)
            ax.text(0.05, 0.95, f"Color: {interact_name}", transform=ax.transAxes, 
                    fontsize=8, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='lightgray'))

    # Clean up empty subplots
    for j in range(num_to_plot, len(axes1)):
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



def analyze_shap_interactions_memory_efficient(model, X_test, feature_names, max_samples=500, batch_size=50):
    """
    Memory-efficient SHAP interaction analysis using batching and subsampling.
    """
    print(f"\n--- Computing SHAP Interaction Values (Memory Efficient) ---")
    
    # 1. Subsample X_test if it is too large
    if len(X_test) > max_samples:
        print(f"Subsampling X_test from {len(X_test)} to {max_samples} for efficiency.")
        X_sub = X_test.sample(n=max_samples, random_state=42)
    else:
        X_sub = X_test

    explainer = shap.TreeExplainer(model)
    n_features = len(feature_names)
    
    # Initialize accumulator for mean absolute interaction values (Global Importance)
    # This matrix is only (3000, 3000), which is ~36MB - very safe.
    global_interaction_matrix = np.zeros((n_features, n_features))
    
    # 2. Process in batches to avoid 'Bad Allocation'
    for i in range(0, len(X_sub), batch_size):
        batch = X_sub.iloc[i : i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(X_sub)-1)//batch_size + 1}...")
        
        # Calculate interactions for this batch only
        batch_interactions = explainer.shap_interaction_values(batch)
        
        # Aggregate the absolute values into our global matrix
        global_interaction_matrix += np.abs(batch_interactions).sum(axis=0)
        
        # Explicitly clean up memory
        del batch_interactions
        gc.collect()

    # Final average
    global_interaction_matrix /= len(X_sub)
    
    # 3. Extract and rank the interactions
    interaction_dict = {}
    for i in range(n_features):
        for j in range(i + 1, n_features):
            # SHAP interactions are symmetric; sum [i,j] and [j,i]
            # Multiplied by 2 to capture the total pair effect
            val = global_interaction_matrix[i, j] * 2
            if val > 0:
                pair_name = f"{feature_names[i]} * {feature_names[j]}"
                interaction_dict[pair_name] = val

    sorted_interactions = dict(sorted(interaction_dict.items(), key=lambda x: x[1], reverse=True))
    
    # Print the top interaction
    if sorted_interactions:
        top_pair = list(sorted_interactions.keys())[0]
        print(f"Strongest Interaction: {top_pair} (Score: {sorted_interactions[top_pair]:.4f})")
    
    return sorted_interactions

# ==========================================
# 4. p-like values for SHAP importances
# ==========================================

def compute_shap_pvalues(model: xgb.XGBRegressor, X_train: pd.DataFrame, y_train: np.ndarray, 
                         X_test: pd.DataFrame, true_shap_values: shap.Explanation, 
                         feature_names: list[str], n_permutations: int = 50) -> dict[str, float]:
    """
    Calculates empirical p-values for SHAP feature importances using a permutation test.
    """
    print(f"\n--- Calculating Empirical SHAP p-values (Permutations: {n_permutations}) ---")
    
    true_importances = np.abs(true_shap_values.values).mean(axis=0)
    null_importances = np.zeros((n_permutations, len(feature_names)))
    
    # Extract exact hyperparameters from the base model
    model_params = model.get_params()
    
    for i in range(n_permutations):
        y_train_shuffled = np.random.permutation(y_train)
        
        # Clone the model perfectly
        null_model = xgb.XGBRegressor(**model_params)
        null_model.fit(X_train, y_train_shuffled)
        
        null_explainer = shap.TreeExplainer(null_model)
        null_shap_vals = null_explainer(X_test)
        null_importances[i, :] = np.abs(null_shap_vals.values).mean(axis=0)
        
    p_values = {}
    for idx, feat in enumerate(feature_names):
        count_extreme = np.sum(null_importances[:, idx] >= true_importances[idx])
        p_val = (count_extreme + 1) / (n_permutations + 1)
        p_values[feat] = p_val
        
    sorted_pvalues = dict(sorted(p_values.items(), key=lambda item: item[1]))

    print("\nSHAP Empirical P-values (< 0.05 is statistically significant):")
    for feat, pval in sorted_pvalues.items():
        significance = "***" if pval < 0.01 else ("*" if pval < 0.05 else "")
        print(f"  - {feat}: {pval:.4f} {significance}")
    print("-" * 52 + "\n")
        
    return sorted_pvalues

def compute_shap_shadow_features(model: xgb.XGBRegressor, X_train: pd.DataFrame, y_train: np.ndarray, 
                                 X_test: pd.DataFrame, feature_names: list[str]) -> dict[str, bool]:
    """
    Uses shadow features to determine which SHAP values are statistically significant.
    """
    print("\n--- Running SHAP Shadow Feature Analysis ---")
    
    X_train_shadow = X_train.copy()
    X_test_shadow = X_test.copy()
    
    shadow_names = [f"Shadow_{feat}" for feat in feature_names]
    X_train_shadow.columns = shadow_names
    X_test_shadow.columns = shadow_names
    
    for col in shadow_names:
        X_train_shadow[col] = np.random.permutation(X_train_shadow[col].values)
        X_test_shadow[col] = np.random.permutation(X_test_shadow[col].values)
        
    X_train_extended = pd.concat([X_train, X_train_shadow], axis=1)
    X_test_extended = pd.concat([X_test, X_test_shadow], axis=1)
    
    # Extract params and clone
    model_params = model.get_params()
    shadow_model = xgb.XGBRegressor(**model_params)
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

def compute_shap_bootstrapping(model: xgb.XGBRegressor, X_train: pd.DataFrame, y_train: np.ndarray, 
                               X_test: pd.DataFrame, feature_names: list[str], 
                               n_bootstraps: int = 50) -> dict[str, tuple[float, float, float]]:
    """
    Calculates 95% Confidence Intervals for SHAP feature importances using bootstrapping.
    """
    print(f"\n--- Running SHAP Bootstrapping Analysis (Iterations: {n_bootstraps}) ---")
    
    bootstrap_importances = np.zeros((n_bootstraps, len(feature_names)))
    
    # Extract params and clone
    model_params = model.get_params()
    
    for i in range(n_bootstraps):
        indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_train_boot = X_train.iloc[indices]
        y_train_boot = y_train[indices] if isinstance(y_train, np.ndarray) else y_train.iloc[indices]
        
        # Clone the exact model configuration
        boot_model = xgb.XGBRegressor(**model_params)
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
    """Main execution pipeline."""
    # 1. Setup & Data Generation
    
    # # Normal Linear Data (Commented out as in original)
    # X, y, feature_names, true_linear_weights = generate_linear_synthetic_data(
    #     num_inputs=10, 
    #     num_samples=5000, 
    #     num_contributing_features=(2, 5), 
    #     input_range = (-1.0, 1.0),
    #     weight_range=(-3, 3),
    #     noise_std=0.05,
    # )
    
    # # With Interactions Data (Commented out as in original)
    # X, y, feature_names, true_linear_weights, true_interaction_weights = generate_synthetic_data_with_interactions(
    #     num_inputs=10, 
    #     num_samples=5000, 
    #     num_contributing_features=(2, 5), 
    #     input_range = (-1.0, 1.0),
    #     weight_range=(-3, 3),
    #     num_interactions=(1, 2),
    #     interaction_weight_range=(-3, 3),
    #     noise_std=0.05,
    # )
    
    # With Hidden Features
    X, hidden_features, y, feature_names, hidden_feature_names, true_linear_weights, true_interaction_weights = generate_synthetic_data_with_hidden_features(
        num_inputs = 10, 
        num_samples = 5000, 
        num_contributing_features = (2, 5), 
        num_hidden_features = (1, 2), 
        input_range = (-1.0, 1.0),
        weight_range = (-3, 3), 
        num_interactions = (1, 2), 
        interaction_weight_range = (-3, 3), 
        noise_std = 0.05, 
        hidden_in_linear = True,           # Should hidden features have a linear impact?
        hidden_in_interactions = True,     # Should hidden features be part of interactions?
        missing_pct = 0.0,                 # 00% of the data will be missing (NaN)
        error_pct = 0.0,                   # 00% of the data will be corrupted
    )

    # Split data here so both OLS and XGBoost train on the exact same subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Traditional Baseline (OLS)
    ols_model, significant_features = perform_traditional_regression(X_train, y_train)

    print("Testing Traditional Baseline (OLS):")
    metrics = evaluate_feature_discovery(significant_features, true_linear_weights, true_interaction_weights)
    
    # 3. Machine Learning Modeling (XGBoost)
    # We pass the pre-split data directly to ensure apples-to-apples comparison
    model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, base_score=np.mean(y_train))
    model.fit(X_train, y_train)
    
    # 4. Evaluation & XGBoost Native Importances
    evaluate_xgb_model(model, X_test, y_test)
    weight_importance, gain_importance, cover_importance = plot_all_xgb_importances(model)
    
    C = -0.5 # Adjust this constant to be more or less strict in filtering features based on standard deviation

    significant_features = weight_importance
    # get only values above the mean
    significant_features = {k: v for k, v in significant_features.items() if v > np.mean(list(significant_features.values())) +C*np.std(list(significant_features.values()))}
    significant_features 

    print("Testing weight_importance:")
    metrics = evaluate_feature_discovery(significant_features, true_linear_weights, true_interaction_weights)

    significant_features = gain_importance
    # get only values above the mean
    significant_features = {k: v for k, v in significant_features.items() if v > np.mean(list(significant_features.values())) +C*np.std(list(significant_features.values()))}
    significant_features 

    print("Testing gain_importance:")
    metrics = evaluate_feature_discovery(significant_features, true_linear_weights, true_interaction_weights)

    significant_features = cover_importance
    # get only values above the mean
    significant_features = {k: v for k, v in significant_features.items() if v > np.mean(list(significant_features.values())) +C*np.std(list(significant_features.values()))}
    significant_features 

    print("Testing cover_importance:")
    metrics = evaluate_feature_discovery(significant_features, true_linear_weights, true_interaction_weights)
    
    # 5. SHAP Explainability & Visualization
    print("Computing SHAP values...")
    shap_values = compute_shap_values(model, X_train, X_test)
    plot_shap_analysis(shap_values, feature_names)
    
    # 6. Deep Dive: SHAP Interaction Analysis
    sorted_interactions = analyze_shap_interactions(model, X_test, feature_names)

    # 7. SHAP Feature Summary
    ranked_features = print_feature_importance(shap_values)
    
    C = -0.5 # Adjust this constant to be more or less strict in filtering features based on standard deviation

    significant_features = ranked_features
    # get only values above the mean
    significant_features = {k: v for k, v in significant_features.items() if v > np.mean(list(significant_features.values())) +C*np.std(list(significant_features.values()))}
    significant_features 

    print("Testing SHAP importance:")
    metrics = evaluate_feature_discovery(significant_features, true_linear_weights, true_interaction_weights)
    
    # ---------------------------------------------------------
    # NEW: SHAP STATISTICAL VALIDATION
    # ---------------------------------------------------------

    # 8. Statistical Significance of SHAP (Slow gold standard permutation test)
    shap_pvalues = compute_shap_pvalues(
        model=model,
        X_train=X_train, 
        y_train=y_train, 
        X_test=X_test, 
        true_shap_values=shap_values, 
        feature_names=feature_names, 
        n_permutations=50 # Increase to 100 or 500 for more robust (but slower) results
    )
    
    p_threshold = 0.05 # Adjust this threshold to be more or less strict in filtering features based on p-values

    significant_features = ranked_features
    # get only values where p-value < p_threshold (e.g., 0.05)
    significant_features = {k: v for k, v in significant_features.items() if shap_pvalues.get(k, 1.0) < p_threshold}
    significant_features 

    print("Testing SHAP importance:")
    metrics = evaluate_feature_discovery(significant_features, true_linear_weights, true_interaction_weights)

    # 9. SHAP Shadow Feature Analysis (Extremely Fast)
    shadow_results = compute_shap_shadow_features(
        model=model,
        X_train=X_train, 
        y_train=y_train, 
        X_test=X_test, 
        feature_names=feature_names
    )

    significant_features = ranked_features
    # get only values where shadow_results indicate significance (e.g., shadow_results[f] == True)
    significant_features = {k: v for k, v in significant_features.items() if shadow_results.get(k, False)}
    significant_features 

    print("Testing SHAP importance:")
    metrics = evaluate_feature_discovery(significant_features, true_linear_weights, true_interaction_weights)

    # 10. SHAP Bootstrapping / Confidence Intervals (Shows Stability)
    bootstrap_results = compute_shap_bootstrapping(
        model=model,
        X_train=X_train, 
        y_train=y_train, 
        X_test=X_test, 
        feature_names=feature_names,
        n_bootstraps=30 # Adjust based on how long you want to wait
    )

if __name__ == "__main__":
    main()