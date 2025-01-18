import os
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import shap
import joblib

from ..config_loader import config

from preprocessing.process_data import continuous_columns


def train_lgbm(X_train, y_train, X_test, y_test, params=None, save_path=None):
    if params is None:
        params = {
            "random_state": 42,
            "n_estimators": 100,
            "learning_rate": 0.1,
            "device": "cpu",
        }

    # Train LightGBM model
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"LightGBM RMSE: {rmse:.4f}")
    r2 = r2_score(y_test, y_pred)
    print(f"R²: {r2:.4f}")

    # Save the model
    if save_path:
        joblib.dump(model, save_path)
        print(f"Model saved to {save_path}")

    return model, rmse


def remove_low_importance_features(
    model, X_train, X_test, y_train, y_test, threshold=0.01, save_path=None
):
    """
    Removes features with SHAP importance values below the specified threshold,
    retrains the model on the reduced feature set, and optionally saves it.
    """
    # Use the new SHAP API for explanation
    explainer = shap.Explainer(model, X_train)
    explanation = explainer(X_test, check_additivity=False)  # Compute SHAP values

    # Calculate mean absolute SHAP values for each feature
    shap_importances = explanation.values.mean(axis=0)
    feature_names = X_train.columns

    # Identify low-importance features
    low_importance_features = [
        feature_names[i]
        for i, importance in enumerate(shap_importances)
        if importance < threshold
    ]

    # Ensure at least one feature remains
    if len(low_importance_features) == len(X_train.columns):
        print(
            "Warning: All features were marked as low importance. Retaining top features."
        )
        low_importance_features = low_importance_features[
            :-1
        ]  # Retain at least one feature

    print(f"Removing features: {low_importance_features}")

    # Drop low-importance features from training and test sets
    X_train_reduced = X_train.drop(columns=low_importance_features)
    X_test_reduced = X_test.drop(columns=low_importance_features)

    # Verify the reduced dataset is valid
    if X_train_reduced.shape[1] == 0:
        raise ValueError(
            "No features remain after feature removal. Adjust the threshold."
        )

    # Retrain the model on the reduced feature set
    reduced_model, reduced_rmse = train_lgbm(
        X_train_reduced, y_train, X_test_reduced, y_test, save_path=save_path
    )

    return reduced_model, X_train_reduced, X_test_reduced


if __name__ == "__main__":
    # Load processed data
    df = pd.read_parquet(config.processed_file)

    # Split the dataset
    X = df[continuous_columns]  # Feature set
    y = df["close"]  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Check if the initial model is already saved
    try:
        lgbm_model = joblib.load(config.model_file)
        print(f"Loaded the initial model from {config.model_file}.")
    except FileNotFoundError:
        print("Initial model not found. Training a new one.")
        lgbm_model, initial_rmse = train_lgbm(
            X_train, y_train, X_test, y_test, save_path=config.model_file
        )
        print(f"Initial model trained. RMSE: {initial_rmse:.4f}")

    # Use SHAP Explainer for the loaded model
    print("Generating SHAP values for feature importance...")
    explainer = shap.Explainer(lgbm_model, X_train)

    # Check for missing values in the test set
    print("Missing values in X_test:", X_test.isnull().sum())
    X_test = X_test.fillna(X_test.mean())

    shap_values = explainer(X_test, check_additivity=False)
    shap.summary_plot(shap_values, X_test)

    # Remove low-importance features and retrain the reduced model
    reduced_model_path = os.path.join(
        config.model_dir, f"{config.model_type}_{config.symbol}_reduced.pkl"
    )
    lgbm_model_reduced, X_train_reduced, X_test_reduced = (
        remove_low_importance_features(
            lgbm_model,
            X_train,
            X_test,
            y_train,
            y_test,
            save_path=reduced_model_path,
        )
    )

    print(f"Reduced model saved to {reduced_model_path}.")
