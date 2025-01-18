import os
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import shap
import joblib

from src.config_loader import config


def train_lgbm(
    X_train, y_train, X_test, y_test, params=None, save_path=None, shap_enabled=True
):
    """
    Train a single-step LightGBM model and optionally compute SHAP values.

    :param X_train: Training feature data.
    :param y_train: Training target data.
    :param X_test: Test feature data.
    :param y_test: Test target data.
    :param save_path: Path to save the model.
    :param shap_enabled: Whether to compute SHAP values.
    """
    if params is None:
        params = {
            "random_state": 42,
            "n_estimators": 100,
            "learning_rate": 0.1,
        }

    # Train LightGBM model
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"LightGBM RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

    # Save the model
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(model, save_path)
        print(f"Model saved to {save_path}")

    # Optionally compute SHAP values
    if shap_enabled:
        compute_shap_values(model, X_train, X_test)

    return model, rmse


def compute_shap_values(model, X_train, X_test):
    """
    Compute and display SHAP values for the given model and datasets.
    """
    print("Generating SHAP values for feature importance...")
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test, check_additivity=False)
    shap.summary_plot(shap_values, X_test)


def remove_low_importance_features(
    model, X_train, X_test, y_train, y_test, threshold=0.01, save_path=None
):
    explainer = shap.Explainer(model, X_train)
    explanation = explainer(X_test, check_additivity=False)
    shap_importances = explanation.values.mean(axis=0)
    feature_names = X_train.columns

    low_importance_features = [
        feature_names[i]
        for i, importance in enumerate(shap_importances)
        if importance < threshold
    ]

    if len(X_train.columns) - len(low_importance_features) < 5:
        low_importance_features = low_importance_features[: len(X_train.columns) - 5]

    print(f"Removing features: {low_importance_features}")
    X_train_reduced = X_train.drop(columns=low_importance_features)
    X_test_reduced = X_test.drop(columns=low_importance_features)

    reduced_model, reduced_rmse = train_lgbm(
        X_train_reduced, y_train, X_test_reduced, y_test, save_path=save_path
    )

    return reduced_model, X_train_reduced, X_test_reduced
