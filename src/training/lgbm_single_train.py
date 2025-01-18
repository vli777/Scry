import os
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import shap
import joblib

from ..config_loader import config


def train_lgbm(X_train, y_train, X_test, y_test, params=None, save_path=None):
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

    return model, rmse


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


if __name__ == "__main__":
    # Load processed data
    df = pd.read_parquet(config.processed_file)

    # Extract features and target
    X = df[config.continuous_columns]
    y = df["close"]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Generate dynamic model path
    single_model_path = os.path.join(
        config.model_dir, f"{config.model_type}_{config.symbol}_single.pkl"
    )

    # Load or train the initial model
    try:
        lgbm_model = joblib.load(single_model_path)
        print(f"Loaded the initial model from {single_model_path}.")
    except FileNotFoundError:
        print("Initial model not found. Training a new one.")
        lgbm_model = train_lgbm(
            X_train, y_train, X_test, y_test, save_path=single_model_path
        )

    # Compute SHAP values and plot summary
    print("Generating SHAP values for feature importance...")
    explainer = shap.Explainer(lgbm_model, X_train)
    shap_values = explainer(X_test, check_additivity=False)
    shap.summary_plot(shap_values, X_test)

    # Remove low-importance features and retrain
    reduced_model_path = os.path.join(
        config.model_dir, f"{config.model_type}_{config.symbol}_single_reduced.pkl"
    )
    lgbm_model_reduced, X_train_reduced, X_test_reduced = (
        remove_low_importance_features(
            lgbm_model, X_train, X_test, y_train, y_test, save_path=reduced_model_path
        )
    )
    print(f"Reduced model saved to {reduced_model_path}.")
