import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import shap
import joblib

from process_data import continuous_columns

# Load processed data
FILE_PATH = "data/processed/SPY_5min_processed.parquet"
df = pd.read_parquet(FILE_PATH)

# Split the dataset
X = df[continuous_columns]  # Feature set
y = df["close"]  # Target variable
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


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
    # Check if the initial model is already saved
    MODEL_PATH = "models/lgbm_model.pkl"
    try:
        lgbm_model = joblib.load(MODEL_PATH)
        print("Loaded the initial model from disk.")
    except FileNotFoundError:
        print("Initial model not found. Training a new one.")
        lgbm_model, initial_rmse = train_lgbm(
            X_train, y_train, X_test, y_test, save_path=MODEL_PATH
        )

    # Use TreeExplainer for the loaded model
    explainer = shap.Explainer(
        lgbm_model, X_train
    )  # Assuming X_train is available in your scope
    # Check for missing values in the test set
    print(X_test.isnull().sum())
    X_test.fillna(X_test.mean(), inplace=True)

    shap_values = explainer(
        X_test, check_additivity=False
    )  # Generate SHAP values for the test set
    shap.summary_plot(shap_values)

    # Remove low-importance features and retrain the reduced model
    lgbm_model_reduced, X_train_reduced, X_test_reduced = (
        remove_low_importance_features(
            lgbm_model,
            X_train,
            X_test,
            y_train,
            y_test,
            save_path="models/lgbm_model_reduced.pkl",
        )
    )
