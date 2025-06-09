# sentinel_project_root/analytics/prediction.py
#
# PLATINUM STANDARD - Predictive Modeling Engine
# This module uses a trained machine learning model to generate patient risk scores,
# replacing the previous brittle, rule-based system.

import logging
import pandas as pd
from typing import Optional, Any

# --- Core Application Imports ---
try:
    from config.settings import settings
    from data_processing.loaders import load_ml_model
except ImportError as e:
    logging.basicConfig(level="CRITICAL")
    logging.critical(f"FATAL ERROR in prediction.py: A core dependency is missing. {e}", exc_info=True)
    raise

logger = logging.getLogger(__name__)

# --- Module-level Model Cache ---
# This lazy-loads the model once and caches it in memory for performance.
_RISK_MODEL: Optional[Any] = None
_MODEL_LOAD_ATTEMPTED: bool = False


def _get_risk_model() -> Optional[Any]:
    """
    Loads the risk prediction model from disk, caching the result.
    This function is for internal use by `predict_patient_risk`.
    """
    global _RISK_MODEL, _MODEL_LOAD_ATTEMPTED

    if _MODEL_LOAD_ATTEMPTED:
        return _RISK_MODEL

    logger.info("First request for risk model. Attempting to load from disk...")
    model_path = settings.ml_models.risk_model_path
    _RISK_MODEL = load_ml_model(model_path)
    _MODEL_LOAD_ATTEMPTED = True

    if _RISK_MODEL:
        logger.info(f"Successfully loaded and cached risk model: {type(_RISK_MODEL)}")
    else:
        logger.error(
            f"Failed to load risk model from '{model_path}'. "
            "Risk prediction will be disabled."
        )
    return _RISK_MODEL


def predict_patient_risk(df_health: pd.DataFrame) -> pd.DataFrame:
    """
    Predicts a patient's risk of an adverse outcome using a trained ML model.

    This function enriches the input DataFrame with a new column, 'ai_risk_score',
    which is a numeric score from 0 to 100.

    Args:
        df_health: A DataFrame containing patient health records, which must
                   have been processed by the `enrich_health_records_with_features`
                   function to ensure required features are present.

    Returns:
        The input DataFrame with the 'ai_risk_score' column added.
    """
    output_col = 'ai_risk_score'
    df_with_predictions = df_health.copy()

    if not isinstance(df_health, pd.DataFrame) or df_health.empty:
        logger.warning("predict_patient_risk received an empty DataFrame. Returning as is.")
        df_with_predictions[output_col] = pd.NA
        return df_with_predictions

    model = _get_risk_model()
    if not model:
        logger.warning("Risk model not available. Assigning NA to risk scores.")
        df_with_predictions[output_col] = pd.NA
        return df_with_predictions

    # --- Feature Preparation ---
    # Ensure the prediction features are present and in the correct order.
    features = settings.ml_models.risk_model_features
    missing_features = set(features) - set(df_health.columns)

    if missing_features:
        logger.error(
            f"Cannot make predictions. The following required features are "
            f"missing from the input DataFrame: {sorted(list(missing_features))}"
        )
        df_with_predictions[output_col] = pd.NA
        return df_with_predictions

    X_predict = df_health[features]

    # --- Prediction ---
    try:
        # `predict_proba` returns probabilities for each class: [class_0, class_1].
        # We want the probability of the positive class (high risk), which is at index 1.
        risk_probabilities = model.predict_proba(X_predict)[:, 1]

        # Scale the probability (0.0 to 1.0) to a user-friendly 0-100 score.
        df_with_predictions[output_col] = (risk_probabilities * 100).astype(int)
        logger.info(f"Successfully generated {len(df_with_predictions)} AI risk scores.")

    except Exception as e:
        logger.error(f"An error occurred during risk score prediction: {e}", exc_info=True)
        df_with_predictions[output_col] = pd.NA

    return df_with_predictions
