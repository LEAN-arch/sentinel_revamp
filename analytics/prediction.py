# sentinel_project_root/analytics/prediction.py
#
# PLATINUM STANDARD - Predictive Modeling Engine (V2.2 - Re-Architected)
# This module uses a complete, encapsulated scikit-learn Pipeline to generate
# patient risk scores. This architecture guarantees consistency between training
# and inference, eliminating a major class of production ML errors.

import logging
import pandas as pd
from typing import Optional, Any, List

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
_RISK_PIPELINE: Optional[Any] = None
_PIPELINE_LOAD_ATTEMPTED: bool = False


def _get_risk_pipeline() -> Optional[Any]:
    """
    Loads the scikit-learn risk prediction pipeline from disk, caching the result.
    The loaded object is expected to be a `sklearn.pipeline.Pipeline`.
    """
    global _RISK_PIPELINE, _PIPELINE_LOAD_ATTEMPTED

    if _PIPELINE_LOAD_ATTEMPTED:
        return _RISK_PIPELINE

    logger.info("First request for risk model pipeline. Attempting to load from disk...")
    model_path = settings.ml_models.risk_model_path
    _RISK_PIPELINE = load_ml_model(model_path)
    _PIPELINE_LOAD_ATTEMPTED = True

    if _RISK_PIPELINE:
        logger.info(f"Successfully loaded and cached risk prediction pipeline: {type(_RISK_PIPELINE)}")
    else:
        logger.error(
            f"Failed to load risk prediction pipeline from '{model_path}'. "
            "Risk prediction will be disabled."
        )
    return _RISK_PIPELINE


def predict_patient_risk(df_health: pd.DataFrame) -> pd.DataFrame:
    """
    Predicts patient risk using a fully encapsulated scikit-learn pipeline.

    This function enriches the input DataFrame with a new column, 'ai_risk_score'.
    It passes the DataFrame directly to the loaded pipeline, which handles all
    necessary feature selection, scaling, and encoding internally.

    Args:
        df_health: A DataFrame of patient health records. It must contain the raw
                   columns that the ML pipeline was trained on.

    Returns:
        The input DataFrame with the 'ai_risk_score' column added.
    """
    output_col = 'ai_risk_score'
    df_with_predictions = df_health.copy()

    # Default to a null/NA score if prediction fails at any point
    df_with_predictions[output_col] = pd.NA

    if not isinstance(df_health, pd.DataFrame) or df_health.empty:
        logger.warning("predict_patient_risk received an empty DataFrame. Returning as is.")
        return df_with_predictions

    pipeline = _get_risk_pipeline()
    if not pipeline:
        logger.warning("Risk prediction pipeline not available. Cannot generate scores.")
        return df_with_predictions

    # --- Input Validation ---
    # The pipeline object has an attribute that lists the feature names it expects.
    # This is a robust way to check for required columns.
    try:
        expected_features: List[str] = pipeline.feature_names_in_
    except AttributeError:
        # Fallback for older scikit-learn versions or custom pipelines
        logger.warning("Could not determine expected features from pipeline. Performing basic check.")
        expected_features = settings.ml_models.risk_model_features

    missing_features = set(expected_features) - set(df_health.columns)
    if missing_features:
        logger.error(
            f"Cannot make predictions. Input DataFrame is missing required columns "
            f"expected by the pipeline: {sorted(list(missing_features))}"
        )
        return df_with_predictions

    # --- Prediction ---
    try:
        # The magic of the pipeline: pass the DataFrame, and it handles everything.
        # We only need to provide the columns the pipeline was trained on.
        X_predict = df_health[expected_features]
        
        # `predict_proba` returns probabilities for each class: [class_0, class_1].
        # We want the probability of the positive class (high risk), which is at index 1.
        risk_probabilities = pipeline.predict_proba(X_predict)[:, 1]

        # Scale the probability (0.0 to 1.0) to a user-friendly 0-100 score.
        df_with_predictions[output_col] = (risk_probabilities * 100).astype(int)
        logger.info(f"Successfully generated {len(df_with_predictions)} AI risk scores using the pipeline.")

    except ValueError as ve:
        logger.error(f"A ValueError occurred during prediction, often due to data type "
                     f"or format issues that the pipeline could not handle. Error: {ve}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred during risk score prediction: {e}", exc_info=True)

    return df_with_predictions
