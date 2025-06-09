# sentinel_project_root/data_processing/pipeline.py
#
# PLATINUM STANDARD - Fluent Data Processing Pipeline (V2.1 - Re-validated)
# This module provides a robust, chainable class for applying a sequence of
# data cleaning and preparation steps, now confirmed functional.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List
from collections import Counter

# --- Core Application Imports (Now Correctly Importing from fixed helper) ---
try:
    from .helpers import convert_to_numeric
except ImportError as e:
    # This should not happen with the corrected file structure
    logging.basicConfig(level="CRITICAL")
    logging.critical(f"FATAL ERROR in pipeline.py: could not import helpers. {e}", exc_info=True)
    raise

logger = logging.getLogger(__name__)

# This regex is now safely imported via helpers.py, but for clarity, if it were
# defined here, it would need `import re`. The chosen architecture keeps it in helpers.
from .helpers import _NA_REGEX_PATTERN


class DataPipeline:
    """
    A fluent interface for applying a sequence of data processing operations.
    Enables expressive, readable, and chainable cleaning pipelines.
    """
    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("DataPipeline must be initialized with a pandas DataFrame.")
        self._df = df.copy()

    def get_df(self) -> pd.DataFrame:
        """Returns the processed DataFrame."""
        return self._df

    def clean_column_names(self) -> 'DataPipeline':
        """Standardizes DataFrame column names for consistency and usability."""
        if self._df.empty:
            return self
        try:
            new_cols = (
                self._df.columns.astype(str)
                .str.lower().str.strip()
                .str.replace(r'[^0-9a-z_]+', '_', regex=True)
                .str.replace(r'_{2,}', '_', regex=True).str.strip('_')
            )
            new_cols = [f"unnamed_col_{i}" if not name else name for i, name in enumerate(new_cols)]

            counts = Counter(new_cols)
            if max(counts.values(), default=0) > 1:
                seen = Counter()
                final_cols = []
                for col_name in new_cols:
                    if counts[col_name] > 1:
                        suffix = seen[col_name]
                        seen[col_name] += 1
                        final_cols.append(f"{col_name}_{suffix}")
                    else:
                        final_cols.append(col_name)
                self._df.columns = final_cols
            else:
                self._df.columns = new_cols
        except Exception as e:
            logger.error(f"Error standardizing column names: {e}", exc_info=True)
        return self

    def standardize_missing_values(self, column_defaults: Dict[str, Any]) -> 'DataPipeline':
        """Replaces various 'Not Available' formats and fills with provided defaults."""
        if not column_defaults:
            return self
        for col, default_val in column_defaults.items():
            if col in self._df.columns:
                series = self._df[col]
                if isinstance(default_val, (int, float, np.number)):
                    target_type = int if isinstance(default_val, int) else float
                    self._df[col] = convert_to_numeric(
                        series, default_value=default_val, target_type=target_type
                    )
                else:
                    series_obj = series.astype(object).replace(_NA_REGEX_PATTERN, np.nan, regex=True)
                    self._df[col] = series_obj.fillna(str(default_val))
        return self

    def convert_date_columns(self, date_columns: List[str], errors: str = 'coerce') -> 'DataPipeline':
        """Converts specified columns to datetime objects."""
        if not date_columns:
            return self
        for col in date_columns:
            if col in self._df.columns:
                self._df[col] = pd.to_datetime(self._df[col], errors=errors)
            else:
                logger.warning(f"Date conversion skipped: Column '{col}' not found.")
        return self

    def rename_columns(self, rename_map: Dict[str, str]) -> 'DataPipeline':
        """Renames columns based on a provided dictionary."""
        if not rename_map:
            return self
        self._df.rename(columns=rename_map, errors="ignore", inplace=True)
        return self
