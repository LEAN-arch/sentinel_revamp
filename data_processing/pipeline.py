# sentinel_project_root/data_processing/pipeline.py
#
# PLATINUM STANDARD - Fluent Data Processing Pipeline
# This module provides a robust, chainable class for applying a sequence of
# data cleaning and preparation steps to a pandas DataFrame.

import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, Any, List
from collections import Counter

from .helpers import convert_to_numeric

logger = logging.getLogger(__name__)

# Pre-compiled regex for finding various "Not Available" strings.
# This is more performant than compiling the regex inside a loop.
NA_REGEX_PATTERN = re.compile(
    r'(?i)^\s*(nan|none|n/a|#n/a|na|null|nil|<na>|undefined|unknown|-|)\s*$'
)


class DataPipeline:
    """
    A fluent interface for applying a sequence of data processing operations.

    Enables expressive, readable, and chainable cleaning pipelines that promote
    reusability and consistency across the application.

    Usage:
        processed_df = (
            DataPipeline(raw_df)
            .clean_column_names()
            .convert_date_columns(['event_date', 'report_date'])
            .standardize_missing_values({'age': 0, 'notes': 'Unknown'})
            .get_df()
        )
    """
    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("DataPipeline must be initialized with a pandas DataFrame.")
        # Work on a copy to avoid side effects on the original DataFrame.
        self._df = df.copy()

    def get_df(self) -> pd.DataFrame:
        """Returns the processed DataFrame at the end of the pipeline."""
        return self._df

    def clean_column_names(self) -> 'DataPipeline':
        """
        Standardizes DataFrame column names for consistency and usability.
        Converts to lowercase, replaces special characters with underscores,
        and handles duplicates by appending a suffix.
        """
        if self._df.empty:
            return self

        try:
            # Vectorized string operations for performance.
            new_cols = (
                self._df.columns.astype(str)
                .str.lower()
                .str.strip()
                .str.replace(r'[^0-9a-z_]+', '_', regex=True)
                .str.replace(r'_{2,}', '_', regex=True)
                .str.strip('_')
            )

            # Handle potentially empty column names after cleaning.
            new_cols = [f"unnamed_col_{i}" if not name else name for i, name in enumerate(new_cols)]

            # High-performance de-duplication for column names.
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
        """
        Replaces various "Not Available" formats with np.nan and then fills
        with provided default values, intelligently inferring data types.
        """
        if not column_defaults:
            return self

        for col, default_val in column_defaults.items():
            if col in self._df.columns:
                series = self._df[col]
                # Intelligently handle type based on default value.
                if isinstance(default_val, (int, float, np.number)):
                    target_type = int if isinstance(default_val, int) else float
                    self._df[col] = convert_to_numeric(
                        series, default_value=default_val, target_type=target_type
                    )
                else:
                    # Treat as a string/object column.
                    series_obj = series.astype(object).replace(NA_REGEX_PATTERN, np.nan, regex=True)
                    self._df[col] = series_obj.fillna(str(default_val))
        return self

    def convert_date_columns(self, date_columns: List[str], errors: str = 'coerce') -> 'DataPipeline':
        """
        Converts specified columns to datetime objects.

        Args:
            date_columns: A list of column names to convert.
            errors: Behavior for parsing errors ('coerce' sets invalid to NaT).
        """
        if not date_columns:
            return self

        for col in date_columns:
            if col in self._df.columns:
                # Use pd.to_datetime which is highly optimized for this task.
                self._df[col] = pd.to_datetime(self._df[col], errors=errors)
            else:
                logger.warning(f"Date conversion skipped: Column '{col}' not found in DataFrame.")
        return self

    def rename_columns(self, rename_map: Dict[str, str]) -> 'DataPipeline':
        """Renames columns based on a provided dictionary."""
        if not rename_map:
            return self

        # The 'errors' parameter prevents crashes if a key in the map doesn't exist.
        self._df.rename(columns=rename_map, errors="ignore", inplace=True)
        return self
