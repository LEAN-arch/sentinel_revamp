# sentinel_project_root/data_processing/helpers.py
#
# PLATINUM STANDARD - Core Data Utilities (V2.1 - Corrected)
# This version corrects the missing 're' module import.

import pandas as pd
import numpy as np
import logging
import json
import re  # <<< THE FIX: Import the 're' module for regular expressions.
from pathlib import Path
from typing import Any, Optional, Union, Dict, List, Type

logger = logging.getLogger(__name__)

# Pre-compiled regex for finding various "Not Available" strings.
# This is more performant than compiling the regex inside a function or loop.
_NA_REGEX_PATTERN = re.compile(
    r'(?i)^\s*(nan|none|n/a|#n/a|na|null|nil|<na>|undefined|unknown|-|)\s*$'
)


def convert_to_numeric(
    data: Any,
    default_value: Any = np.nan,
    target_type: Optional[Type] = None
) -> Any:
    """
    Robustly converts various inputs to a numeric type (scalar or Series),
    correctly handling common "Not Available" string representations.

    Args:
        data: The input data, can be a scalar, list, or pandas Series.
        default_value: The value to use for items that cannot be converted.
        target_type: The desired output type (e.g., int, float). If int,
                     will use pandas' nullable Int64Dtype if NaNs are present.

    Returns:
        The converted data in the same format as the input (scalar or Series).
    """
    is_series = isinstance(data, pd.Series)
    series = data if is_series else pd.Series([data], dtype=object)

    if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
        # Replace common NA strings with a true NaN for consistent conversion.
        series = series.replace(_NA_REGEX_PATTERN, np.nan, regex=True)

    numeric_series = pd.to_numeric(series, errors='coerce')
    if not pd.isna(default_value):
        numeric_series.fillna(default_value, inplace=True)

    # Apply specific integer or float typing if requested.
    if target_type is int and pd.api.types.is_numeric_dtype(numeric_series.dtype):
        # Use pandas' nullable integer type to safely handle potential NaNs.
        if numeric_series.isnull().any():
            numeric_series = numeric_series.astype(pd.Int64Dtype())
        else:
            numeric_series = numeric_series.astype(int)
    elif target_type is float:
        numeric_series = numeric_series.astype(float)

    if is_series:
        return numeric_series
    else:
        return numeric_series.iloc[0] if not numeric_series.empty else default_value


def robust_json_load(
    file_path: Path,
    context: str = "JSON"
) -> Optional[Union[Dict, List]]:
    """
    Loads JSON data from a file with standardized error handling.

    Args:
        file_path: The absolute path to the JSON file.
        context: A string descriptor for logging purposes (e.g., 'GeoJSON').

    Returns:
        The loaded JSON data as a dictionary or list, or None on failure.
    """
    log_ctx = f"{context}({file_path.name})"
    if not file_path.is_file():
        logger.error(f"[{log_ctx}] Load failed: File not found at {file_path}")
        return None
    try:
        with file_path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error(f"[{log_ctx}] File is malformed or has encoding issues: {e}")
        return None
    except Exception as e:
        logger.error(f"[{log_ctx}] An unexpected error occurred while reading file: {e}", exc_info=True)
        return None
