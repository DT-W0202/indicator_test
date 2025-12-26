"""
Factor Transformation Pipeline
==============================

Provides transformation layers for factor processing:
- Layer 1: Winsorize (outlier handling)
- Layer 2: MAD Z-Score (robust standardization)
- Layer 3a: Rolling Percentile
- Layer 3b: Z to CDF (probability transformation)

Also includes factor decomposition functions.
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from typing import Dict, Any, Tuple, Optional


class TransformPipeline:
    """Factor Transformation Pipeline"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize transformation pipeline.

        Args:
            config: Configuration dictionary with keys:
                - winsorize_limits: (lower, upper) quantiles, e.g., (0.01, 0.99)
                - zscore_window: Rolling window for Z-score (in periods)
                - percentile_window: Rolling window for percentile (in periods)
        """
        self.config = config or {
            'winsorize_limits': (0.01, 0.99),
            'zscore_window': 120,  # 10Y for monthly data
            'percentile_window': 120  # 10Y for monthly data
        }

    # ========== Layer 1: Winsorize ==========

    def winsorize(self,
                  series: pd.Series,
                  limits: Tuple[float, float] = None) -> pd.Series:
        """
        Winsorize outliers by clipping to quantile bounds.

        Args:
            series: Input series
            limits: (lower_quantile, upper_quantile), e.g., (0.01, 0.99)

        Returns:
            Winsorized series
        """
        limits = limits or self.config['winsorize_limits']
        lower = series.quantile(limits[0])
        upper = series.quantile(limits[1])
        return series.clip(lower=lower, upper=upper)

    # ========== Layer 2: MAD Z-Score ==========

    def rolling_mad_zscore(self,
                           series: pd.Series,
                           window: int = None) -> pd.Series:
        """
        Calculate rolling MAD-based Z-Score.

        MAD (Median Absolute Deviation) is more robust than standard deviation.
        Z = (x - median) / (MAD * 1.4826)

        The scaling factor 1.4826 makes MAD consistent with standard deviation
        for normally distributed data.

        Args:
            series: Input series
            window: Rolling window size (in periods)

        Returns:
            Z-Score series
        """
        window = window or self.config['zscore_window']
        min_periods = max(1, window // 4)

        result = pd.Series(index=series.index, dtype=float)

        for i in range(len(series)):
            if i < min_periods - 1:
                continue

            start_idx = max(0, i - window + 1)
            historical = series.iloc[start_idx:i + 1]

            median = historical.median()
            mad = np.median(np.abs(historical - median))

            if mad > 0:
                result.iloc[i] = (series.iloc[i] - median) / (mad * 1.4826)
            else:
                # If MAD is 0, use a small epsilon
                result.iloc[i] = 0

        return result

    # ========== Layer 3a: Rolling Percentile ==========

    def rolling_percentile(self,
                           series: pd.Series,
                           window: int = None) -> pd.Series:
        """
        Calculate rolling percentile rank.

        Computes the percentile rank of current value within historical window.

        Args:
            series: Input series
            window: Rolling window size (in periods)

        Returns:
            Percentile series (0-100)
        """
        window = window or self.config['percentile_window']

        result = pd.Series(index=series.index, dtype=float)

        for i in range(len(series)):
            start_idx = max(0, i - window + 1)
            historical = series.iloc[start_idx:i + 1]

            if len(historical) > 0:
                rank = (historical < series.iloc[i]).sum()
                result.iloc[i] = rank / len(historical) * 100

        return result

    # ========== Layer 3b: Z to CDF (Probability) ==========

    def zscore_to_probability(self, zscore: pd.Series) -> pd.Series:
        """
        Convert Z-Score to probability using standard normal CDF.

        Maps Z-Score to [0, 100] range using cumulative normal distribution.

        Args:
            zscore: Z-Score series

        Returns:
            Probability series (0-100)
        """
        return pd.Series(norm.cdf(zscore) * 100, index=zscore.index)

    # ========== Complete Pipeline ==========

    def transform(self,
                  series: pd.Series,
                  output_type: str = 'percentile') -> pd.Series:
        """
        Apply complete transformation pipeline.

        Args:
            series: Raw factor series
            output_type: Output type
                - 'percentile': Winsorize -> Rolling Percentile
                - 'probability': Winsorize -> MAD Z-Score -> CDF
                - 'zscore': Winsorize -> MAD Z-Score
                - 'winsorized': Winsorize only

        Returns:
            Transformed series
        """
        # Layer 1: Winsorize
        winsorized = self.winsorize(series)

        if output_type == 'winsorized':
            return winsorized

        if output_type == 'zscore':
            return self.rolling_mad_zscore(winsorized)

        if output_type == 'probability':
            zscore = self.rolling_mad_zscore(winsorized)
            return self.zscore_to_probability(zscore)

        # Default: percentile
        return self.rolling_percentile(winsorized)

    def transform_all(self, series: pd.Series) -> Dict[str, pd.Series]:
        """
        Apply all transformations and return dictionary of results.

        Args:
            series: Raw factor series

        Returns:
            Dictionary with keys: 'raw', 'winsorized', 'zscore',
                                 'percentile', 'probability'
        """
        winsorized = self.winsorize(series)
        zscore = self.rolling_mad_zscore(winsorized)

        return {
            'raw': series,
            'winsorized': winsorized,
            'zscore': zscore,
            'percentile': self.rolling_percentile(winsorized),
            'probability': self.zscore_to_probability(zscore)
        }


# ========== Factor Decomposition ==========

def compute_factor_change(factor_level: pd.Series) -> pd.Series:
    """
    Compute factor change (first difference).

    Delta_F_t = F_t - F_{t-1}

    For monthly data, this represents month-over-month change.

    Args:
        factor_level: Factor level series

    Returns:
        Factor change series
    """
    return factor_level.diff()


def compute_factor_acceleration(factor_level: pd.Series) -> pd.Series:
    """
    Compute factor acceleration (second difference).

    Accel_F_t = Delta_F_t - Delta_F_{t-1}

    Args:
        factor_level: Factor level series

    Returns:
        Factor acceleration series
    """
    return factor_level.diff().diff()


def compute_factor_velocity_score(factor_level: pd.Series,
                                  window: int = 36) -> pd.Series:
    """
    Compute velocity score (percentile rank of changes).

    Args:
        factor_level: Factor level series
        window: Rolling window for percentile calculation

    Returns:
        Velocity score series (0-100)
    """
    change = factor_level.diff()

    result = pd.Series(index=change.index, dtype=float)

    for i in range(len(change)):
        if pd.isna(change.iloc[i]):
            continue

        start_idx = max(0, i - window + 1)
        historical = change.iloc[start_idx:i + 1].dropna()

        if len(historical) > 0:
            rank = (historical < change.iloc[i]).sum()
            result.iloc[i] = rank / len(historical) * 100

    return result
