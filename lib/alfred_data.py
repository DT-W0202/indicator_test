"""
ALFRED Point-in-Time Data Loading Module
========================================

Provides functions to load FRED data using ALFRED (Archival FRED) API
for point-in-time backtesting without look-ahead bias.

Key Features:
- Get all historical releases/revisions of a series
- Build month-end as-of sequences
- Calculate YoY within same vintage (critical for PIT)
"""

import pandas as pd
import numpy as np
from fredapi import Fred
from typing import Dict, Optional
from datetime import datetime


class ALFREDDataLoader:
    """ALFRED Point-in-Time Data Loader"""

    def __init__(self, api_key: str):
        """
        Initialize ALFRED data loader.

        Args:
            api_key: FRED API key
        """
        self.fred = Fred(api_key=api_key)
        self._cache: Dict[str, pd.DataFrame] = {}

    def get_all_releases(self,
                         series_id: str,
                         realtime_start: str = None,
                         realtime_end: str = None) -> pd.DataFrame:
        """
        Get all historical releases of a series from ALFRED.

        Args:
            series_id: FRED series ID (e.g., 'GDP', 'BCNSDODNS')
            realtime_start: Start date for realtime filter (YYYY-MM-DD)
            realtime_end: End date for realtime filter (YYYY-MM-DD)

        Returns:
            DataFrame with columns:
            - date: Observation date (the period the data represents)
            - realtime_start: Release/revision date (when this value became known)
            - value: The observation value
        """
        cache_key = f"{series_id}_{realtime_start}_{realtime_end}"

        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        try:
            df = self.fred.get_series_all_releases(
                series_id,
                realtime_start=realtime_start,
                realtime_end=realtime_end
            )

            # Ensure proper column types
            df = df.reset_index(drop=True)
            df['date'] = pd.to_datetime(df['date'])
            df['realtime_start'] = pd.to_datetime(df['realtime_start'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')

            self._cache[cache_key] = df
            return df.copy()

        except Exception as e:
            print(f"[ALFRED] Error fetching {series_id}: {e}")
            return pd.DataFrame(columns=['date', 'realtime_start', 'value'])

    def get_series_as_of(self, series_id: str, as_of_date: str) -> pd.Series:
        """
        Get series values as known at a specific date.

        Args:
            series_id: FRED series ID
            as_of_date: The as-of date (YYYY-MM-DD format)

        Returns:
            Series indexed by observation date, containing the latest
            values known as of the specified date.
        """
        df = self.fred.get_series_as_of_date(series_id, as_of_date)

        if len(df) == 0:
            return pd.Series(dtype=float)

        # For each observation date, take the most recent revision
        df = df.reset_index(drop=True)
        df['date'] = pd.to_datetime(df['date'])
        latest = df.groupby('date')['value'].last()

        return latest.sort_index()

    def build_monthly_pit_series(self,
                                  series_id: str,
                                  start_date: str,
                                  end_date: str,
                                  verbose: bool = True) -> pd.DataFrame:
        """
        Build month-end point-in-time series.

        For each month-end date, get the data as it was known at that time.
        Quarterly data naturally forms a step function when viewed monthly.

        Args:
            series_id: FRED series ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            verbose: Print progress

        Returns:
            DataFrame with columns:
            - as_of_date: Month-end date (when the data was observed)
            - date: Observation date (the period the data represents)
            - value: The value known at as_of_date
        """
        # Generate month-end dates
        month_ends = pd.date_range(start=start_date, end=end_date, freq='ME')

        if verbose:
            print(f"  Building PIT series for {series_id}...")
            print(f"  Date range: {start_date} to {end_date} ({len(month_ends)} months)")

        # Get all historical releases at once (more efficient)
        all_releases = self.get_all_releases(series_id)

        if len(all_releases) == 0:
            print(f"  WARNING: No data found for {series_id}")
            return pd.DataFrame(columns=['as_of_date', 'date', 'value'])

        if verbose:
            print(f"  Total releases: {len(all_releases)}")

        pit_data = []

        for i, as_of_date in enumerate(month_ends):
            # Filter to releases available at this as_of_date
            mask = all_releases['realtime_start'] <= as_of_date
            available = all_releases[mask].copy()

            if len(available) == 0:
                continue

            # For each observation date, take the latest revision
            latest_by_obs = available.sort_values('realtime_start').groupby('date').last()
            latest_by_obs = latest_by_obs.reset_index()
            latest_by_obs['as_of_date'] = as_of_date

            pit_data.append(latest_by_obs[['as_of_date', 'date', 'value']])

        if len(pit_data) == 0:
            return pd.DataFrame(columns=['as_of_date', 'date', 'value'])

        result = pd.concat(pit_data, ignore_index=True)

        if verbose:
            print(f"  PIT records: {len(result)}")

        return result

    def clear_cache(self):
        """Clear the internal cache."""
        self._cache.clear()


def build_pit_factor_series(debt_pit: pd.DataFrame,
                            gdp_pit: pd.DataFrame,
                            verbose: bool = True) -> pd.DataFrame:
    """
    Build point-in-time factor series: F = YoY%(Debt) - YoY%(GDP)

    CRITICAL: YoY is calculated within the same vintage (as_of_date),
    meaning both t and t-4Q values come from the same as-of perspective.

    Args:
        debt_pit: PIT DataFrame for Debt (from build_monthly_pit_series)
        gdp_pit: PIT DataFrame for GDP (from build_monthly_pit_series)
        verbose: Print progress

    Returns:
        DataFrame with columns:
        - as_of_date: Month-end date (factor's effective date)
        - obs_date: Latest observation date used
        - debt_value: Debt value at obs_date
        - gdp_value: GDP value at obs_date
        - debt_yoy: Debt YoY%
        - gdp_yoy: GDP YoY%
        - factor_level: F = debt_yoy - gdp_yoy
    """
    if verbose:
        print("  Building PIT factor series...")

    # Get all unique as-of dates (intersection)
    debt_dates = set(debt_pit['as_of_date'].unique())
    gdp_dates = set(gdp_pit['as_of_date'].unique())
    common_dates = sorted(debt_dates & gdp_dates)

    if verbose:
        print(f"  Common as-of dates: {len(common_dates)}")

    results = []

    for as_of_date in common_dates:
        # Get snapshot for this as_of_date
        debt_snap = debt_pit[debt_pit['as_of_date'] == as_of_date].copy()
        gdp_snap = gdp_pit[gdp_pit['as_of_date'] == as_of_date].copy()

        # Set index to observation date
        debt_snap = debt_snap.set_index('date').sort_index()
        gdp_snap = gdp_snap.set_index('date').sort_index()

        # Get common observation dates
        common_obs = debt_snap.index.intersection(gdp_snap.index)

        if len(common_obs) < 5:
            # Need at least 5 quarters for YoY calculation
            continue

        # Calculate YoY within this vintage
        # For quarterly data, shift 4 periods = 1 year
        debt_yoy = (debt_snap['value'] / debt_snap['value'].shift(4) - 1) * 100
        gdp_yoy = (gdp_snap['value'] / gdp_snap['value'].shift(4) - 1) * 100

        # Get the latest valid observation
        for obs_date in reversed(common_obs):
            debt_yoy_val = debt_yoy.get(obs_date)
            gdp_yoy_val = gdp_yoy.get(obs_date)

            if pd.notna(debt_yoy_val) and pd.notna(gdp_yoy_val):
                factor = debt_yoy_val - gdp_yoy_val

                results.append({
                    'as_of_date': as_of_date,
                    'obs_date': obs_date,
                    'debt_value': debt_snap.loc[obs_date, 'value'],
                    'gdp_value': gdp_snap.loc[obs_date, 'value'],
                    'debt_yoy': debt_yoy_val,
                    'gdp_yoy': gdp_yoy_val,
                    'factor_level': factor
                })
                break

    result_df = pd.DataFrame(results)

    if verbose and len(result_df) > 0:
        print(f"  Factor records: {len(result_df)}")
        print(f"  Date range: {result_df['as_of_date'].min()} to {result_df['as_of_date'].max()}")

    return result_df


def get_latest_series(fred: Fred, series_id: str,
                      start_date: str = None) -> pd.Series:
    """
    Get the latest (non-PIT) version of a series for comparison.

    Args:
        fred: Fred instance
        series_id: FRED series ID
        start_date: Optional start date

    Returns:
        Series with observation dates as index
    """
    try:
        data = fred.get_series(series_id, observation_start=start_date)
        return data
    except Exception as e:
        print(f"[FRED] Error fetching {series_id}: {e}")
        return pd.Series(dtype=float)


# ============== 发布滞后配置 ==============
# 基于实际观测的发布滞后天数（从季度结束到首次发布）
RELEASE_LAG_DAYS = {
    'BCNSDODNS': 72,  # ~2.4 个月
    'GDP': 30,        # ~1 个月
    # 可以添加更多序列的滞后配置
}

DEFAULT_RELEASE_LAG = 60  # 默认滞后天数


def build_simulated_pit_series(fred: Fred,
                                series_id: str,
                                start_date: str,
                                end_date: str,
                                release_lag_days: int = None,
                                verbose: bool = True) -> pd.DataFrame:
    """
    使用 FRED 最新数据 + 发布滞后模拟构建 Point-in-Time 序列。

    适用于 ALFRED 数据不可用的早期时间段（如 2010 年前）。

    原理：
    - 从 FRED 获取最新数据（假设这是"最终修订值"）
    - 根据发布滞后，模拟每个月末能看到哪些季度的数据

    Args:
        fred: Fred instance
        series_id: FRED series ID
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        release_lag_days: 发布滞后天数，None 则使用默认配置
        verbose: Print progress

    Returns:
        DataFrame with columns: ['as_of_date', 'date', 'value']
        与 build_monthly_pit_series 格式相同
    """
    if release_lag_days is None:
        release_lag_days = RELEASE_LAG_DAYS.get(series_id, DEFAULT_RELEASE_LAG)

    if verbose:
        print(f"  Building simulated PIT series for {series_id}...")
        print(f"  Using release lag: {release_lag_days} days ({release_lag_days/30:.1f} months)")

    # 获取 FRED 最新数据
    try:
        latest_data = fred.get_series(series_id, observation_start=start_date)
    except Exception as e:
        print(f"  ERROR: Failed to fetch {series_id}: {e}")
        return pd.DataFrame(columns=['as_of_date', 'date', 'value'])

    if len(latest_data) == 0:
        print(f"  WARNING: No data found for {series_id}")
        return pd.DataFrame(columns=['as_of_date', 'date', 'value'])

    # 转换为 DataFrame
    latest_df = pd.DataFrame({
        'date': latest_data.index,
        'value': latest_data.values
    })
    latest_df['date'] = pd.to_datetime(latest_df['date'])

    # 计算每个观测日期的"可用日期"（季度结束 + 发布滞后）
    # 季度数据的 date 是季度第一天，需要先找到季度结束日
    latest_df['quarter_end'] = latest_df['date'] + pd.offsets.QuarterEnd(0)
    latest_df['available_date'] = latest_df['quarter_end'] + pd.Timedelta(days=release_lag_days)

    if verbose:
        print(f"  Latest data: {len(latest_df)} observations")
        print(f"  Date range: {latest_df['date'].min().strftime('%Y-%m-%d')} to {latest_df['date'].max().strftime('%Y-%m-%d')}")

    # 生成月末日期序列
    month_ends = pd.date_range(start=start_date, end=end_date, freq='ME')

    pit_data = []

    for as_of_date in month_ends:
        # 筛选在 as_of_date 之前已发布的数据
        available = latest_df[latest_df['available_date'] <= as_of_date].copy()

        if len(available) == 0:
            continue

        # 添加 as_of_date 列
        available = available[['date', 'value']].copy()
        available['as_of_date'] = as_of_date

        pit_data.append(available[['as_of_date', 'date', 'value']])

    if len(pit_data) == 0:
        return pd.DataFrame(columns=['as_of_date', 'date', 'value'])

    result = pd.concat(pit_data, ignore_index=True)

    if verbose:
        print(f"  Simulated PIT records: {len(result)}")
        unique_months = result['as_of_date'].nunique()
        print(f"  Covered months: {unique_months}")

    return result


class HybridPITLoader:
    """
    混合 PIT 数据加载器

    - 2010年后：使用 ALFRED 真实 PIT 数据
    - 2010年前：使用 FRED 最新数据 + 发布滞后模拟
    """

    def __init__(self, api_key: str, alfred_cutoff: str = '2010-01-01'):
        """
        初始化混合加载器。

        Args:
            api_key: FRED API key
            alfred_cutoff: ALFRED 数据可用的起始日期
        """
        self.fred = Fred(api_key=api_key)
        self.alfred_loader = ALFREDDataLoader(api_key)
        self.alfred_cutoff = pd.to_datetime(alfred_cutoff)

    def build_monthly_pit_series(self,
                                  series_id: str,
                                  start_date: str,
                                  end_date: str,
                                  release_lag_days: int = None,
                                  verbose: bool = True) -> pd.DataFrame:
        """
        构建完整的 PIT 月末序列，自动混合 ALFRED 和模拟数据。

        Args:
            series_id: FRED series ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            release_lag_days: 发布滞后天数（用于模拟部分）
            verbose: Print progress

        Returns:
            DataFrame with columns: ['as_of_date', 'date', 'value']
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        if verbose:
            print(f"\n  === {series_id} ===")
            print(f"  Requested range: {start_date} to {end_date}")
            print(f"  ALFRED cutoff: {self.alfred_cutoff.strftime('%Y-%m-%d')}")

        results = []

        # Part 1: 2010年前使用模拟 PIT
        if start_dt < self.alfred_cutoff:
            sim_end = min(end_dt, self.alfred_cutoff - pd.Timedelta(days=1))
            if verbose:
                print(f"\n  [Simulated PIT] {start_date} to {sim_end.strftime('%Y-%m-%d')}")

            sim_pit = build_simulated_pit_series(
                self.fred,
                series_id,
                start_date,
                sim_end.strftime('%Y-%m-%d'),
                release_lag_days=release_lag_days,
                verbose=verbose
            )
            if len(sim_pit) > 0:
                results.append(sim_pit)

        # Part 2: 2010年后使用 ALFRED 真实 PIT
        if end_dt >= self.alfred_cutoff:
            alfred_start = max(start_dt, self.alfred_cutoff)
            if verbose:
                print(f"\n  [ALFRED PIT] {alfred_start.strftime('%Y-%m-%d')} to {end_date}")

            alfred_pit = self.alfred_loader.build_monthly_pit_series(
                series_id,
                alfred_start.strftime('%Y-%m-%d'),
                end_date,
                verbose=verbose
            )
            if len(alfred_pit) > 0:
                results.append(alfred_pit)

        if len(results) == 0:
            return pd.DataFrame(columns=['as_of_date', 'date', 'value'])

        # 合并结果
        combined = pd.concat(results, ignore_index=True)

        # 去重（如果有重叠，优先 ALFRED 数据）
        combined = combined.sort_values(['as_of_date', 'date'])
        combined = combined.drop_duplicates(subset=['as_of_date', 'date'], keep='last')

        if verbose:
            print(f"\n  Combined PIT records: {len(combined)}")
            print(f"  Total months covered: {combined['as_of_date'].nunique()}")

        return combined
