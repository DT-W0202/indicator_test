"""
LQI Trend Factors

Potential leading indicators for market risk/return:

T1. VIX Term Structure (VTS): VIX / VIX3M - 1
    - > 0: Backwardation (short-term fear exceeds medium-term)
    - < 0: Contango (complacency, normal market)

T2. VVIX (Vol of Vol) [DEACTIVATED - weak IC, redundant with T7]
    - High VVIX: uncertainty about future volatility
    - Typically spikes before/during market stress

T3. SKEW Index
    - High SKEW: demand for put protection (tail risk concern)
    - Low SKEW: complacency

T4a. HYG Flow (High Yield Bond ETF)
    - Based on Shares Outstanding changes
    - 20-day flow as % of AUM
    - Positive = inflows to high yield = risk-on

T4b. LQD Flow (Investment Grade Bond ETF)
    - Based on Shares Outstanding changes
    - 20-day flow as % of AUM
    - Positive = inflows to investment grade

T5. TLT Flow (Treasury Bond ETF)
    - Based on Shares Outstanding changes
    - 20-day flow as % of AUM
    - Negative = outflows from treasuries = potential stress

T6. Funding Spread (Z-Score)
    - Pre-2018: TED Spread (LIBOR - T-Bill)
    - Post-2018: EFFR - SOFR
    - Z-score normalized for comparability
    - High = funding stress above recent average

T7. VIX Level (raw) [DEACTIVATED - high AUC but no IC, pure risk indicator not trend factor]
    - Classic fear gauge

T8. Dealer Inventory (Short/Mid/Long)
    - NY Fed Primary Dealer Positions
    - Z-score of weekly changes
    - Lower = dealers reducing inventory = liquidity stress

T9. (Removed - merged into T5)

T10. GCF-IORB Spread
    - GCF Treasury Repo Rate - IORB
    - Higher = repo funding stress relative to reserves
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from trend.V9_LQI_FACTORS.lqi_data_loader import LQIDataLoader


class LQIFactors:
    """Compute LQI trend factors"""

    def __init__(self, cache_path: str = None):
        self.loader = LQIDataLoader(cache_path)
        self._factors: Dict[str, pd.Series] = {}

    def compute_vix_term_structure(self) -> pd.Series:
        """
        T1: VIX Term Structure = VIX / VIX3M - 1

        Interpretation:
        - > 0: Backwardation (fear, short-term stress)
        - < 0: Contango (complacency)
        - Extreme positive: panic
        """
        if 'T1_VTS' not in self._factors:
            vix = self.loader.load_vix()
            vix3m = self.loader.load_vix3m()

            # Align to common dates
            common_idx = vix.index.intersection(vix3m.index)
            vix_aligned = vix.reindex(common_idx)
            vix3m_aligned = vix3m.reindex(common_idx)

            vts = (vix_aligned / vix3m_aligned) - 1
            vts.name = 'T1_VTS'
            self._factors['T1_VTS'] = vts

        return self._factors['T1_VTS']

    def compute_vvix(self) -> pd.Series:
        """
        T2: VVIX (Vol of Vol)

        Interpretation:
        - High (>100): High uncertainty
        - Normal (80-100): Normal
        - Low (<80): Complacency
        """
        if 'T2_VVIX' not in self._factors:
            vvix = self.loader.load_vvix()
            vvix.name = 'T2_VVIX'
            self._factors['T2_VVIX'] = vvix

        return self._factors['T2_VVIX']

    def compute_skew(self) -> pd.Series:
        """
        T3: CBOE SKEW Index

        Interpretation:
        - High (>130): High tail risk concern
        - Normal (115-130): Normal
        - Low (<115): Low tail risk concern
        """
        if 'T3_SKEW' not in self._factors:
            skew = self.loader.load_skew()
            skew.name = 'T3_SKEW'
            self._factors['T3_SKEW'] = skew

        return self._factors['T3_SKEW']

    def _compute_flow_pct(self, df: pd.DataFrame) -> pd.Series:
        """Compute 20-day flow as % of AUM"""
        df = df.copy()
        df['Flow'] = df['Shares'].diff() * df['NAV']
        df['Flow_20D'] = df['Flow'].rolling(20).sum()
        df['AUM'] = df['NAV'] * df['Shares']
        df['Flow_Pct'] = df['Flow_20D'] / df['AUM'].shift(20) * 100
        return df['Flow_Pct']

    def compute_hyg_flow(self) -> pd.Series:
        """
        T4a: HYG Flow (High Yield Bond ETF)

        Based on ETF Shares Outstanding changes:
        - 20-day flow as % of AUM
        - Positive = inflows to high yield = risk-on sentiment
        - Negative = outflows from high yield = risk-off sentiment
        """
        if 'T4a_HYG_Flow' not in self._factors:
            hyg = self.loader.load_hyg_full()
            hyg_flow = self._compute_flow_pct(hyg)
            hyg_flow.name = 'T4a_HYG_Flow'
            self._factors['T4a_HYG_Flow'] = hyg_flow

        return self._factors['T4a_HYG_Flow']

    def compute_lqd_flow(self) -> pd.Series:
        """
        T4b: LQD Flow (Investment Grade Bond ETF)

        Based on ETF Shares Outstanding changes:
        - 20-day flow as % of AUM
        - Positive = inflows to investment grade
        - Negative = outflows from investment grade
        """
        if 'T4b_LQD_Flow' not in self._factors:
            lqd = self.loader.load_lqd_full()
            lqd_flow = self._compute_flow_pct(lqd)
            lqd_flow.name = 'T4b_LQD_Flow'
            self._factors['T4b_LQD_Flow'] = lqd_flow

        return self._factors['T4b_LQD_Flow']

    def compute_tlt_flow(self) -> pd.Series:
        """
        T5: TLT Flow (Treasury Bond ETF)

        Based on ETF Shares Outstanding changes:
        - 20-day flow as % of AUM
        - Positive = inflows to treasuries = flight to safety
        - Negative = outflows from treasuries = risk-on or liquidity stress
        """
        if 'T5_TLT_Flow' not in self._factors:
            tlt = self.loader.load_tlt_full()
            tlt_flow = self._compute_flow_pct(tlt)
            tlt_flow.name = 'T5_TLT_Flow'
            self._factors['T5_TLT_Flow'] = tlt_flow

        return self._factors['T5_TLT_Flow']

    def compute_funding_spread(self, zscore_window: int = 252) -> pd.Series:
        """
        T6: Funding Spread (Z-Score)

        Combines two funding stress measures:
        - Pre-2018: TED Spread (LIBOR - T-Bill, discontinued 2022)
        - Post-2018: EFFR - SOFR (unsecured vs secured overnight)

        Both are converted to rolling z-scores for comparability,
        then spliced at the transition point (2018-04-03).

        For seamless transition, the first year of EFFR-SOFR uses
        expanding window z-score bootstrapped from initial statistics.

        Interpretation:
        - High z-score: funding stress above recent average
        - Low z-score: funding stress below recent average
        """
        if 'T6_Funding' not in self._factors:
            # Load TED Spread (1986 - 2022-01)
            ted = self.loader.load_ted_spread().dropna()

            # Load EFFR-SOFR spread (2018-04+)
            effr = self.loader.load_effr()
            sofr = self.loader.load_sofr()
            common_idx = effr.index.intersection(sofr.index)
            effr_sofr = (effr.reindex(common_idx) - sofr.reindex(common_idx)).dropna()

            # Compute rolling z-scores for TED
            def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
                return (s - s.rolling(window).mean()) / s.rolling(window).std()

            ted_zscore = rolling_zscore(ted, zscore_window)

            # For EFFR-SOFR, use expanding window for first year, then rolling
            # This ensures no gap at the splice point
            def expanding_then_rolling_zscore(s: pd.Series, window: int) -> pd.Series:
                result = pd.Series(index=s.index, dtype=float)
                for i in range(len(s)):
                    if i < window:
                        # Expanding window for warmup period
                        if i >= 20:  # Minimum 20 observations
                            window_data = s.iloc[:i+1]
                            result.iloc[i] = (s.iloc[i] - window_data.mean()) / window_data.std()
                    else:
                        # Rolling window after warmup
                        window_data = s.iloc[i-window+1:i+1]
                        result.iloc[i] = (s.iloc[i] - window_data.mean()) / window_data.std()
                return result

            effr_sofr_zscore = expanding_then_rolling_zscore(effr_sofr, zscore_window)

            # Splice at SOFR start date (2018-04-03)
            # Use TED before this date, EFFR-SOFR after
            splice_date = pd.Timestamp('2018-04-03')

            ted_part = ted_zscore[ted_zscore.index < splice_date]
            effr_sofr_part = effr_sofr_zscore[effr_sofr_zscore.index >= splice_date]

            # Combine the two series
            combined = pd.concat([ted_part, effr_sofr_part])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined = combined.sort_index()
            combined.name = 'T6_Funding'

            self._factors['T6_Funding'] = combined

        return self._factors['T6_Funding']

    def compute_vix_level(self) -> pd.Series:
        """
        T7: VIX Level (raw)

        Classic fear gauge
        """
        if 'T7_VIX' not in self._factors:
            vix = self.loader.load_vix()
            vix.name = 'T7_VIX'
            self._factors['T7_VIX'] = vix

        return self._factors['T7_VIX']

    def compute_vix_percentile(self, window: int = 252) -> pd.Series:
        """
        T7b: VIX Rolling Percentile
        """
        if 'T7b_VIX_Pctl' not in self._factors:
            vix = self.loader.load_vix()

            def rolling_pctl(x):
                return (x.rank().iloc[-1] / len(x)) * 100

            vix_pctl = vix.rolling(window).apply(rolling_pctl, raw=False)
            vix_pctl.name = 'T7b_VIX_Pctl'
            self._factors['T7b_VIX_Pctl'] = vix_pctl

        return self._factors['T7b_VIX_Pctl']

    def compute_dealer_inventory_short(self, zscore_window: int = 52) -> pd.Series:
        """
        T8a: Dealer Inventory Short (Bills + <=2Y)

        Z-score of weekly position levels.
        Lower = dealers reducing short-term inventory = liquidity stress
        """
        if 'T8a_Dealer_Short' not in self._factors:
            inventory = self.loader.load_dealer_inventory()
            short = inventory['Short'].dropna()

            # Z-score over rolling window (weekly data, 52 = 1 year)
            zscore = (short - short.rolling(zscore_window).mean()) / short.rolling(zscore_window).std()
            zscore.name = 'T8a_Dealer_Short'
            self._factors['T8a_Dealer_Short'] = zscore

        return self._factors['T8a_Dealer_Short']

    def compute_dealer_inventory_mid(self, zscore_window: int = 52) -> pd.Series:
        """
        T8b: Dealer Inventory Mid (2-7Y)

        Z-score of weekly position levels.
        """
        if 'T8b_Dealer_Mid' not in self._factors:
            inventory = self.loader.load_dealer_inventory()
            mid = inventory['Mid'].dropna()

            zscore = (mid - mid.rolling(zscore_window).mean()) / mid.rolling(zscore_window).std()
            zscore.name = 'T8b_Dealer_Mid'
            self._factors['T8b_Dealer_Mid'] = zscore

        return self._factors['T8b_Dealer_Mid']

    def compute_dealer_inventory_long(self, zscore_window: int = 52) -> pd.Series:
        """
        T8c: Dealer Inventory Long (7Y+)

        Z-score of weekly position levels.
        """
        if 'T8c_Dealer_Long' not in self._factors:
            inventory = self.loader.load_dealer_inventory()
            long = inventory['Long'].dropna()

            zscore = (long - long.rolling(zscore_window).mean()) / long.rolling(zscore_window).std()
            zscore.name = 'T8c_Dealer_Long'
            self._factors['T8c_Dealer_Long'] = zscore

        return self._factors['T8c_Dealer_Long']

    def compute_gcf_iorb_spread(self) -> pd.Series:
        """
        T10: GCF-IORB Spread

        GCF Treasury Repo Rate - IORB (using EFFR as IORB proxy)
        Higher = repo funding stress relative to reserves
        """
        if 'T10_GCF_IORB' not in self._factors:
            gcf = self.loader.load_gcf_repo_full()
            effr = self.loader.load_effr()  # IORB proxy

            # Align to common dates
            common_idx = gcf.index.intersection(effr.index)
            gcf_aligned = gcf['Treasury_Rate'].reindex(common_idx)
            effr_aligned = effr.reindex(common_idx)

            # GCF - IORB spread (in percentage points)
            spread = gcf_aligned - effr_aligned
            spread.name = 'T10_GCF_IORB'
            self._factors['T10_GCF_IORB'] = spread

        return self._factors['T10_GCF_IORB']

    def compute_all_factors(self) -> pd.DataFrame:
        """Compute all active factors and return as DataFrame"""
        factors = {
            'T1_VTS': self.compute_vix_term_structure(),
            # 'T2_VVIX': self.compute_vvix(),  # DEACTIVATED - weak IC, redundant with T7
            'T3_SKEW': self.compute_skew(),
            'T4a_HYG_Flow': self.compute_hyg_flow(),
            'T4b_LQD_Flow': self.compute_lqd_flow(),
            'T5_TLT_Flow': self.compute_tlt_flow(),
            'T6_Funding': self.compute_funding_spread(),
            # 'T7_VIX': self.compute_vix_level(),  # DEACTIVATED - no IC, pure risk indicator
            'T8a_Dealer_Short': self.compute_dealer_inventory_short(),
            'T8b_Dealer_Mid': self.compute_dealer_inventory_mid(),
            'T8c_Dealer_Long': self.compute_dealer_inventory_long(),
            'T10_GCF_IORB': self.compute_gcf_iorb_spread(),
        }

        # Combine all factors, aligned to common dates
        df = pd.DataFrame(factors)
        return df

    def summary(self) -> None:
        """Print summary of all active factors"""
        print("=" * 70)
        print("LQI Trend Factors Summary (Active Only)")
        print("=" * 70)

        factors = [
            ('T1: VIX Term Structure', self.compute_vix_term_structure),
            # ('T2: VVIX', self.compute_vvix),  # DEACTIVATED
            ('T3: SKEW', self.compute_skew),
            ('T4a: HYG Flow', self.compute_hyg_flow),
            ('T4b: LQD Flow', self.compute_lqd_flow),
            ('T5: TLT Flow', self.compute_tlt_flow),
            ('T6: Funding Spread', self.compute_funding_spread),
            # ('T7: VIX Level', self.compute_vix_level),  # DEACTIVATED
            ('T8a: Dealer Inventory Short', self.compute_dealer_inventory_short),
            ('T8b: Dealer Inventory Mid', self.compute_dealer_inventory_mid),
            ('T8c: Dealer Inventory Long', self.compute_dealer_inventory_long),
            ('T10: GCF-IORB Spread', self.compute_gcf_iorb_spread),
        ]

        for name, compute_fn in factors:
            try:
                data = compute_fn()
                data_clean = data.dropna()
                print(f"\n{name}:")
                print(f"  Range: {data_clean.index.min().date()} to {data_clean.index.max().date()}")
                print(f"  Count: {len(data_clean)}")
                print(f"  Current: {data_clean.iloc[-1]:.4f}")
                print(f"  Mean: {data_clean.mean():.4f}, Std: {data_clean.std():.4f}")
                print(f"  Min: {data_clean.min():.4f}, Max: {data_clean.max():.4f}")
            except Exception as e:
                print(f"\n{name}: Error - {e}")


if __name__ == '__main__':
    lqi = LQIFactors()
    lqi.summary()

    print("\n" + "=" * 70)
    print("Current Factor Readings")
    print("=" * 70)

    df = lqi.compute_all_factors()
    latest = df.dropna().iloc[-1]
    print(f"\nDate: {df.dropna().index[-1].date()}")
    for col in latest.index:
        print(f"  {col}: {latest[col]:.4f}")
