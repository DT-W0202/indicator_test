"""
LQI Trend Factors Validation

Tests predictive power of trend factors using:
1. Information Coefficient (IC) - rank correlation with future returns
2. AUC - crash prediction ability (binary classification)
3. Quintile analysis - return/risk characteristics by factor level

Rolling windows: 5Y (1260 days) and 10Y (2520 days) for percentile calculation
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from trend.V9_LQI_FACTORS.lqi_factors import LQIFactors
from trend.V9_LQI_FACTORS.lqi_data_loader import LQIDataLoader


class LQITrendValidation:
    """Validate LQI trend factors"""

    def __init__(self):
        self.lqi = LQIFactors()
        self.loader = LQIDataLoader()
        self.spx = self.loader.load_spx()

        # Forward return horizons (trading days)
        self.horizons = {
            '1M': 21,
            '3M': 63,
            '6M': 126,
            '12M': 252
        }

        # Crash thresholds for AUC
        self.crash_thresholds = [0.10, 0.15, 0.20]

        # Rolling windows for percentile (trading days)
        self.pctl_windows = {
            '5Y': 1260,
            '10Y': 2520
        }

    def compute_forward_returns(self) -> pd.DataFrame:
        """Compute forward returns for SPX"""
        returns = pd.DataFrame(index=self.spx.index)
        for name, days in self.horizons.items():
            returns[f'Fwd_{name}'] = self.spx.pct_change(days).shift(-days)
        return returns

    def compute_forward_mdd(self) -> pd.DataFrame:
        """Compute forward maximum drawdown"""
        mdd = pd.DataFrame(index=self.spx.index)

        for name, days in self.horizons.items():
            fwd_mdd = pd.Series(index=self.spx.index, dtype=float)
            for i in range(len(self.spx) - days):
                window = self.spx.iloc[i:i+days+1]
                peak = window.expanding().max()
                dd = (window - peak) / peak
                fwd_mdd.iloc[i] = dd.min()
            mdd[f'Fwd_MDD_{name}'] = fwd_mdd

        return mdd

    def compute_rolling_percentile(self, factor: pd.Series, window: int) -> pd.Series:
        """Compute rolling percentile of factor"""
        def pctl_rank(x):
            if len(x.dropna()) < window * 0.5:  # Require at least 50% data
                return np.nan
            return stats.percentileofscore(x.dropna(), x.iloc[-1]) if not pd.isna(x.iloc[-1]) else np.nan

        return factor.rolling(window, min_periods=int(window * 0.5)).apply(pctl_rank, raw=False)

    def compute_ic(self, factor: pd.Series, returns: pd.Series, method: str = 'spearman') -> dict:
        """Compute Information Coefficient with significance test"""
        # Align data
        common_idx = factor.dropna().index.intersection(returns.dropna().index)
        f = factor.reindex(common_idx)
        r = returns.reindex(common_idx)

        if len(common_idx) < 30:
            return {'ic': np.nan, 'pvalue': np.nan, 'n': len(common_idx)}

        if method == 'spearman':
            ic, pvalue = stats.spearmanr(f, r)
        else:
            ic, pvalue = stats.pearsonr(f, r)

        return {'ic': ic, 'pvalue': pvalue, 'n': len(common_idx)}

    def compute_auc(self, factor: pd.Series, crash_indicator: pd.Series) -> dict:
        """Compute AUC for crash prediction"""
        # Align data
        common_idx = factor.dropna().index.intersection(crash_indicator.dropna().index)
        f = factor.reindex(common_idx)
        c = crash_indicator.reindex(common_idx)

        if len(common_idx) < 30 or c.sum() < 5:
            return {'auc': np.nan, 'n': len(common_idx), 'n_crash': int(c.sum()) if not pd.isna(c.sum()) else 0}

        try:
            auc = roc_auc_score(c, f)
            return {'auc': auc, 'n': len(common_idx), 'n_crash': int(c.sum())}
        except:
            return {'auc': np.nan, 'n': len(common_idx), 'n_crash': int(c.sum()) if not pd.isna(c.sum()) else 0}

    def quintile_analysis(self, factor: pd.Series, returns: pd.Series, mdd: pd.Series) -> pd.DataFrame:
        """Analyze returns and risk by factor quintile"""
        # Align data
        common_idx = factor.dropna().index.intersection(returns.dropna().index).intersection(mdd.dropna().index)
        f = factor.reindex(common_idx)
        r = returns.reindex(common_idx)
        m = mdd.reindex(common_idx)

        if len(common_idx) < 100:
            return pd.DataFrame()

        # Create quintiles
        quintiles = pd.qcut(f, 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')

        results = []
        for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            mask = quintiles == q
            if mask.sum() < 10:
                continue
            results.append({
                'Quintile': q,
                'N': mask.sum(),
                'Mean_Return': r[mask].mean() * 100,
                'Std_Return': r[mask].std() * 100,
                'Mean_MDD': m[mask].mean() * 100,
                'Crash_Rate_10': (m[mask] < -0.10).mean() * 100,
                'Crash_Rate_15': (m[mask] < -0.15).mean() * 100,
                'Crash_Rate_20': (m[mask] < -0.20).mean() * 100,
            })

        return pd.DataFrame(results)

    def validate_single_factor(self, factor_name: str, factor: pd.Series,
                                pctl_window_name: str, pctl_window: int) -> dict:
        """Validate a single factor with given percentile window"""
        # Compute factor percentile
        factor_pctl = self.compute_rolling_percentile(factor, pctl_window)

        # Get returns and MDD
        returns = self.compute_forward_returns()
        mdd = self.compute_forward_mdd()

        results = {
            'factor': factor_name,
            'pctl_window': pctl_window_name,
            'data_start': factor.dropna().index.min(),
            'data_end': factor.dropna().index.max(),
            'n_obs': len(factor.dropna()),
        }

        # IC for each horizon
        for horizon in ['1M', '3M', '6M', '12M']:
            ic_result = self.compute_ic(factor_pctl, returns[f'Fwd_{horizon}'])
            results[f'IC_{horizon}'] = ic_result['ic']
            results[f'IC_{horizon}_p'] = ic_result['pvalue']

        # AUC for crash prediction (12M horizon)
        for thresh in self.crash_thresholds:
            crash = (mdd['Fwd_MDD_12M'] < -thresh).astype(float)
            auc_result = self.compute_auc(factor_pctl, crash)
            results[f'AUC_{int(thresh*100)}'] = auc_result['auc']
            results[f'N_Crash_{int(thresh*100)}'] = auc_result['n_crash']

        # Quintile analysis
        quintile_df = self.quintile_analysis(factor_pctl, returns['Fwd_12M'], mdd['Fwd_MDD_12M'])
        if not quintile_df.empty:
            results['Q1_Return'] = quintile_df[quintile_df['Quintile'] == 'Q1']['Mean_Return'].values[0] if 'Q1' in quintile_df['Quintile'].values else np.nan
            results['Q5_Return'] = quintile_df[quintile_df['Quintile'] == 'Q5']['Mean_Return'].values[0] if 'Q5' in quintile_df['Quintile'].values else np.nan
            results['Q1_Crash20'] = quintile_df[quintile_df['Quintile'] == 'Q1']['Crash_Rate_20'].values[0] if 'Q1' in quintile_df['Quintile'].values else np.nan
            results['Q5_Crash20'] = quintile_df[quintile_df['Quintile'] == 'Q5']['Crash_Rate_20'].values[0] if 'Q5' in quintile_df['Quintile'].values else np.nan

        return results

    def run_full_validation(self) -> pd.DataFrame:
        """Run validation for all factors and windows"""
        # Get all factors
        factors = {
            'T1_VTS': self.lqi.compute_vix_term_structure(),
            'T2_VVIX': self.lqi.compute_vvix(),
            'T3_SKEW': self.lqi.compute_skew(),
            'T4a_HYG_Flow': self.lqi.compute_hyg_flow(),
            'T4b_LQD_Flow': self.lqi.compute_lqd_flow(),
            'T5_TLT_Flow': self.lqi.compute_tlt_flow(),
            'T6_Funding': self.lqi.compute_funding_spread(),
            'T7_VIX': self.lqi.compute_vix_level(),
            'T8a_Dealer_Short': self.lqi.compute_dealer_inventory_short(),
            'T8b_Dealer_Mid': self.lqi.compute_dealer_inventory_mid(),
            'T8c_Dealer_Long': self.lqi.compute_dealer_inventory_long(),
            'T10_GCF_IORB': self.lqi.compute_gcf_iorb_spread(),
        }

        all_results = []

        for factor_name, factor in factors.items():
            print(f"\nValidating {factor_name}...")

            for window_name, window in self.pctl_windows.items():
                print(f"  Window: {window_name}")
                try:
                    result = self.validate_single_factor(factor_name, factor, window_name, window)
                    all_results.append(result)
                except Exception as e:
                    print(f"    Error: {e}")

        return pd.DataFrame(all_results)

    def print_summary(self, results: pd.DataFrame):
        """Print validation summary"""
        print("\n" + "=" * 100)
        print("LQI TREND FACTORS VALIDATION SUMMARY")
        print("=" * 100)

        for window in ['5Y', '10Y']:
            print(f"\n{'='*50}")
            print(f"Percentile Window: {window}")
            print("=" * 50)

            window_results = results[results['pctl_window'] == window]

            # IC Summary
            print("\n--- Information Coefficient (IC) ---")
            print(f"{'Factor':<20} {'IC_1M':>8} {'IC_3M':>8} {'IC_6M':>8} {'IC_12M':>8} {'N':>8}")
            print("-" * 60)
            for _, row in window_results.iterrows():
                ic_1m = f"{row['IC_1M']:.3f}" if not pd.isna(row['IC_1M']) else "N/A"
                ic_3m = f"{row['IC_3M']:.3f}" if not pd.isna(row['IC_3M']) else "N/A"
                ic_6m = f"{row['IC_6M']:.3f}" if not pd.isna(row['IC_6M']) else "N/A"
                ic_12m = f"{row['IC_12M']:.3f}" if not pd.isna(row['IC_12M']) else "N/A"

                # Add significance markers
                if not pd.isna(row['IC_12M_p']) and row['IC_12M_p'] < 0.05:
                    ic_12m += "*"
                if not pd.isna(row['IC_12M_p']) and row['IC_12M_p'] < 0.01:
                    ic_12m += "*"

                print(f"{row['factor']:<20} {ic_1m:>8} {ic_3m:>8} {ic_6m:>8} {ic_12m:>8} {row['n_obs']:>8}")

            # AUC Summary
            print("\n--- Crash Prediction AUC (12M Forward MDD) ---")
            print(f"{'Factor':<20} {'AUC_10%':>10} {'AUC_15%':>10} {'AUC_20%':>10} {'N_Crash':>10}")
            print("-" * 60)
            for _, row in window_results.iterrows():
                auc_10 = f"{row['AUC_10']:.3f}" if not pd.isna(row['AUC_10']) else "N/A"
                auc_15 = f"{row['AUC_15']:.3f}" if not pd.isna(row['AUC_15']) else "N/A"
                auc_20 = f"{row['AUC_20']:.3f}" if not pd.isna(row['AUC_20']) else "N/A"
                n_crash = int(row['N_Crash_20']) if not pd.isna(row['N_Crash_20']) else 0
                print(f"{row['factor']:<20} {auc_10:>10} {auc_15:>10} {auc_20:>10} {n_crash:>10}")

            # Quintile Summary
            print("\n--- Quintile Analysis (12M Forward) ---")
            print(f"{'Factor':<20} {'Q1_Ret%':>10} {'Q5_Ret%':>10} {'Q1_Crash%':>10} {'Q5_Crash%':>10}")
            print("-" * 60)
            for _, row in window_results.iterrows():
                q1_ret = f"{row['Q1_Return']:.1f}" if not pd.isna(row.get('Q1_Return')) else "N/A"
                q5_ret = f"{row['Q5_Return']:.1f}" if not pd.isna(row.get('Q5_Return')) else "N/A"
                q1_crash = f"{row['Q1_Crash20']:.1f}" if not pd.isna(row.get('Q1_Crash20')) else "N/A"
                q5_crash = f"{row['Q5_Crash20']:.1f}" if not pd.isna(row.get('Q5_Crash20')) else "N/A"
                print(f"{row['factor']:<20} {q1_ret:>10} {q5_ret:>10} {q1_crash:>10} {q5_crash:>10}")

        # Best factors summary
        print("\n" + "=" * 100)
        print("KEY FINDINGS")
        print("=" * 100)

        # Find best IC factors
        best_ic = results.nlargest(5, 'IC_12M')[['factor', 'pctl_window', 'IC_12M', 'IC_12M_p']]
        print("\nTop 5 Factors by IC (12M):")
        for _, row in best_ic.iterrows():
            sig = "***" if row['IC_12M_p'] < 0.001 else "**" if row['IC_12M_p'] < 0.01 else "*" if row['IC_12M_p'] < 0.05 else ""
            print(f"  {row['factor']} ({row['pctl_window']}): IC = {row['IC_12M']:.3f}{sig}")

        # Find best AUC factors
        best_auc = results.nlargest(5, 'AUC_20')[['factor', 'pctl_window', 'AUC_20']]
        print("\nTop 5 Factors by AUC (20% Crash):")
        for _, row in best_auc.iterrows():
            print(f"  {row['factor']} ({row['pctl_window']}): AUC = {row['AUC_20']:.3f}")


def main():
    """Main function"""
    print("=" * 100)
    print("LQI TREND FACTORS VALIDATION")
    print("=" * 100)

    validator = LQITrendValidation()

    print("\nRunning validation...")
    results = validator.run_full_validation()

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), 'validation_results.csv')
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Print summary
    validator.print_summary(results)


if __name__ == '__main__':
    main()
