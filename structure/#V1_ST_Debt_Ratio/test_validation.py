#!/usr/bin/env python3
"""
V1 (New): Corporate Short-Term Debt Ratio - Structural Validation
==================================================================

Factor: BOGZ1FL104140006Q - Nonfinancial Corporate Business;
        Short-Term Debt as a Percentage of Total Debt

This replaces the old V1 (Debt-GDP Gap) with a cleaner, pre-calculated ratio.

Validation Framework:
1. Series Metadata Audit
2. Structural Break Analysis (on BETA)
3. Interest Rate Regime + Interaction Regression
4. Risk Target Variables + Tail Quantile
5. Quintile Analysis
6. Bootstrap Robustness
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import spearmanr
from fredapi import Fred
import warnings
warnings.filterwarnings('ignore')

# Add project path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from lib.structural_break import (
    analyze_structural_break,
    rolling_beta_ols,
    compute_subsample_beta,
)
from lib.regime_analysis import (
    compute_current_drawdown,
    compute_forward_max_drawdown,
    compute_forward_realized_vol,
    classify_drawdown_regime,
    compute_regime_ic,
    compute_risk_target_ic,
    compute_drawdown_event_auc,
    run_conditional_regression,
    run_interaction_regression,
    run_quintile_analysis,
)
from lib.hac_inference import (
    ols_with_hac,
    compute_tail_quantile_ic,
    block_bootstrap_regression,
)

# ============== Configuration ==============
FRED_API_KEY = 'b37a95dcefcfcc0f98ddfb87daca2e34'
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'structure', 'V1_ST_Debt_Ratio')

# Factor settings
FACTOR_SERIES = 'BOGZ1FL104140006Q'
RELEASE_LAG_MONTHS = 5  # ~161 days = ~5 months lag based on ALFRED analysis

# Percentile windows (in quarters, for quarterly data)
PERCENTILE_WINDOWS = {
    '5Y': 20,   # 5 years = 20 quarters
    '10Y': 40,  # 10 years = 40 quarters
}

# HAC lag for 12M overlapping returns
HAC_LAG = 11

# Rate threshold (median will be used)
USE_MEDIAN_RATE = True


def load_factor_from_fred():
    """
    Load factor data from FRED with proper lag adjustment.

    Data Processing:
    1. Get quarterly FRED data
    2. Apply 5-month publication lag (based on ALFRED analysis)
    3. Compute rolling percentile on quarterly data (5Y and 10Y windows)
    4. Forward-fill to monthly for alignment with SPX
    """
    print("\n[Loading Factor Data from FRED with Lag Adjustment]")
    fred = Fred(api_key=FRED_API_KEY)

    # Get factor series (quarterly)
    factor_raw = fred.get_series(FACTOR_SERIES)
    factor_raw = factor_raw.dropna()

    print(f"  Series: {FACTOR_SERIES}")
    print(f"  Data range: {factor_raw.index.min()} to {factor_raw.index.max()}")
    print(f"  Quarterly data points: {len(factor_raw)}")
    print(f"  Release lag: {RELEASE_LAG_MONTHS} months (based on ALFRED)")

    # Rolling percentile function (on quarterly data)
    def rolling_percentile(series, window):
        """Compute rolling percentile rank (0-100) on quarterly data"""
        result = pd.Series(index=series.index, dtype=float)
        for i in range(window, len(series)):
            window_data = series.iloc[i-window:i]
            current = series.iloc[i]
            pct = (window_data < current).sum() / len(window_data) * 100
            result.iloc[i] = pct
        return result

    # Compute percentiles on quarterly data BEFORE forward-filling
    percentiles = {}
    for name, window in PERCENTILE_WINDOWS.items():
        pctl = rolling_percentile(factor_raw, window)
        percentiles[name] = pctl
        valid_count = pctl.notna().sum()
        print(f"  {name} window ({window}Q): {valid_count} valid points")

    # Create quarterly dataframe
    df_quarterly = pd.DataFrame({
        'factor_raw': factor_raw,
        'factor_pctl_5Y': percentiles['5Y'],
        'factor_pctl_10Y': percentiles['10Y'],
    })

    # Apply publication lag: shift index forward by 5 months
    # This means Q1 data (dated 2024-01-01) becomes available around 2024-06-01
    df_quarterly.index = df_quarterly.index + pd.DateOffset(months=RELEASE_LAG_MONTHS)
    print(f"  After lag adjustment: {df_quarterly.index.min()} to {df_quarterly.index.max()}")

    # Forward-fill to monthly
    df_monthly = df_quarterly.resample('ME').last().ffill()
    df_monthly.index.name = 'as_of_date'

    # Drop NaN
    df_monthly = df_monthly.dropna()

    print(f"  Monthly data points (after ffill): {len(df_monthly)}")
    print(f"  Factor raw range: {df_monthly['factor_raw'].min():.2f}% - {df_monthly['factor_raw'].max():.2f}%")
    print(f"  Current value (as of {df_monthly.index[-1].strftime('%Y-%m')}): {df_monthly['factor_raw'].iloc[-1]:.2f}%")

    return df_monthly


def load_spx():
    """Load SPX daily data"""
    filepath = os.path.join(PROJECT_ROOT, 'SPX 1D Data (1).csv')
    df = pd.read_csv(filepath, parse_dates=['time'])
    df = df.set_index('time')[['close']].rename(columns={'close': 'SPX'})
    return df['SPX']


def load_fed_funds_rate():
    """Load Federal Funds Rate from FRED"""
    fred = Fred(api_key=FRED_API_KEY)
    ffr = fred.get_series('FEDFUNDS')
    ffr = ffr.resample('ME').last()
    return ffr


# ============== Step 0: Series Metadata Audit ==============

def run_metadata_audit() -> dict:
    """Audit FRED series metadata"""
    print("\n" + "=" * 60)
    print("Step 0: Series Metadata Audit")
    print("=" * 60)

    fred = Fred(api_key=FRED_API_KEY)

    series_list = [FACTOR_SERIES, 'FEDFUNDS']
    results = {}

    for series_id in series_list:
        try:
            info = fred.get_series_info(series_id)
            results[series_id] = {
                'title': info.get('title', 'N/A'),
                'units': info.get('units', 'N/A'),
                'frequency': info.get('frequency', 'N/A'),
                'seasonal_adjustment': info.get('seasonal_adjustment', 'N/A'),
            }
            print(f"\n  [{series_id}]")
            print(f"    Title: {results[series_id]['title']}")
            print(f"    Units: {results[series_id]['units']}")
            print(f"    Frequency: {results[series_id]['frequency']}")
        except Exception as e:
            print(f"  [{series_id}] Error: {e}")
            results[series_id] = {'error': str(e)}

    return results


# ============== Step 1: Structural Break Analysis ==============

def run_structural_break_analysis(factor: pd.Series, spx: pd.Series) -> dict:
    """Analyze structural breaks on regression BETA"""
    print("\n" + "=" * 60)
    print("Step 1: Structural Break Analysis (on BETA)")
    print("=" * 60)

    spx_monthly = spx.resample('ME').last()
    fwd_return_12m = np.log(spx_monthly.shift(-13) / spx_monthly.shift(-1))

    common_idx = factor.index.intersection(fwd_return_12m.index)
    factor_aligned = factor.loc[common_idx]
    return_aligned = fwd_return_12m.loc[common_idx]

    results = analyze_structural_break(
        factor_aligned,
        return_aligned,
        candidate_breakpoints=['2008-01-01', '2015-01-01', '2020-01-01'],
        rolling_window=120
    )

    print("\n[Chow Test Results]")
    for bp, stats in results['chow_tests'].items():
        sig = "***" if stats['significant'] else ""
        print(f"  {bp}: F={stats['f_statistic']:.2f}, p={stats['p_value']:.4f} {sig}")

    print("\n[Subsample BETA]")
    for period, stats in results['subsample_beta'].items():
        sig = "***" if stats['p_value'] < 0.01 else "**" if stats['p_value'] < 0.05 else "*" if stats['p_value'] < 0.1 else ""
        print(f"  {period}: beta={stats['beta']:.4f}, t={stats['t_stat']:.2f}, p={stats['p_value']:.4f} {sig}")

    return results


def plot_structural_break(results: dict, factor: pd.Series, spx: pd.Series, save_path: str = None):
    """Plot structural break analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('V1 ST Debt Ratio: Structural Break Analysis', fontsize=14, fontweight='bold')

    # Plot 1: Rolling BETA
    ax1 = axes[0, 0]
    rolling_beta = results.get('rolling_beta')
    if rolling_beta is not None and len(rolling_beta) > 0:
        ax1.plot(rolling_beta.index, rolling_beta['beta'], 'b-', linewidth=1.5, label='Rolling β')
        ax1.axhline(0, color='black', linewidth=0.5)
        for cp in results.get('changepoints', []):
            ax1.axvline(cp, color='red', linestyle='--', alpha=0.7)
    ax1.set_title('1. Rolling 10Y Regression β')
    ax1.set_ylabel('β (Factor → 12M Return)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Subsample BETA
    ax2 = axes[0, 1]
    sub_beta = results.get('subsample_beta', {})
    if sub_beta:
        periods = list(sub_beta.keys())
        betas = [sub_beta[p]['beta'] for p in periods]
        colors = ['green' if b < 0 else 'red' for b in betas]
        ax2.bar(range(len(periods)), betas, color=colors, alpha=0.7, edgecolor='black')
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.set_xticks(range(len(periods)))
        ax2.set_xticklabels(periods, rotation=45)
        ax2.set_ylabel('Regression β')
        ax2.set_title('2. Subsample β Comparison')

        for i, p in enumerate(periods):
            pval = sub_beta[p]['p_value']
            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            if sig:
                ax2.annotate(sig, (i, betas[i]), ha='center',
                             va='bottom' if betas[i] > 0 else 'top')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Chow test
    ax3 = axes[1, 0]
    chow = results.get('chow_tests', {})
    if chow:
        breakpoints = list(chow.keys())
        f_stats = [chow[bp]['f_statistic'] for bp in breakpoints]
        colors = ['red' if chow[bp]['significant'] else 'gray' for bp in breakpoints]
        ax3.bar(range(len(breakpoints)), f_stats, color=colors, alpha=0.7, edgecolor='black')
        ax3.axhline(3.0, color='orange', linestyle='--', alpha=0.7, label='~p=0.05')
        ax3.set_xticks(range(len(breakpoints)))
        ax3.set_xticklabels(breakpoints)
        ax3.set_ylabel('F-statistic')
        ax3.set_title('3. Chow Test')
        ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Factor timeseries
    ax4 = axes[1, 1]
    ax4.plot(factor.index, factor, 'b-', linewidth=1)
    ax4.set_title('4. Factor Percentile Time Series')
    ax4.set_ylabel('Percentile Rank')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()
    return fig


# ============== Step 2: Rate Regime Analysis ==============

def run_rate_regime_analysis(factor: pd.Series, spx: pd.Series) -> dict:
    """Analyze factor under different interest rate regimes"""
    print("\n" + "=" * 60)
    print("Step 2: Interest Rate Regime + Interaction Regression")
    print("=" * 60)

    try:
        ffr = load_fed_funds_rate()
    except Exception as e:
        print(f"  WARNING: Failed to load Fed Funds Rate: {e}")
        return {'error': str(e)}

    spx_monthly = spx.resample('ME').last()
    fwd_return_12m = np.log(spx_monthly.shift(-13) / spx_monthly.shift(-1))

    common_idx = factor.index.intersection(fwd_return_12m.index).intersection(ffr.index)
    factor_aligned = factor.loc[common_idx]
    return_aligned = fwd_return_12m.loc[common_idx]
    ffr_aligned = ffr.loc[common_idx]

    results = {}

    # Use median as threshold
    rate_threshold = ffr_aligned.median()
    print(f"\n  Rate threshold (median): {rate_threshold:.2f}%")

    rate_regime = (ffr_aligned > rate_threshold).astype(int)

    # Grouped IC
    grouped_ic = compute_regime_ic(factor_aligned, return_aligned, rate_regime)
    grouped_ic['high_rate'] = grouped_ic.pop('stress')
    grouped_ic['low_rate'] = grouped_ic.pop('normal')
    results['grouped_ic'] = grouped_ic
    results['rate_threshold'] = rate_threshold

    print("\n[Grouped IC by Rate Regime]")
    for regime, stats in grouped_ic.items():
        sig = "***" if stats.get('p_value', 1) < 0.01 else "**" if stats.get('p_value', 1) < 0.05 else ""
        print(f"  {regime}: IC={stats['ic']:.3f}, n={stats['n_samples']} {sig}")

    # Interaction regression
    print("\n[Interaction Regression: R = α + β·F + γ·Rate + δ·(F×Rate)]")
    interaction_reg = run_interaction_regression(
        return_aligned,
        factor_aligned,
        ffr_aligned,
        use_hac=True,
        hac_lag=HAC_LAG
    )
    results['interaction_regression'] = interaction_reg

    if 'error' not in interaction_reg:
        print(f"\n  Coefficients (HAC t-stats):")
        for name in ['const', 'factor', 'condition', 'interaction']:
            coef = interaction_reg['coefficients'][name]
            t = interaction_reg['t_stats'][name]
            p = interaction_reg['p_values'][name]
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            print(f"    {name}: {coef:.6f} (t={t:.2f}, p={p:.4f}) {sig}")

        print(f"\n  R²: {interaction_reg.get('r_squared', 'N/A')}")

    results['ffr'] = ffr_aligned
    results['regime'] = rate_regime

    return results


def plot_rate_regime(results: dict, factor: pd.Series, save_path: str = None):
    """Plot rate regime analysis"""
    if 'error' in results:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('V1 ST Debt Ratio: Interest Rate Regime Analysis', fontsize=14, fontweight='bold')

    ffr = results['ffr']
    regime = results['regime']
    rate_threshold = results['rate_threshold']

    # Plot 1: FFR timeline
    ax1 = axes[0, 0]
    ax1.plot(ffr.index, ffr, 'b-', linewidth=1.5)
    ax1.axhline(rate_threshold, color='red', linestyle='--', label=f'Threshold={rate_threshold:.2f}%')
    ax1.set_title('1. Federal Funds Rate with Regime')
    ax1.set_ylabel('FFR (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Grouped IC
    ax2 = axes[0, 1]
    grouped_ic = results['grouped_ic']
    regimes = ['low_rate', 'high_rate', 'full']
    ics = [grouped_ic[r]['ic'] for r in regimes]
    colors = ['green' if ic < 0 else 'red' for ic in ics]
    ax2.bar(range(len(regimes)), ics, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_xticks(range(len(regimes)))
    ax2.set_xticklabels([f'Low Rate\n(FFR<{rate_threshold:.1f}%)',
                         f'High Rate\n(FFR>={rate_threshold:.1f}%)', 'Full Sample'])
    ax2.set_ylabel('Spearman IC')
    ax2.set_title('2. Grouped IC by Rate Regime')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Interaction coefficients
    ax3 = axes[1, 0]
    if 'error' not in results['interaction_regression']:
        reg = results['interaction_regression']
        names = ['factor', 'condition', 'interaction']
        coefs = [reg['coefficients'][n] for n in names]
        colors = ['green' if c < 0 else 'red' for c in coefs]
        ax3.bar(range(len(names)), coefs, color=colors, alpha=0.7, edgecolor='black')
        ax3.axhline(0, color='black', linewidth=0.5)
        ax3.set_xticks(range(len(names)))
        ax3.set_xticklabels(['β (Factor)', 'γ (Rate)', 'δ (F×Rate)'])
        ax3.set_ylabel('Coefficient')
        ax3.set_title('3. Interaction Regression Coefficients (HAC)')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')
    if 'error' not in results['interaction_regression']:
        reg = results['interaction_regression']
        interp = reg.get('interpretation', {})
        text = f"""
Interaction Regression Results (with HAC SE):

Model: R_12M = α + β·F + γ·Rate + δ·(F × Rate) + ε

Interaction coefficient (δ): {interp.get('interaction_coef', 'N/A')}
Significant at 5%: {interp.get('interaction_significant', 'N/A')}

Rate Threshold (Median): {rate_threshold:.2f}%

High Rate IC: {grouped_ic['high_rate']['ic']:.3f}
Low Rate IC: {grouped_ic['low_rate']['ic']:.3f}
"""
        ax4.text(0.05, 0.95, text, transform=ax4.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    ax4.set_title('4. Key Findings', fontsize=12, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()
    return fig


# ============== Step 3: Risk Target Analysis ==============

def run_risk_target_analysis(factor: pd.Series, spx: pd.Series) -> dict:
    """Test factor against risk targets"""
    print("\n" + "=" * 60)
    print("Step 3: Risk Target Variable Analysis")
    print("=" * 60)

    results = {}
    spx_monthly = spx.resample('ME').last()

    # Risk target IC
    print("\n[Risk Target IC]")
    risk_ic = compute_risk_target_ic(factor, spx, horizons=[126, 252])
    results['risk_target_ic'] = risk_ic

    for horizon, targets in risk_ic.items():
        print(f"\n  {horizon} Horizon:")
        for target, stats in targets.items():
            sig = "***" if stats['p_value'] < 0.01 else "**" if stats['p_value'] < 0.05 else ""
            print(f"    {target}: IC={stats['ic']:.3f}, p={stats['p_value']:.4f} {sig}")

    # Drawdown AUC
    print("\n[Drawdown Event Prediction (AUC)]")
    auc_results = {}
    for threshold in [-0.10, -0.15, -0.20]:
        auc = compute_drawdown_event_auc(factor, spx, threshold=threshold)
        auc_results[f'MDD<{int(threshold*100)}%'] = auc
        print(f"  MDD<{int(threshold*100)}%: AUC={auc['auc']:.3f}, "
              f"Events={auc['n_events']}/{auc['n_samples']} ({auc['event_rate']*100:.1f}%)")
    results['auc'] = auc_results

    # Tail quantile
    print("\n[Tail Quantile Analysis (5% quantile)]")
    tail_results = compute_tail_quantile_ic(factor, spx_monthly, horizon=12, quantile=0.05)
    results['tail_quantile'] = tail_results

    if 'interpretation' in tail_results:
        interp = tail_results['interpretation']
        print(f"  Mean regression β: {interp['mean_beta']:.4f}")
        print(f"  5% quantile β: {interp['quantile_beta']:.4f}")
        ratio = abs(interp['quantile_beta'] / interp['mean_beta']) if interp['mean_beta'] != 0 else 0
        print(f"  Tail Ratio: {ratio:.2f}")
        results['tail_ratio'] = ratio

    return results


def plot_risk_target_analysis(results: dict, factor: pd.Series, spx: pd.Series, save_path: str = None):
    """Plot risk target analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('V1 ST Debt Ratio: Risk Target Analysis', fontsize=14, fontweight='bold')

    # Plot 1: Risk target IC
    ax1 = axes[0, 0]
    if results['risk_target_ic']:
        targets = []
        ics = []
        for horizon in ['6M', '12M']:
            if horizon in results['risk_target_ic']:
                for target in ['ic_return', 'ic_max_drawdown', 'ic_volatility']:
                    ic = results['risk_target_ic'][horizon].get(target, {}).get('ic', np.nan)
                    targets.append(f'{horizon}\n{target.replace("ic_", "")}')
                    ics.append(ic)
        colors = ['green' if ic < 0 else 'red' if not np.isnan(ic) else 'gray' for ic in ics]
        ax1.bar(range(len(targets)), ics, color=colors, alpha=0.7, edgecolor='black')
        ax1.axhline(0, color='black', linewidth=0.5)
        ax1.set_xticks(range(len(targets)))
        ax1.set_xticklabels(targets, rotation=45, ha='right')
    ax1.set_ylabel('Spearman IC')
    ax1.set_title('1. IC vs Different Risk Targets')
    ax1.grid(True, alpha=0.3)

    # Plot 2: ROC curves
    ax2 = axes[0, 1]
    colors_roc = ['blue', 'green', 'red']
    for i, (event_name, auc_data) in enumerate(results['auc'].items()):
        if 'fpr' in auc_data and 'tpr' in auc_data:
            ax2.plot(auc_data['fpr'], auc_data['tpr'],
                     color=colors_roc[i % len(colors_roc)],
                     linewidth=2,
                     label=f'{event_name} (AUC={auc_data["auc"]:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('2. ROC Curves for Drawdown Prediction')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Tail quantile comparison
    ax3 = axes[1, 0]
    if 'tail_quantile' in results and 'interpretation' in results['tail_quantile']:
        interp = results['tail_quantile']['interpretation']
        betas = [interp['mean_beta'], interp['quantile_beta']]
        labels = ['Mean β', '5% Quantile β']
        colors = ['blue', 'red']
        ax3.bar(range(len(labels)), betas, color=colors, alpha=0.7, edgecolor='black')
        ax3.axhline(0, color='black', linewidth=0.5)
        ax3.set_xticks(range(len(labels)))
        ax3.set_xticklabels(labels)
        ax3.set_ylabel('Regression Coefficient')
        ax3.set_title('3. Mean vs Tail Effect (5% Quantile)')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    best_auc = max(results['auc'].items(), key=lambda x: x[1]['auc'])
    tail_ratio = results.get('tail_ratio', 'N/A')
    tail_ratio_str = f"{tail_ratio:.2f}" if isinstance(tail_ratio, (int, float)) else str(tail_ratio)
    text = f"""
Risk Target Analysis Summary:

Best AUC: {best_auc[0]} = {best_auc[1]['auc']:.3f}
Tail Ratio: {tail_ratio_str}

Interpretation:
- AUC > 0.60: Meaningful crash prediction
- Tail Ratio > 1.5: Stronger effect on extreme downside
"""
    ax4.text(0.05, 0.95, text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))
    ax4.set_title('4. Summary', fontsize=12, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()
    return fig


# ============== Step 4: Quintile Analysis ==============

def run_quintile_analysis_step(factor: pd.Series, spx: pd.Series) -> dict:
    """Run quintile analysis"""
    print("\n" + "=" * 60)
    print("Step 4: Quintile Analysis")
    print("=" * 60)

    spx_monthly = spx.resample('ME').last()
    fwd_return_12m = np.log(spx_monthly.shift(-13) / spx_monthly.shift(-1))

    fwd_mdd = compute_forward_max_drawdown(spx, horizon=252)
    fwd_mdd_monthly = fwd_mdd.resample('ME').last()
    event = (fwd_mdd_monthly < -0.15).astype(int)

    common_idx = factor.index.intersection(fwd_return_12m.index).intersection(event.index)

    results = run_quintile_analysis(
        factor.loc[common_idx],
        fwd_return_12m.loc[common_idx],
        event.loc[common_idx],
        n_quantiles=5
    )

    if 'error' not in results:
        print("\n[Quintile Statistics]")
        print(f"{'Quintile':<10} {'Mean Factor':<12} {'Mean Return':<12} {'Crash Rate':<12} {'N':<6}")
        print("-" * 54)
        for q, stats in results['quintile_stats'].items():
            print(f"{q:<10} {stats['mean_factor']:>10.2f}%  {stats['mean_return']*100:>10.2f}%  "
                  f"{stats.get('event_rate', 0)*100:>10.1f}%  {stats['n_samples']:<6}")

        print(f"\n[Monotonicity Test]")
        mono = results['monotonicity']
        print(f"  Spearman corr: {mono['spearman_corr']:.3f}")
        print(f"  Q5-Q1 spread: {mono['q5_minus_q1']*100:.2f}%")

    return results


def plot_quintile_analysis(results: dict, save_path: str = None):
    """Plot quintile analysis"""
    if 'error' in results:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('V1 ST Debt Ratio: Quintile Analysis', fontsize=14, fontweight='bold')

    quintiles = list(results['quintile_stats'].keys())

    # Plot 1: Mean return
    ax1 = axes[0]
    returns = [results['quintile_stats'][q]['mean_return'] * 100 for q in quintiles]
    colors = ['green' if r > 0 else 'red' for r in returns]
    ax1.bar(range(len(quintiles)), returns, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_xticks(range(len(quintiles)))
    ax1.set_xticklabels(quintiles)
    ax1.set_xlabel('Factor Quintile (Q1=Low, Q5=High)')
    ax1.set_ylabel('Mean 12M Forward Return (%)')
    ax1.set_title('Mean Return by Factor Quintile')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Crash rate
    ax2 = axes[1]
    crash_rates = [results['quintile_stats'][q].get('event_rate', 0) * 100 for q in quintiles]
    ax2.bar(range(len(quintiles)), crash_rates, color='orange', alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(quintiles)))
    ax2.set_xticklabels(quintiles)
    ax2.set_xlabel('Factor Quintile (Q1=Low, Q5=High)')
    ax2.set_ylabel('Crash Rate (MDD > 15%)')
    ax2.set_title('Crash Probability by Factor Quintile')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()
    return fig


# ============== Step 5: Bootstrap Robustness ==============

def run_bootstrap_analysis(factor: pd.Series, spx: pd.Series, ffr: pd.Series = None) -> dict:
    """Run block bootstrap for robustness"""
    print("\n" + "=" * 60)
    print("Step 5: Bootstrap Robustness Test")
    print("=" * 60)

    spx_monthly = spx.resample('ME').last()
    fwd_return_12m = np.log(spx_monthly.shift(-13) / spx_monthly.shift(-1))

    common_idx = factor.index.intersection(fwd_return_12m.index)
    factor_aligned = factor.loc[common_idx]
    return_aligned = fwd_return_12m.loc[common_idx]

    results = {}

    # Full sample bootstrap
    print("\n[Full Sample Bootstrap]")
    try:
        full_bootstrap = block_bootstrap_regression(return_aligned, factor_aligned, n_bootstrap=1000, block_size=12)
        results['full_sample'] = full_bootstrap
        print(f"  β: {full_bootstrap['original_beta']:.4f}")
        print(f"  95% CI: [{full_bootstrap['ci_lower']:.4f}, {full_bootstrap['ci_upper']:.4f}]")
        print(f"  p-value: {full_bootstrap['p_value']:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
        results['full_sample'] = {'error': str(e)}

    # High rate period bootstrap
    if ffr is not None:
        print("\n[High Rate Period Bootstrap]")
        try:
            common_idx_rate = factor.index.intersection(fwd_return_12m.index).intersection(ffr.index)
            factor_rate = factor.loc[common_idx_rate]
            return_rate = fwd_return_12m.loc[common_idx_rate]
            ffr_rate = ffr.loc[common_idx_rate]

            rate_threshold = ffr_rate.median()
            high_rate_mask = ffr_rate > rate_threshold

            factor_high = factor_rate[high_rate_mask]
            return_high = return_rate[high_rate_mask]

            high_bootstrap = block_bootstrap_regression(return_high, factor_high, n_bootstrap=1000, block_size=12)
            results['high_rate'] = high_bootstrap
            print(f"  β: {high_bootstrap['original_beta']:.4f}")
            print(f"  95% CI: [{high_bootstrap['ci_lower']:.4f}, {high_bootstrap['ci_upper']:.4f}]")
            print(f"  p-value: {high_bootstrap['p_value']:.4f}")
        except Exception as e:
            print(f"  Error: {e}")
            results['high_rate'] = {'error': str(e)}

    return results


# ============== Report Generation ==============

def generate_summary_report(all_results: dict, factor_df: pd.DataFrame, output_dir: str):
    """Generate V1_SUMMARY.md report with 5Y and 10Y comparison"""

    current_value = factor_df['factor_raw'].iloc[-1]
    current_pctl_5y = factor_df['factor_pctl_5Y'].iloc[-1] if 'factor_pctl_5Y' in factor_df else 'N/A'
    current_pctl_10y = factor_df['factor_pctl_10Y'].iloc[-1] if 'factor_pctl_10Y' in factor_df else 'N/A'

    report = f"""# V1 ST Debt Ratio 因子研究总结

## 因子定义

**Series**: `{FACTOR_SERIES}` - Nonfinancial Corporate Business; Short-Term Debt as a Percentage of Total Debt

| 属性 | 值 |
|------|-----|
| Units | Percent |
| Frequency | Quarterly |
| Data Start | 1945 |
| Release Lag | {RELEASE_LAG_MONTHS} months (based on ALFRED) |
| Percentile Windows | 5Y (20Q), 10Y (40Q) |

**关键特性**: 高值 = 高短期债务依赖 = 潜在高再融资风险

---

## 当前市场状态

| 指标 | 值 |
|------|-----|
| **当前 ST Debt Ratio** | **{current_value:.2f}%** |
| **5Y 百分位** | **{current_pctl_5y:.1f}%** |
| **10Y 百分位** | **{current_pctl_10y:.1f}%** |

---

## 核心结论: 5Y vs 10Y Window 对比

### IC (Information Coefficient)

| Window | Full IC | High Rate IC | Low Rate IC |
|--------|---------|--------------|-------------|
"""

    # Add IC comparison for both windows
    for window in ['5Y', '10Y']:
        if window in all_results and 'rate_regime' in all_results[window]:
            rr = all_results[window]['rate_regime']
            if 'grouped_ic' in rr:
                full_ic = rr['grouped_ic']['full']['ic']
                high_ic = rr['grouped_ic']['high_rate']['ic']
                low_ic = rr['grouped_ic']['low_rate']['ic']
                report += f"| {window} | {full_ic:.3f} | {high_ic:.3f} | {low_ic:.3f} |\n"

    report += "\n### Drawdown AUC\n\n"
    report += "| Window | MDD<-10% | MDD<-15% | MDD<-20% |\n"
    report += "|--------|----------|----------|----------|\n"

    for window in ['5Y', '10Y']:
        if window in all_results and 'risk_target' in all_results[window]:
            auc = all_results[window]['risk_target']['auc']
            auc_10 = auc.get('MDD<-10%', {}).get('auc', 'N/A')
            auc_15 = auc.get('MDD<-15%', {}).get('auc', 'N/A')
            auc_20 = auc.get('MDD<-20%', {}).get('auc', 'N/A')
            report += f"| {window} | {auc_10:.3f} | {auc_15:.3f} | {auc_20:.3f} |\n"

    report += "\n### Quintile Analysis\n\n"
    report += "| Window | Spearman | Q5-Q1 Spread |\n"
    report += "|--------|----------|-------------|\n"

    for window in ['5Y', '10Y']:
        if window in all_results and 'quintile' in all_results[window]:
            mono = all_results[window]['quintile'].get('monotonicity', {})
            spearman = mono.get('spearman_corr', 'N/A')
            spread = mono.get('q5_minus_q1', 0) * 100
            if isinstance(spearman, float):
                report += f"| {window} | {spearman:.3f} | {spread:.2f}% |\n"

    # Detailed results for best window
    best_window = '10Y'  # Default to 10Y
    for window in ['5Y', '10Y']:
        if window in all_results and 'rate_regime' in all_results[window]:
            if abs(all_results[window]['rate_regime']['grouped_ic']['full']['ic']) > abs(all_results.get(best_window, {}).get('rate_regime', {}).get('grouped_ic', {}).get('full', {}).get('ic', 0)):
                best_window = window

    report += f"\n---\n\n## 详细结果 ({best_window} Window)\n\n"

    if best_window in all_results and 'quintile' in all_results[best_window]:
        report += "### Quintile Details\n\n"
        report += "| Quintile | Mean Factor | Mean Return | Crash Rate |\n"
        report += "|----------|-------------|-------------|------------|\n"
        for q, stats in all_results[best_window]['quintile']['quintile_stats'].items():
            report += f"| {q} | {stats['mean_factor']:.1f}% | {stats['mean_return']*100:.2f}% | {stats.get('event_rate', 0)*100:.1f}% |\n"

    report += f"""
---

## 验证状态

| 检验项 | 5Y | 10Y |
|--------|-----|-----|
"""

    # Validation comparison
    for metric in ['Full IC', 'Best AUC', 'Spearman']:
        row = f"| {metric} |"
        for window in ['5Y', '10Y']:
            if window in all_results:
                if metric == 'Full IC' and 'rate_regime' in all_results[window]:
                    val = all_results[window]['rate_regime']['grouped_ic']['full']['ic']
                    status = '✓' if abs(val) > 0.15 else '△'
                    row += f" {val:.3f} {status} |"
                elif metric == 'Best AUC' and 'risk_target' in all_results[window]:
                    val = max([v['auc'] for v in all_results[window]['risk_target']['auc'].values()])
                    status = '✓' if val > 0.60 else '✗'
                    row += f" {val:.3f} {status} |"
                elif metric == 'Spearman' and 'quintile' in all_results[window]:
                    val = all_results[window]['quintile']['monotonicity']['spearman_corr']
                    status = '✓' if abs(val) > 0.7 else '△' if abs(val) > 0.4 else '✗'
                    row += f" {val:.3f} {status} |"
                else:
                    row += " N/A |"
            else:
                row += " N/A |"
        report += row + "\n"

    report += f"""
---

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    filepath = os.path.join(output_dir, 'V1_SUMMARY.md')
    with open(filepath, 'w') as f:
        f.write(report)
    print(f"\n  Saved: {filepath}")


# ============== Main ==============

def validate_single_window(factor: pd.Series, spx: pd.Series, window_name: str, output_dir: str) -> dict:
    """Run validation for a single percentile window"""
    print(f"\n{'='*60}")
    print(f"Validating {window_name} Window")
    print(f"{'='*60}")

    results = {}

    # Step 1: Structural Break
    results['structural_break'] = run_structural_break_analysis(factor, spx)

    # Step 2: Rate Regime
    results['rate_regime'] = run_rate_regime_analysis(factor, spx)

    # Step 3: Risk Target
    results['risk_target'] = run_risk_target_analysis(factor, spx)

    # Step 4: Quintile
    results['quintile'] = run_quintile_analysis_step(factor, spx)

    # Step 5: Bootstrap
    ffr = results['rate_regime'].get('ffr') if 'error' not in results['rate_regime'] else None
    results['bootstrap'] = run_bootstrap_analysis(factor, spx, ffr)

    return results


def main():
    """Main validation workflow - validates both 5Y and 10Y windows"""
    print("=" * 60)
    print("V1 (New): Corporate Short-Term Debt Ratio - Validation")
    print("=" * 60)
    print(f"\nFactor: {FACTOR_SERIES}")
    print("Nonfinancial Corporate Business; Short-Term Debt / Total Debt")
    print(f"Release Lag: {RELEASE_LAG_MONTHS} months (based on ALFRED)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    factor_df = load_factor_from_fred()
    spx = load_spx()
    print(f"\n  SPX data: {len(spx)} days")

    # Save factor data
    factor_df.to_csv(os.path.join(OUTPUT_DIR, 'all_methods_data.csv'))
    print(f"  Saved factor data to {OUTPUT_DIR}/all_methods_data.csv")

    # Step 0: Metadata
    metadata = run_metadata_audit()

    # Validate both windows
    all_results = {
        'metadata': metadata,
        'factor_df': factor_df,
    }

    for window_name in ['5Y', '10Y']:
        factor_col = f'factor_pctl_{window_name}'
        if factor_col in factor_df.columns:
            factor = factor_df[factor_col].dropna()
            results = validate_single_window(factor, spx, window_name, OUTPUT_DIR)
            all_results[window_name] = results

    # Generate comparison summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY: 5Y vs 10Y Window Comparison")
    print("=" * 60)

    print("\n[IC Comparison]")
    print(f"{'Window':<8} {'Full IC':<12} {'High Rate IC':<14} {'Low Rate IC':<14}")
    print("-" * 50)
    for window in ['5Y', '10Y']:
        if window in all_results and 'rate_regime' in all_results[window]:
            rr = all_results[window]['rate_regime']
            if 'grouped_ic' in rr:
                full_ic = rr['grouped_ic']['full']['ic']
                high_ic = rr['grouped_ic']['high_rate']['ic']
                low_ic = rr['grouped_ic']['low_rate']['ic']
                print(f"{window:<8} {full_ic:<12.3f} {high_ic:<14.3f} {low_ic:<14.3f}")

    print("\n[AUC Comparison (20% MDD)]")
    for window in ['5Y', '10Y']:
        if window in all_results and 'risk_target' in all_results[window]:
            auc_20 = all_results[window]['risk_target']['auc'].get('MDD<-20%', {}).get('auc', 'N/A')
            print(f"  {window}: AUC = {auc_20:.3f}" if isinstance(auc_20, float) else f"  {window}: AUC = {auc_20}")

    print("\n[Quintile Monotonicity]")
    for window in ['5Y', '10Y']:
        if window in all_results and 'quintile' in all_results[window]:
            mono = all_results[window]['quintile'].get('monotonicity', {})
            spearman = mono.get('spearman_corr', 'N/A')
            spread = mono.get('q5_minus_q1', 0) * 100
            print(f"  {window}: Spearman = {spearman:.3f}, Q5-Q1 = {spread:.2f}%" if isinstance(spearman, float) else f"  {window}: N/A")

    # Generate detailed report
    print("\n[Generating Summary Report]")
    generate_summary_report(all_results, factor_df, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
