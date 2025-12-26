"""
Information Coefficient (IC) Analysis Module
============================================

Provides functions for evaluating factor predictive power:
- Spearman IC (rank correlation)
- Pearson IC (linear correlation)
- ICIR (IC Information Ratio)
- Rolling IC analysis
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


class ICAnalyzer:
    """Information Coefficient Analyzer"""

    def __init__(self,
                 factor_series: pd.Series,
                 price_series: pd.Series):
        """
        Initialize IC Analyzer.

        Args:
            factor_series: Factor series (index = dates)
            price_series: Price series for return calculation (index = dates)
        """
        # Align data to common dates
        common_idx = factor_series.index.intersection(price_series.index)

        self.factor = factor_series.loc[common_idx].copy()
        self.prices = price_series.loc[common_idx].copy()

        # Sort by date
        self.factor = self.factor.sort_index()
        self.prices = self.prices.sort_index()

    def calculate_forward_returns(self,
                                   periods: List[int] = None) -> pd.DataFrame:
        """
        Calculate forward returns for different horizons.

        Args:
            periods: List of forward periods (trading days)
                Default: [21, 63, 126, 252] for 1M, 3M, 6M, 12M

        Returns:
            DataFrame with forward returns for each period
        """
        if periods is None:
            periods = [21, 63, 126, 252]

        forward_returns = {}

        for period in periods:
            # Forward log return
            fr = np.log(self.prices.shift(-period) / self.prices)
            forward_returns[f'{period}D'] = fr

        return pd.DataFrame(forward_returns)

    def compute_period_ic(self,
                          forward_return: pd.Series,
                          method: str = 'spearman') -> Tuple[float, float]:
        """
        Compute IC for a single period.

        Args:
            forward_return: Forward return series
            method: 'spearman' or 'pearson'

        Returns:
            (IC value, p-value)
        """
        # Align and remove NaN
        valid = pd.DataFrame({
            'factor': self.factor,
            'return': forward_return
        }).dropna()

        if len(valid) < 10:
            return np.nan, np.nan

        if method == 'spearman':
            ic, pvalue = spearmanr(valid['factor'], valid['return'])
        else:
            ic, pvalue = pearsonr(valid['factor'], valid['return'])

        return ic, pvalue

    def compute_rolling_ic(self,
                           forward_return: pd.Series,
                           window: int = 120,
                           method: str = 'spearman') -> pd.Series:
        """
        Compute rolling IC.

        Args:
            forward_return: Forward return series
            window: Rolling window size (periods)
            method: 'spearman' or 'pearson'

        Returns:
            Rolling IC series
        """
        valid = pd.DataFrame({
            'factor': self.factor,
            'return': forward_return
        }).dropna()

        rolling_ic = pd.Series(index=valid.index, dtype=float)

        for i in range(window, len(valid)):
            window_data = valid.iloc[i - window:i]

            if method == 'spearman':
                ic, _ = spearmanr(window_data['factor'], window_data['return'])
            else:
                ic, _ = pearsonr(window_data['factor'], window_data['return'])

            rolling_ic.iloc[i] = ic

        return rolling_ic

    def full_ic_analysis(self,
                         periods: List[int] = None,
                         rolling_window: int = 120) -> Dict:
        """
        Perform full IC analysis.

        Args:
            periods: Forward return periods (trading days)
            rolling_window: Window for rolling IC calculation

        Returns:
            Dictionary containing:
            - summary: IC statistics for each period
            - rolling_ic: Rolling IC series for each period
        """
        if periods is None:
            periods = [21, 63, 126, 252]

        period_names = {
            21: '1M',
            63: '3M',
            126: '6M',
            252: '12M'
        }

        results = {
            'summary': {},
            'rolling_ic': {}
        }

        for period in periods:
            period_name = period_names.get(period, f'{period}D')

            # Calculate forward return
            forward_return = np.log(self.prices.shift(-period) / self.prices)

            # Full sample IC
            valid = pd.DataFrame({
                'factor': self.factor,
                'return': forward_return
            }).dropna()

            if len(valid) < 20:
                continue

            spearman_ic, spearman_pval = spearmanr(
                valid['factor'], valid['return']
            )
            pearson_ic, pearson_pval = pearsonr(
                valid['factor'], valid['return']
            )

            # Rolling IC
            rolling_ic_series = self.compute_rolling_ic(
                forward_return,
                window=rolling_window,
                method='spearman'
            )

            # Calculate ICIR
            rolling_ic_valid = rolling_ic_series.dropna()
            if len(rolling_ic_valid) > 0:
                mean_ic = rolling_ic_valid.mean()
                std_ic = rolling_ic_valid.std()
                icir = mean_ic / std_ic if std_ic > 0 else 0
            else:
                mean_ic = std_ic = icir = np.nan

            results['summary'][period_name] = {
                'spearman_ic': spearman_ic,
                'spearman_pval': spearman_pval,
                'pearson_ic': pearson_ic,
                'pearson_pval': pearson_pval,
                'mean_rolling_ic': mean_ic,
                'std_rolling_ic': std_ic,
                'icir': icir,
                'n_samples': len(valid)
            }

            results['rolling_ic'][period_name] = rolling_ic_series

        return results

    def plot_ic_analysis(self,
                         ic_results: Dict,
                         title: str = 'IC Analysis',
                         save_path: str = None) -> plt.Figure:
        """
        Plot IC analysis results.

        Args:
            ic_results: Results from full_ic_analysis
            title: Plot title
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=14, fontweight='bold')

        period_names = list(ic_results['summary'].keys())

        # Plot 1: IC Summary Bar Chart
        ax1 = axes[0, 0]
        x = range(len(period_names))
        spearman_ics = [ic_results['summary'][p]['spearman_ic']
                        for p in period_names]
        pearson_ics = [ic_results['summary'][p]['pearson_ic']
                       for p in period_names]

        width = 0.35
        ax1.bar([i - width / 2 for i in x], spearman_ics, width,
                label='Spearman IC', color='steelblue')
        ax1.bar([i + width / 2 for i in x], pearson_ics, width,
                label='Pearson IC', color='coral')
        ax1.axhline(0, color='black', linewidth=0.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels(period_names)
        ax1.set_ylabel('IC')
        ax1.set_title('1. IC by Forward Return Period')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: ICIR Bar Chart
        ax2 = axes[0, 1]
        icirs = [ic_results['summary'][p]['icir'] for p in period_names]
        colors = ['green' if ic > 0.5 else 'orange' if ic > 0 else 'red'
                  for ic in icirs]
        ax2.bar(x, icirs, color=colors)
        ax2.axhline(0.5, color='green', linestyle='--', alpha=0.7,
                    label='ICIR=0.5 (Good)')
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(period_names)
        ax2.set_ylabel('ICIR')
        ax2.set_title('2. ICIR (IC / Std(IC))')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Rolling IC (3M)
        ax3 = axes[1, 0]
        if '3M' in ic_results['rolling_ic']:
            rolling_3m = ic_results['rolling_ic']['3M'].dropna()
            if len(rolling_3m) > 0:
                ax3.plot(rolling_3m.index, rolling_3m.values,
                         color='steelblue', alpha=0.8)
                ax3.axhline(0, color='black', linewidth=0.5)
                mean_ic = ic_results['summary']['3M']['mean_rolling_ic']
                ax3.axhline(mean_ic, color='red', linestyle='--',
                            label=f'Mean IC={mean_ic:.3f}')
                ax3.fill_between(rolling_3m.index, 0, rolling_3m.values,
                                 where=rolling_3m.values > 0,
                                 alpha=0.3, color='green')
                ax3.fill_between(rolling_3m.index, 0, rolling_3m.values,
                                 where=rolling_3m.values < 0,
                                 alpha=0.3, color='red')
        ax3.set_ylabel('Rolling IC (10Y)')
        ax3.set_title('3. Rolling IC - 3M Forward Return')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Rolling IC (12M)
        ax4 = axes[1, 1]
        if '12M' in ic_results['rolling_ic']:
            rolling_12m = ic_results['rolling_ic']['12M'].dropna()
            if len(rolling_12m) > 0:
                ax4.plot(rolling_12m.index, rolling_12m.values,
                         color='steelblue', alpha=0.8)
                ax4.axhline(0, color='black', linewidth=0.5)
                mean_ic = ic_results['summary']['12M']['mean_rolling_ic']
                ax4.axhline(mean_ic, color='red', linestyle='--',
                            label=f'Mean IC={mean_ic:.3f}')
                ax4.fill_between(rolling_12m.index, 0, rolling_12m.values,
                                 where=rolling_12m.values > 0,
                                 alpha=0.3, color='green')
                ax4.fill_between(rolling_12m.index, 0, rolling_12m.values,
                                 where=rolling_12m.values < 0,
                                 alpha=0.3, color='red')
        ax4.set_ylabel('Rolling IC (10Y)')
        ax4.set_title('4. Rolling IC - 12M Forward Return')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig


def format_ic_summary(ic_results: Dict) -> str:
    """
    Format IC results as a readable string.

    Args:
        ic_results: Results from full_ic_analysis

    Returns:
        Formatted string
    """
    lines = []
    lines.append("IC Analysis Summary")
    lines.append("=" * 50)
    lines.append(f"{'Period':<8} {'Spearman IC':<12} {'ICIR':<8} {'N':<6}")
    lines.append("-" * 50)

    for period, stats in ic_results['summary'].items():
        lines.append(
            f"{period:<8} {stats['spearman_ic']:>10.4f}   "
            f"{stats['icir']:>6.3f}   {stats['n_samples']:<6}"
        )

    return "\n".join(lines)
