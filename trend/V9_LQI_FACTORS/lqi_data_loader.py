"""
LQI (Liquidity Quality Index) Data Loader

Loads and parses various liquidity/sentiment data sources from lqi_cache:
- VIX, VIX3M (volatility term structure)
- VVIX (vol of vol)
- SKEW (tail risk)
- EFFR, SOFR (funding rates)
- GCF Repo (repo rates)
- iShares ETF NAVs (LQD, HYG, TLT)
- NY Fed Primary Dealer Positions (dealer inventory)
"""

import os
import pandas as pd
import numpy as np
from lxml import etree
from typing import Optional, Dict

# Default cache path
LQI_CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'lqi_cache')


def parse_ishares_historical(filepath: str, include_shares: bool = False) -> pd.DataFrame:
    """
    Parse Historical sheet from iShares XML Excel file
    Returns DataFrame with Date index and NAV column (optionally Shares Outstanding)
    """
    with open(filepath, 'rb') as f:
        content = f.read()
        # Remove BOM if present
        if content.startswith(b'\xef\xbb\xbf\xef\xbb\xbf'):
            content = content[6:]
        elif content.startswith(b'\xef\xbb\xbf'):
            content = content[3:]

    parser = etree.XMLParser(recover=True)
    root = etree.fromstring(content, parser)
    ns = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}

    # Find Historical worksheet
    for ws in root.xpath('.//ss:Worksheet', namespaces=ns):
        name = ws.get('{urn:schemas-microsoft-com:office:spreadsheet}Name')
        if name == 'Historical':
            rows = []
            for table in ws.xpath('.//ss:Table', namespaces=ns):
                for row in table.xpath('ss:Row', namespaces=ns):
                    cells = []
                    for cell in row.xpath('ss:Cell', namespaces=ns):
                        data = cell.xpath('ss:Data', namespaces=ns)
                        if data:
                            cells.append(data[0].text)
                        else:
                            cells.append(None)
                    if cells:
                        rows.append(cells)

            if len(rows) > 1:
                # First row is header: As Of, NAV per Share, Ex-Dividends, Shares Outstanding
                df = pd.DataFrame(rows[1:], columns=rows[0])
                df['Date'] = pd.to_datetime(df['As Of'], format='%b %d, %Y')
                df['NAV'] = pd.to_numeric(df['NAV per Share'], errors='coerce')

                if include_shares and 'Shares Outstanding' in df.columns:
                    df['Shares'] = pd.to_numeric(
                        df['Shares Outstanding'].str.replace(',', ''),
                        errors='coerce'
                    )
                    df = df[['Date', 'NAV', 'Shares']].dropna()
                else:
                    df = df[['Date', 'NAV']].dropna()

                df = df.set_index('Date').sort_index()
                return df

    return pd.DataFrame()


class LQIDataLoader:
    """Loads all LQI data sources"""

    def __init__(self, cache_path: str = None):
        self.cache_path = cache_path or LQI_CACHE_PATH
        self._cache: Dict[str, pd.DataFrame] = {}

    def load_vix(self) -> pd.Series:
        """Load VIX (CBOE Volatility Index)"""
        if 'vix' not in self._cache:
            df = pd.read_parquet(os.path.join(self.cache_path, 'fred_VIXCLS.parquet'))
            self._cache['vix'] = df['vix']
        return self._cache['vix']

    def load_vix3m(self) -> pd.Series:
        """Load VIX3M (3-Month VIX)"""
        if 'vix3m' not in self._cache:
            df = pd.read_parquet(os.path.join(self.cache_path, 'fred_VXVCLS.parquet'))
            self._cache['vix3m'] = df['vix3m']
        return self._cache['vix3m']

    def load_vvix(self) -> pd.Series:
        """Load VVIX (VIX of VIX)"""
        if 'vvix' not in self._cache:
            df = pd.read_parquet(os.path.join(self.cache_path, 'cboe_vvix.parquet'))
            df['DATE'] = pd.to_datetime(df['DATE'])
            df = df.set_index('DATE')
            self._cache['vvix'] = df['VVIX']
        return self._cache['vvix']

    def load_skew(self) -> pd.Series:
        """Load CBOE SKEW Index"""
        if 'skew' not in self._cache:
            df = pd.read_parquet(os.path.join(self.cache_path, 'cboe_skew.parquet'))
            df['DATE'] = pd.to_datetime(df['DATE'])
            df = df.set_index('DATE')
            self._cache['skew'] = df['SKEW']
        return self._cache['skew']

    def load_effr(self) -> pd.Series:
        """Load EFFR (Effective Fed Funds Rate)"""
        if 'effr' not in self._cache:
            df = pd.read_parquet(os.path.join(self.cache_path, 'fred_EFFR.parquet'))
            self._cache['effr'] = df['value']
        return self._cache['effr']

    def load_sofr(self) -> pd.Series:
        """Load SOFR (Secured Overnight Financing Rate)"""
        if 'sofr' not in self._cache:
            df = pd.read_parquet(os.path.join(self.cache_path, 'fred_SOFR.parquet'))
            self._cache['sofr'] = df['value']
        return self._cache['sofr']

    def load_ted_spread(self) -> pd.Series:
        """Load TED Spread (discontinued 2022)"""
        if 'ted' not in self._cache:
            df = pd.read_parquet(os.path.join(self.cache_path, 'fred_TEDRATE.parquet'))
            self._cache['ted'] = df['value']
        return self._cache['ted']

    def load_gcf_repo(self) -> pd.DataFrame:
        """Load DTCC GCF Repo Index"""
        if 'gcf_repo' not in self._cache:
            df = pd.read_excel(
                os.path.join(self.cache_path, 'DTCC GCF Repo Index 2005-2024.xlsx'),
                skiprows=6
            )
            df.columns = ['Date', 'MBS_Rate', 'Treasury_Rate']
            df = df.dropna(subset=['Date'])
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
            df['MBS_Rate'] = pd.to_numeric(df['MBS_Rate'], errors='coerce')
            df['Treasury_Rate'] = pd.to_numeric(df['Treasury_Rate'], errors='coerce')
            self._cache['gcf_repo'] = df
        return self._cache['gcf_repo']

    def load_lqd_nav(self) -> pd.Series:
        """Load LQD (Investment Grade Corporate Bond) NAV"""
        if 'lqd' not in self._cache:
            filepath = os.path.join(self.cache_path, 'iShares iBoxx Investment Grade Corporate Bond ETF.xls')
            df = parse_ishares_historical(filepath)
            self._cache['lqd'] = df['NAV']
        return self._cache['lqd']

    def load_hyg_nav(self) -> pd.Series:
        """Load HYG (High Yield Corporate Bond) NAV"""
        if 'hyg' not in self._cache:
            filepath = os.path.join(self.cache_path, 'iShares HYG Fund.xls')
            df = parse_ishares_historical(filepath)
            self._cache['hyg'] = df['NAV']
        return self._cache['hyg']

    def load_tlt_nav(self) -> pd.Series:
        """Load TLT (20+ Year Treasury Bond) NAV"""
        if 'tlt' not in self._cache:
            filepath = os.path.join(self.cache_path, 'iShares 20 Year Treasury Bond ETF.xls')
            df = parse_ishares_historical(filepath)
            self._cache['tlt'] = df['NAV']
        return self._cache['tlt']

    def load_lqd_full(self) -> pd.DataFrame:
        """Load LQD with NAV and Shares Outstanding"""
        if 'lqd_full' not in self._cache:
            filepath = os.path.join(self.cache_path, 'iShares iBoxx Investment Grade Corporate Bond ETF.xls')
            self._cache['lqd_full'] = parse_ishares_historical(filepath, include_shares=True)
        return self._cache['lqd_full']

    def load_hyg_full(self) -> pd.DataFrame:
        """Load HYG with NAV and Shares Outstanding"""
        if 'hyg_full' not in self._cache:
            filepath = os.path.join(self.cache_path, 'iShares HYG Fund.xls')
            self._cache['hyg_full'] = parse_ishares_historical(filepath, include_shares=True)
        return self._cache['hyg_full']

    def load_tlt_full(self) -> pd.DataFrame:
        """Load TLT with NAV and Shares Outstanding"""
        if 'tlt_full' not in self._cache:
            filepath = os.path.join(self.cache_path, 'iShares 20 Year Treasury Bond ETF.xls')
            self._cache['tlt_full'] = parse_ishares_historical(filepath, include_shares=True)
        return self._cache['tlt_full']

    def load_spx(self) -> pd.Series:
        """Load SPX daily price"""
        if 'spx' not in self._cache:
            filepath = os.path.join(self.cache_path, 'SPX 1D Data.csv')
            df = pd.read_csv(filepath)
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time').sort_index()
            self._cache['spx'] = df['close']
        return self._cache['spx']

    def load_dealer_inventory(self) -> pd.DataFrame:
        """
        Load NY Fed Primary Dealer Positions (Treasury holdings by maturity)

        Returns DataFrame with columns:
        - Short: Bills + <=3Y coupons
        - Mid: 3-6Y coupons
        - Long: 6Y+ coupons
        - Total: Total holdings

        Data is weekly, from 2001 to present
        """
        if 'dealer_inventory' not in self._cache:
            dfs = []

            # SBP2013 format (2001-2013) - from xlsx files with maturity buckets
            # PDPUSGTBNOP: T-Bills
            # PDPUSGCS3LNOP: Coupon <=3Y
            # PDPUSGCS36NOP: Coupon 3-6Y
            # PDPUSGCS611NOP: Coupon 6-11Y
            # PDPUSGCSM11NOP: Coupon >11Y
            tbills_file = os.path.join(self.cache_path, 'nyfed_SBP2013_tbills.xlsx')
            coupon_file = os.path.join(self.cache_path, 'nyfed_SBP2013_coupon_buckets.xlsx')

            if os.path.exists(tbills_file) and os.path.exists(coupon_file):
                # Load T-Bills
                tbills_df = pd.read_excel(tbills_file)
                tbills_df['As Of Date'] = pd.to_datetime(tbills_df['As Of Date'])
                tbills = tbills_df.pivot(index='As Of Date', columns='Time Series', values='Value (millions)')

                # Load Coupon buckets
                coupon_df = pd.read_excel(coupon_file)
                coupon_df['As Of Date'] = pd.to_datetime(coupon_df['As Of Date'])
                coupons = coupon_df.pivot(index='As Of Date', columns='Time Series', values='Value (millions)')

                # Merge
                legacy = tbills.join(coupons, how='outer')

                # Short = T-Bills + <=3Y coupons
                # Mid = 3-6Y coupons
                # Long = 6-11Y + >11Y coupons
                legacy['Short'] = legacy.get('PDPUSGTBNOP', 0).fillna(0) + legacy.get('PDPUSGCS3LNOP', 0).fillna(0)
                legacy['Mid'] = legacy.get('PDPUSGCS36NOP', 0).fillna(0)
                legacy['Long'] = legacy.get('PDPUSGCS611NOP', 0).fillna(0) + legacy.get('PDPUSGCSM11NOP', 0).fillna(0)
                legacy['Total'] = legacy['Short'] + legacy['Mid'] + legacy['Long']
                legacy.index.name = 'date'

                dfs.append(legacy[['Short', 'Mid', 'Long', 'Total']])

            # SBN2013 format (2013-2014) - has maturity buckets but different columns
            files_old_format = [
                'nyfed_api_SBN2013_PDPOSGS-B_PDPOSGSC-G11L21_PDPOSGSC-G21_PDPOSGSC-G2.parquet',
                'nyfed_api_SBN2015_PDPOSGS-B_PDPOSGSC-G11L21_PDPOSGSC-G21_PDPOSGSC-G2.parquet',
            ]

            for fname in files_old_format:
                fpath = os.path.join(self.cache_path, fname)
                if os.path.exists(fpath):
                    df = pd.read_parquet(fpath)
                    # Old format: no G11L21, no G21 columns
                    # Short = Bills (PDPOSGS-B) + <=2Y (PDPOSGSC-L2)
                    # Mid = 2-7Y (G2L3 + G3L6 + G6L7)
                    # Long = 7Y+ (G7L11)
                    short = df.get('PDPOSGS-B', 0) + df.get('PDPOSGSC-L2', 0)
                    mid = df.get('PDPOSGSC-G2L3', 0) + df.get('PDPOSGSC-G3L6', 0) + df.get('PDPOSGSC-G6L7', 0)
                    long = df.get('PDPOSGSC-G7L11', 0)
                    total = df.get('PDPOSGST-TOT', short + mid + long)

                    result = pd.DataFrame({
                        'Short': short,
                        'Mid': mid,
                        'Long': long,
                        'Total': total
                    }, index=df.index)
                    dfs.append(result)

            # New format (2022+) - has all maturity buckets including G11L21 and G21
            files_new_format = [
                'nyfed_api_SBN2022_PDPOSGS-B_PDPOSGSC-G11L21_PDPOSGSC-G21_PDPOSGSC-G2.parquet',
                'nyfed_api_SBN2024_PDPOSGS-B_PDPOSGSC-G11L21_PDPOSGSC-G21_PDPOSGSC-G2.parquet',
            ]

            for fname in files_new_format:
                fpath = os.path.join(self.cache_path, fname)
                if os.path.exists(fpath):
                    df = pd.read_parquet(fpath)
                    # New format has G11L21 and G21
                    # Short = Bills + <=2Y
                    # Mid = 2-7Y (G2L3 + G3L6 + G6L7)
                    # Long = 7Y+ (G7L11 + G11L21 + G21)
                    short = df.get('PDPOSGS-B', 0) + df.get('PDPOSGSC-L2', 0)
                    mid = df.get('PDPOSGSC-G2L3', 0) + df.get('PDPOSGSC-G3L6', 0) + df.get('PDPOSGSC-G6L7', 0)
                    long = df.get('PDPOSGSC-G7L11', 0) + df.get('PDPOSGSC-G11L21', 0) + df.get('PDPOSGSC-G21', 0)
                    total = df.get('PDPOSGST-TOT', short + mid + long)

                    result = pd.DataFrame({
                        'Short': short,
                        'Mid': mid,
                        'Long': long,
                        'Total': total
                    }, index=df.index)
                    dfs.append(result)

            # Concatenate and remove duplicates
            if dfs:
                combined = pd.concat(dfs)
                combined = combined[~combined.index.duplicated(keep='last')]
                combined = combined.sort_index()
                self._cache['dealer_inventory'] = combined
            else:
                self._cache['dealer_inventory'] = pd.DataFrame()

        return self._cache['dealer_inventory']

    def load_gcf_repo_full(self) -> pd.DataFrame:
        """
        Load DTCC GCF Repo Index with 2025 data merged

        Returns DataFrame with Treasury_Rate column
        """
        if 'gcf_repo_full' not in self._cache:
            # Load 2005-2024 xlsx
            df_xlsx = pd.read_excel(
                os.path.join(self.cache_path, 'DTCC GCF Repo Index 2005-2024.xlsx'),
                skiprows=6
            )
            df_xlsx.columns = ['Date', 'MBS_Rate', 'Treasury_Rate']
            df_xlsx = df_xlsx.dropna(subset=['Date'])
            df_xlsx['Date'] = pd.to_datetime(df_xlsx['Date'])
            df_xlsx = df_xlsx.set_index('Date').sort_index()
            df_xlsx['MBS_Rate'] = pd.to_numeric(df_xlsx['MBS_Rate'], errors='coerce')
            df_xlsx['Treasury_Rate'] = pd.to_numeric(df_xlsx['Treasury_Rate'], errors='coerce')

            # Load 2025 csv
            csv_path = os.path.join(self.cache_path, 'DTCC GCF Repo Index 2025.csv')
            if os.path.exists(csv_path):
                df_csv = pd.read_csv(csv_path, encoding='latin-1')
                # Find Treasury Rate column (has weird characters)
                treas_col = [c for c in df_csv.columns if 'Treasury' in c and 'Rate' in c][0]
                mbs_col = [c for c in df_csv.columns if 'MBS' in c and 'Rate' in c][0]

                df_csv = df_csv[['Date', mbs_col, treas_col]].copy()
                df_csv.columns = ['Date', 'MBS_Rate', 'Treasury_Rate']
                df_csv = df_csv[df_csv['Date'] != 'TRAILER']  # Remove trailer row
                df_csv['Date'] = pd.to_datetime(df_csv['Date'], format='%m/%d/%y')
                df_csv = df_csv.set_index('Date').sort_index()
                df_csv['MBS_Rate'] = pd.to_numeric(df_csv['MBS_Rate'], errors='coerce')
                df_csv['Treasury_Rate'] = pd.to_numeric(df_csv['Treasury_Rate'], errors='coerce')

                # Merge
                combined = pd.concat([df_xlsx, df_csv])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined = combined.sort_index()
                self._cache['gcf_repo_full'] = combined
            else:
                self._cache['gcf_repo_full'] = df_xlsx

        return self._cache['gcf_repo_full']

    def summary(self) -> None:
        """Print summary of all available data"""
        print("=" * 70)
        print("LQI Data Summary")
        print("=" * 70)

        datasets = [
            ('VIX', self.load_vix),
            ('VIX3M', self.load_vix3m),
            ('VVIX', self.load_vvix),
            ('SKEW', self.load_skew),
            ('EFFR', self.load_effr),
            ('SOFR', self.load_sofr),
            ('TED Spread', self.load_ted_spread),
            ('LQD NAV', self.load_lqd_nav),
            ('HYG NAV', self.load_hyg_nav),
            ('TLT NAV', self.load_tlt_nav),
            ('SPX', self.load_spx),
        ]

        for name, loader in datasets:
            try:
                data = loader()
                print(f"\n{name}:")
                print(f"  Range: {data.index.min()} to {data.index.max()}")
                print(f"  Count: {len(data)}")
                print(f"  Latest: {data.iloc[-1]:.2f}")
            except Exception as e:
                print(f"\n{name}: Error - {e}")

        # GCF Repo (DataFrame)
        try:
            gcf = self.load_gcf_repo()
            print(f"\nGCF Repo:")
            print(f"  Range: {gcf.index.min()} to {gcf.index.max()}")
            print(f"  Latest Treasury Rate: {gcf['Treasury_Rate'].iloc[-1]:.3f}%")
        except Exception as e:
            print(f"\nGCF Repo: Error - {e}")


if __name__ == '__main__':
    loader = LQIDataLoader()
    loader.summary()
