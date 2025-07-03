import pandas as pd
import numpy as np
from xverse.transformer import MonotonicBinning, WOE
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas.core.algorithms as algos

algos.quantile = np.quantile


def compute_woe_iv(series: pd.Series,
                   target: pd.Series,
                   bins: int = 5,
                   strategy: str = 'quantile'):
    """
    Compute Weight of Evidence (WOE) and Information Value (IV) for a feature.
    
    Parameters:
    - series: pd.Series of feature values (numeric or categorical)
    - target: pd.Series of binary target (0 for good, 1 for bad)
    - bins: int, number of bins for numeric features
    - strategy: 'quantile' or 'uniform' for numeric binning
    
    Returns:
    - bin_table: pd.DataFrame with columns for each bin/category:
      ['bin', 'count', 'good', 'bad', 'dist_good', 'dist_bad', 'woe', 'iv']
    - iv: float, total Information Value for the feature
    """
    # Combine series & target into DataFrame
    df = pd.DataFrame({'feature': series, 'target': target})
    
    # Determine if numeric or categorical
    if pd.api.types.is_numeric_dtype(series):
        # Numeric: bin into intervals
        if strategy == 'quantile':
            df['bin'] = pd.qcut(df['feature'], q=bins, duplicates='drop')
        else:  # 'uniform'
            df['bin'] = pd.cut(df['feature'], bins=bins)
    else:
        # Categorical: each category is its own bin
        df['bin'] = df['feature'].astype('object')
    
    # Calculate totals
    total_good = (df['target'] == 0).sum()
    total_bad  = (df['target'] == 1).sum()
    
    # Group by bin and compute stats
    agg = df.groupby('bin').agg(
        count=('target', 'size'),
        bad=('target', 'sum'),
        good=('target', lambda x: (x == 0).sum())
    ).reset_index()
    
    # Calculate distributions, WOE and IV
    laplace = 0.7
    agg['dist_bad']  = (agg['bad']  + laplace) / (total_bad + laplace * len(agg))
    agg['dist_good'] = (agg['good'] + laplace) / (total_good + laplace * len(agg))
    eps = 0.5
    agg['woe'] = np.log((agg['dist_good'] + eps) / (agg['dist_bad'] + eps))
    agg['iv']  = (agg['dist_good'] - agg['dist_bad']) * agg['woe']
    
    # Sort bins for consistent mapping
    if pd.api.types.is_numeric_dtype(series):
        agg = agg.sort_values(by='bin')
    
    # Total IV
    iv = agg['iv'].sum()
    
    return agg[['bin','count','good','bad','dist_good','dist_bad','woe','iv']], iv


def transform_woe(series: pd.Series, bin_table: pd.DataFrame) -> pd.Series:
    """
    Map each value in series to its corresponding WOE based on bin_table.
    
    Parameters:
    - series: pd.Series of feature values
    - bin_table: DataFrame returned by compute_woe_iv containing 'bin' and 'woe'
    
    Returns:
    - pd.Series of WOE-transformed values
    """
    # Prepare mapping for categorical bins
    if pd.api.types.is_numeric_dtype(series):
        # Numeric: bins are Interval objects in bin_table['bin']
        def map_numeric(x):
            for interval, w in zip(bin_table['bin'], bin_table['woe']):
                if pd.isna(x):
                    return 0.0
                if x in interval:
                    return w
            return 0.0  # values outside known bins
        return series.map(map_numeric)
    else:
        # Categorical
        mapping = {cat: w for cat, w in zip(bin_table['bin'], bin_table['woe'])}
        return series.map(lambda x: mapping.get(x, 0.0))


class WoeTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn transformer for applying WOE transformation to multiple features.
    
    Usage:
        woe = WoeTransformer(features=['FeatureA', 'FeatureB'], bins=5)
        woe.fit(X, y)
        X_woe = woe.transform(X)
    """
    def __init__(self, features, bins=10, strategy='quantile'):
        self.features = features
        self.bins = bins
        self.strategy = strategy
        self.bin_tables_ = {}
        self.iv_values_ = {}
        self.numeric_binners_ = {}

    def fit(self, X, y=None):
        """Learn bins, WOE, and IV for each feature."""
        for feat in self.features:
            if pd.api.types.is_numeric_dtype(X[feat]):
                woe_bin = MonotonicBinning()
                woe_bin.fit(X[[feat]], y)
                bins = woe_bin.bins

                woe = WOE(woe_bins={feat: bins[feat]})
                woe.fit(X[[feat]], y)

                self.numeric_binners_[feat] = woe_bin
                self.bin_tables_[feat] = woe

                # FIX: match xverse output
                if 'Variable_Name' in woe.woe_df.columns:
                    iv = woe.woe_df[woe.woe_df['Variable_Name'] == feat]['Information_Value'].iloc[0]
                else:
                    raise KeyError(
                        f"WOE output missing 'Variable_Name' column. "
                        f"Columns found: {woe.woe_df.columns}"
                    )

                self.iv_values_[feat] = iv
            else:
                # ✅ CATEGORICAL: use manual fallback logic
                bin_table, iv = compute_woe_iv(X[feat], y)
                self.bin_tables_[feat] = bin_table
                self.iv_values_[feat] = iv

        return self


    def transform(self, X):
        """Transform features to their WOE values."""
        X_transformed = X.copy()
        for feat in self.features:
            if feat in self.numeric_binners_:
                woe = self.bin_tables_[feat]
                woe_trans = woe.transform(X[[feat]])
                X_transformed[feat + '_woe'] = woe_trans[feat]
                X_transformed = X_transformed.drop(columns=[feat])
            else:
                bin_table = self.bin_tables_[feat]
                X_transformed[feat + '_woe'] = transform_woe(X[feat], bin_table)
                X_transformed = X_transformed.drop(columns=[feat])
        return X_transformed

    def get_iv(self):
        """Return the computed IV values for features."""
        return self.iv_values_