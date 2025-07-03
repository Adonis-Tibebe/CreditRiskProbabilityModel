import pandas as pd
import numpy as np
from src.core.woe_transformer import WoeTransformer

def test_woe_transformer_fit_transform():
    # Create a DataFrame with both numeric and categorical features
    df = pd.DataFrame({
        'NetSpend': [100, 200, 300, 400, 500, 600],
        'GrossVolume': [1000, 2000, 3000, 4000, 5000, 6000],
        'TxnCount': [1, 2, 3, 4, 5, 6],
        'AvgTxnValue': [10, 20, 30, 40, 50, 60],
        'Recency': [5, 10, 15, 20, 25, 30],
        'PreferredProvider': ['A', 'B', 'A', 'B', 'C', 'C'],
        'MostCommonPricingStrategy': [1, 3, 2, 3, 3, 1],
        'PreferredChannel': ['web', 'app', 'web', 'app', 'ussd', 'ussd'],
        'is_high_risk': [0, 1, 0, 1, 0, 1]
    })
    features = [
        'NetSpend', 'GrossVolume', 'TxnCount', 'AvgTxnValue', 'Recency',
        'PreferredProvider', 'MostCommonPricingStrategy', 'PreferredChannel'
    ]
    X = df[features]
    y = df['is_high_risk']

    woe = WoeTransformer(features=features, bins=3)
    woe.fit(X, y)
    X_woe = woe.transform(X)

    # Check that all original features are replaced by _woe columns
    for feat in features:
        assert feat + '_woe' in X_woe.columns
        assert feat not in X_woe.columns

    # Check shape and type
    assert isinstance(X_woe, pd.DataFrame)
    assert X_woe.shape[0] == X.shape[0]

    # Check IV values
    ivs = woe.get_iv()
    assert isinstance(ivs, dict)
    assert set(ivs.keys()) == set(features)