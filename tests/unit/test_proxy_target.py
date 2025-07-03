import pandas as pd
from src.core.proxy_target import create_proxy_target

def test_create_proxy_target_adds_is_high_risk():
    # Minimal valid input with required columns
    df = pd.DataFrame({
        'Recency': [10, 20, 30, 40, 50, 60],
        'TxnCount': [1, 2, 3, 4, 5, 6],
        'GrossVolume': [100, 200, 300, 400, 500, 600]
    })
    result = create_proxy_target(df.copy())
    assert 'is_high_risk' in result.columns
    # Should be only 0 or 1
    assert set(result['is_high_risk'].unique()).issubset({0, 1})
    # Should have same number of rows
    assert len(result) == len(df)