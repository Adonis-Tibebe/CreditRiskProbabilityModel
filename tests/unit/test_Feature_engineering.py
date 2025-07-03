import pandas as pd
import pytest
from src.core.Feature_engineering import FeatureEngineering

def test_extract_time_features():
    df = pd.DataFrame({'TransactionStartTime': ['2024-01-01 10:00:00']})
    fe = FeatureEngineering([], [], [])
    result = fe.extract_time_features(df)
    assert isinstance(result, pd.DataFrame)
    assert 'TransactionStartTime' in result.columns or len(result.columns) > 0

def test_aggregate_features():
    df = pd.DataFrame({
        'CustomerId': [1, 1],
        'Amount': [-100, 200],
        'ChannelId': [1, 1],
        'FraudResult': [0, 1],
        'PricingStrategy': [1, 2],
        'ProductCategory': ['A', 'B'],
        'ProductId': [101, 102],
        'ProviderId': [201, 202],
        "TransactionHour":[1, 24],
        "Value": [100,200],
        'TransactionStartTime': ['2024-01-01 10:00:00', '2024-01-01 11:00:00']
    })
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime']) 

    fe = FeatureEngineering([], [], [])
    result = fe.aggregate_features(df)
    assert isinstance(result, pd.DataFrame)