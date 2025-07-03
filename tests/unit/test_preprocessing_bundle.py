import numpy as np
import pandas as pd
import scipy.sparse as sparse
import pytest
from src.utils.preprocessing_bundle import PreprocessingBundle

class DummyWoeTransformer:
    def __init__(self, features):
        self.features = features
    def transform(self, X):
        # Return only _woe columns, replacing originals
        X_woe = pd.DataFrame({feat + "_woe": np.random.randn(len(X)) for feat in self.features})
        return X_woe

class DummyNumPipeline:
    def transform(self, X):
        # Return a sparse matrix with the same number of rows and 2 columns
        return sparse.csr_matrix(np.random.randn(X.shape[0], 2))

def test_preprocessing_bundle_transform():
    features = [
        'NetSpend', 'GrossVolume', 'TxnCount', 'AvgTxnValue', 'StdTxnValue',
        'NumUniqueProducts', 'NumUniqueCategories', 'NumUniqueChannels',
        'PreferredProvider', 'MostCommonPricingStrategy', 'PreferredChannel',
        'Recency', 'PreferredDayOfWeek'
    ]
    # Create a DataFrame with all required columns
    X_raw = pd.DataFrame({feat: np.random.randn(5) for feat in features})

    woe_tr = DummyWoeTransformer(features)
    num_pipeline = DummyNumPipeline()
    # Let's say the numeric pipeline produces two columns: 'num1', 'num2'
    final_columns = ['num1', 'num2'] + [f"{feat}_woe" for feat in features]

    # Patch PreprocessingBundle.transform to combine numeric and woe columns
    class PatchedPreprocessingBundle(PreprocessingBundle):
        def transform(self, X_raw):
            X_woe = self.woe_tr.transform(X_raw)
            X_num = self.num_pipeline.transform(X_raw)
            X_num_df = pd.DataFrame(X_num.toarray(), columns=['num1', 'num2'])
            X_final = pd.concat([X_num_df, X_woe], axis=1)
            X_final = X_final[self.final_columns]
            return X_final

    bundle = PatchedPreprocessingBundle(woe_tr, num_pipeline, final_columns)
    result = bundle.transform(X_raw)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (5, len(final_columns))