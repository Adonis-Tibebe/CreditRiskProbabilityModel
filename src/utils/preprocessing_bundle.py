import pandas as pd
import scipy.sparse as sparse

class PreprocessingBundle:
    def __init__(self, woe_tr, num_pipeline, final_columns):
        self.woe_tr = woe_tr
        self.num_pipeline = num_pipeline
        self.final_columns = final_columns

    def transform(self, X_raw):
        X_raww = X_raw[['NetSpend', 'GrossVolume', 'TxnCount', 'AvgTxnValue', 'StdTxnValue', 'NumUniqueProducts', 'NumUniqueCategories', 'NumUniqueChannels', 'PreferredProvider', 'MostCommonPricingStrategy', 'PreferredChannel', 'Recency', 'PreferredDayOfWeek']].copy()
        X_woe = self.woe_tr.transform(X_raww)
        print(self.woe_tr.features)
        X_scaled = self.num_pipeline.transform(X_woe)
        woe_cols = [col for col in X_woe.columns if '_woe' in col]
        nonwoe_cols = [col for col in X_woe.columns if '_woe' not in col]
        X_combined = sparse.hstack([
            X_scaled,
            sparse.csr_matrix(X_woe[woe_cols].values)
        ])
        dense = X_combined.toarray()
        #print(X_woe.columns)
        #X_woe = X_woe.drop(columns=["CustomerId", "FraudCount", "PreferredHour"], axis=1)
        #print(X_woe.columns)
        return pd.DataFrame(dense, columns= X_woe.columns)