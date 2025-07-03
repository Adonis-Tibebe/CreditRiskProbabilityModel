"""
Feature Engineering module for BNPL credit risk project.

This module defines pipelines for both raw feature processing (for tree-based models)
and WoE-transformed features (for logistic regression). Logging is integrated for transparency.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer



# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__) # Set up logger for this module

class FeatureEngineering:
    """
    Encapsulates feature engineering logic:
      - aggregate_features: compute per-customer aggregates from raw transactions
      - extract_time_features: compute recency and timestamp components
      - raw_pipeline: preprocessing for tree-based models
      - woe_pipeline: preprocessing for logistic regression with WoE
    """

    def __init__(self, num_features, cat_features, woe_features=None):
        
        """
        Initialize pipelines with feature lists.
        :param num_features: List[str] of numeric column names
        :param cat_features: List[str] of categorical column names
        :param woe_features: List[str] of features for WoE transformation
        """
        self.num_features = num_features
        self.cat_features = cat_features
        self.woe_features = woe_features or []
        self.tree_pipeline = None
        self.woe_pipeline = None

        logger.info("Initializing FeatureEngineering with numeric features %s and categorical features %s",
                    num_features, cat_features) # Daynamic logging of features


    def build_tree_pipeline(self):
        """Builds numeric + categorical pipeline for Tree models."""
        self.num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        self.cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.tree_pipeline = ColumnTransformer([
            ('num', self.num_pipeline, self.num_features),
            ('cat', self.cat_pipeline, self.cat_features)
        ])

    def build_woe_pipeline(self):
            """Builds numeric + categorical + WoE pipeline for Logistic Regression."""
            self.num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            self.cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            self.woe_pipeline = ColumnTransformer([
                ('num', self.num_pipeline, self.num_features),
                ('cat', self.cat_pipeline, self.cat_features)
                # Do NOT add WoE transformer here if you run it manually outside
            ])

    
    def aggregate_features(self, df):
        """
        Aggregates transaction features at the customer level.

        This function groups the input DataFrame by 'CustomerId' and calculates
        various aggregated features, such as net spend, transaction count, average
        transaction value, standard deviation of transaction value, number of unique
        products, categories, and channels, preferred provider, fraud count, preferred
        transaction hour, most common pricing strategy, and preferred channel. It also
        calculates the recency of transactions for each customer.

        :param df: pd.DataFrame, input data containing transaction details.
        :return: pd.DataFrame, customer-level aggregated features.
        """

        logger.info("Aggregating features to customer level")

        # Net Spend (directional)
        agg = df.groupby('CustomerId').agg(
            NetSpend=('Amount', 'sum'),
            GrossVolume=('Value', 'sum'),
            TxnCount=('Value', 'count'),
            AvgTxnValue=('Value', 'mean'),
            StdTxnValue=('Value', 'std'),
            NumUniqueProducts=('ProductId', pd.Series.nunique),
            NumUniqueCategories=('ProductCategory', pd.Series.nunique),
            NumUniqueChannels=('ChannelId', pd.Series.nunique),
            PreferredProvider=('ProviderId', lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'),
            FraudCount=('FraudResult', 'sum'),
            PreferredHour=('TransactionHour', lambda x: x.mode().iloc[0] if not x.mode().empty else -1),
            MostCommonPricingStrategy=('PricingStrategy', lambda x: x.mode().iloc[0] if not x.mode().empty else -1),
            PreferredChannel=('ChannelId', lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown')
        ).reset_index()

        # Recency
        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
        recency_df = df.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (snapshot_date - x.max()).days
        }).reset_index().rename(columns={'TransactionStartTime': 'Recency'})

        agg = agg.merge(recency_df, on='CustomerId')

        # Fill NaN for StdTxnValue if only 1 txn
        agg['StdTxnValue'] = agg['StdTxnValue'].fillna(0)

        agg['PreferredDayOfWeek'] = (
            df.groupby('CustomerId')['TransactionStartTime']
            .agg(lambda x: x.dt.dayofweek.mode().iloc[0] if not x.dt.dayofweek.mode().empty else -1)
            .values
        )

        logger.info(f"Aggregation complete — {agg.shape[0]} customers")
        return agg


    def extract_time_features(self, df):
        """
        Extracts time-based features from TransactionStartTime in a DataFrame.
        Note: operates on raw transactions.
        :param df: pd.DataFrame with raw transactions
        :return: pd.DataFrame with new columns in place
        """
        logger.info("Extracting datetime features...")
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        df['TransactionHour'] = df['TransactionStartTime'].dt.hour
        df['TransactionDay'] = df['TransactionStartTime'].dt.day
        df['TransactionMonth'] = df['TransactionStartTime'].dt.month
        df['TransactionYear'] = df['TransactionStartTime'].dt.year
        logger.info("Datetime features added: Hour, Day, Month, Year.")
        return df


    def fit_transform_tree(self, X):
        """Fit & transform for tree-based models."""
        return self.tree_pipeline.fit_transform(X)

    def transform_tree(self, X):
        """Transform for tree-based models."""
        return self.tree_pipeline.transform(X)

    def fit_transform_woe(self, X):
        """Fit & transform for logistic regression models (numeric + cat only)."""
        return self.woe_pipeline.fit_transform(X)

    def transform_woe(self, X):
        """Transform for logistic regression models (numeric + cat only)."""
        return self.woe_pipeline.transform(X)

    def get_feature_names_out_tree(self):
        """
        Returns the final feature names after tree pipeline transform.
        Useful for DataFrame output.
        """
        # Numeric features stay the same
        num_names = self.num_features

        # Get names from OHE inside cat_pipeline
        ohe = self.cat_pipeline.named_steps['onehot']
        ohe_names = list(ohe.get_feature_names_out(self.cat_features))

        return num_names + ohe_names


    def get_feature_names_out_woe(self):
        """
        Returns the final feature names after WoE pipeline transform.
        WoE drops the original columns and adds _woe suffix.
        """
        # Numeric features stay the same
        num_names = self.num_features

        # Categorical still goes through OHE
        ohe = self.cat_pipeline.named_steps['onehot']
        ohe_names = list(ohe.get_feature_names_out(self.cat_features))

        # WoE features get _woe suffix
        woe_names = [feat + '_woe' for feat in self.woe_features]

        return num_names + ohe_names + woe_names