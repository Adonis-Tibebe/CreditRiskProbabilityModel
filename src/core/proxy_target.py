"""
Proxy Target Label module.
"""

import logging
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_proxy_target(df):
    """
    Adds is_high_risk to aggregated customer features using KMeans on RFM.
    """

    logger.info("Starting RFM clustering for proxy target...")

    rfm_features = df[['Recency', 'TxnCount', 'GrossVolume']].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm_features)

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    df['rfm_cluster'] = clusters

    logger.info(f"Cluster centers:\n{ kmeans.cluster_centers_ }")

    # Inspect centroids to find the "high risk" one:
    centroids = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=['Recency', 'TxnCount', 'GrossVolume']
    )
    logger.info(f"Inversed centroids:\n{ centroids }")

    # This step requires your judgment:
    # Find cluster with High Recency & Low Txn & Low Spend
    risky_cluster = centroids.sort_values(['Recency', 'TxnCount', 'GrossVolume'], ascending=[False, True, True]).index[0]
    logger.info(f"Identified risky cluster: {risky_cluster}")

    df['is_high_risk'] = (df['rfm_cluster'] == risky_cluster).astype(int)

    df = df.drop(columns=['rfm_cluster'])
    logger.info("Proxy target created: is_high_risk ✅")
    return df
