from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from .data_loader import get_sp500_data, get_volume, get_market_cap



def kmeans_centers_MC_norm(volume_df: pd.DataFrame, n_clusters: int, random_state: int) -> np.ndarray:
    """
    Returns: an array of clustering centers

    Normalised data by market caps

    """


    volume_df = volume_df.fillna(0)
    market_caps = get_market_cap(volume_df.columns.tolist())
    market_caps = market_caps.reindex(volume_df.columns)  

    norm_volume_df = volume_df.divide(market_caps, axis = 1) 
    norm_volume_df.fillna(0)

    X = norm_volume_df.T.values

    print("Shape of volume_df:", volume_df.shape)
    print("Shape of X:", volume_df.T.shape)

    kmeans = KMeans(n_clusters = n_clusters, random_state = random_state, n_init = 10)

    kmeans.fit(X)
    
    return kmeans.cluster_centers_

