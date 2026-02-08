from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from data_loader import get_sp500_data, get_volume, get_market_cap

tickers = ["AAPL", "NVDA", "SPY", "GLD", "GOOG"]

data = get_sp500_data(tickers, period = "5d", interval = "1m")
volume_df = get_volume(data)


def kmeans_centers_MC_norm(volume_df: pd.DataFrame, n_clusters: int, random_state: int) -> np.ndarray:
    volume_df = volume_df.fillna(0)
    market_caps = get_market_cap(volume_df.columns.tolist())
    market_caps = market_caps.reindex(volume_df.columns)  

    norm_volume_df = volume_df.divide(market_caps, axis = 1) 
    norm_volume_df.fillna(0)

    X = norm_volume_df.T.values

    kmeans = KMeans(n_clusters = n_clusters, random_state = random_state, n_init = 10)

    kmeans.fit(X)
    
    return kmeans.cluster_centers_

centers = kmeans_centers_MC_norm(volume_df = volume_df, n_clusters = 3, random_state = 42)

print(centers)