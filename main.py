from src.data_loader import get_sp500_data, get_volume, get_market_cap, get_tickers  
from src.clustering import kmeans_centers_MC_norm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def main(period: str, interval: str, n_clusters: int, random_state: int):
    # Step 1: Kmeans clustering of stocks with similar liquidity profiles and visualise.

    tickers = get_tickers()
    data = get_sp500_data(tickers = tickers, period = period, interval = interval)
    volume_df = get_volume(data)

    centers = kmeans_centers_MC_norm(volume_df, n_clusters = n_clusters, random_state = random_state)

    # Plot clusters

    for i, center in enumerate(centers):
        plt.plot(center, label = f"Cluster {i}")
    
    plt.title("Cluster Center Intraday Volume Profiles")
    plt.xlabel("Time Index (1-min intervals)")
    plt.ylabel("Normalised Volume")
    plt.legend()
    plt.grid(True)
    plt.show()


    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))

    for cluster in range(kmeans.n_clusters):
        subset = X_pca[labels == cluster]
        plt.scatter(subset[:, 0], subset[:, 1], label=f"Cluster {cluster}", alpha=0.7)

    # Project centers
    centers_pca = pca.transform(centers)
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1],
                marker='X', s=200, label='Centers')

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("KMeans Clusters (PCA Projection)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main(period = "5d", interval = "1m", n_clusters = 4, random_state = 42)
