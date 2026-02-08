import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# Step 1: Data Collection

def get_data_volume(tickers: list[str], start_date, end_date, interval) -> pd.DataFrame:
    if not tickers:
        raise ValueError("Tickers list empty.")

    data = pd.DataFrame()

    for ticker in tickers:
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False
        )

        if df.empty or "Volume" not in df.columns:
            raise ValueError(f"Ticker '{ticker}' does not exist or has no data.")

        data[ticker] = df["Volume"]

    return data



sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
sp500 = pd.read_html(sp500_url)[0]

tickers = sp500["Symbol"].tolist()
print(tickers)

data = get_data_volume(tickers, "2026-01-21", "2026-01-22", "1m")
print(data.head())

    