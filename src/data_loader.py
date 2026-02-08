import numpy as np
import pandas as pd
import yfinance as yf
from pytickersymbols import PyTickerSymbols

def get_sp500_data(tickers: list, period: str, interval: str) -> dict[str, pd.DataFrame]:
    if not tickers:
        raise ValueError("Tickers list is empty.")
    
    data = {}

    for ticker in tickers:
        df = yf.download(
            ticker,
            period = period,
            interval = interval,
            progress = True
            )
        if df.empty:
            print(f"Skipping {ticker}: No data.")
            continue

        data[ticker] = df

    return data


def get_volume(data: dict[str, pd.DataFrame]) -> pd.DataFrame:

    volume_df = pd.DataFrame()

    for ticker, df in data.items():
        if "Volume" not in df.columns:
            continue
        volume_df[ticker] = df["Volume"]

    return volume_df


def get_market_cap(tickers: list[str]) -> pd.Series:
    market_caps = {}

    for ticker in tickers:
        try: 
            t = yf.Ticker(ticker)
            mc = t.info.get("marketCap")

            if mc is None:
                print(f"Market cap not found for {ticker}.")
                continue
            market_caps[ticker] = mc
        
        except Exception as e:
            print(f"Failed to fetch market cap for {ticker}: {e}")

    return pd.Series(market_caps)

# mc = get_market_cap(["AAPL", "SPY", "GLD"])
# print(print(mc))


def get_tickers() -> list[str]:
    tickers_df = pd.read_csv("data\SP500.csv")

    return list(tickers_df["Symbol"])

tickers = ["AAPL", "NVDA", "SPY", "GLD", "GOOG"]

data = get_sp500_data(tickers, period = "5d", interval = "1m")
volume_df = get_volume(data)
print(np.array(volume_df))