import numpy as np
import pandas as pd
import yfinance as yf
from pytickersymbols import PyTickerSymbols

def get_sp500_data(tickers: list, period: str, interval: str) -> pd.DataFrame:
    if not tickers:
        raise ValueError("Tickers list is empty.")
    
    data = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=True,
        threads=True,      # important
        progress=True
    )
    return data


def get_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the 'Volume' columns for all tickers from a multi-index DataFrame
    returned by yfinance when downloading multiple tickers at once.
    """
    volume_df = pd.DataFrame()

    # check if df has MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        for ticker in df.columns.levels[0]:
            # select the 'Volume' column for this ticker
            volume_df[ticker] = df[ticker]["Volume"]
    else:
        # single ticker case
        volume_df[df.columns.name] = df["Volume"]  # optional, or just df["Volume"]
    
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

