import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ret_1'] = df['Close'].pct_change(1)
    df['ret_5'] = df['Close'].pct_change(5)
    df['ma_5'] = df['Close'].rolling(5).mean()
    df['ma_20'] = df['Close'].rolling(20).mean()
    df['vol_20'] = df['Volume'].rolling(20).mean()
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df.dropna()
