import yfinance as yf
import yaml

def load_config(path="config/settings.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_data(cfg):
    df = yf.download(cfg["ticker"], start=cfg["start_date"], end=cfg["end_date"], progress=False)
    return df[['Open','High','Low','Close','Volume']].dropna()

if __name__ == "__main__":
    cfg = load_config()
    print(get_data(cfg).tail())
