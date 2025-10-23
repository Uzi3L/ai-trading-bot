from data_loader import load_config, get_data
from features import add_features
from model import train_model
from backtest import run_backtest

def main():
    cfg = load_config()
    df = get_data(cfg)
    df = add_features(df)
    model = train_model(df)
    run_backtest(df, model, cfg)

if __name__ == "__main__":
    main()
