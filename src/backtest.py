import vectorbt as vbt

def run_backtest(df, model, cfg):
    features = ['ret_1','ret_5','ma_5','ma_20','vol_20']
    df['signal'] = model.predict(df[features])
    price = df['Close']
    entries = df['signal'] == 1
    exits = df['signal'] == 0

    pf = vbt.Portfolio.from_signals(
        close=price,
        entries=entries,
        exits=exits,
        init_cash=cfg["initial_cash"],
        fees=0.001,
        slippage=0.0005
    )
    print(pf.stats())
