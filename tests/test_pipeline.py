from src.data_loader import load_config, get_data
from src.features import add_features

def test_data_pipeline():
    cfg = load_config()
    df = get_data(cfg)
    df_feat = add_features(df)
    assert not df_feat.empty
    assert 'target' in df_feat.columns
