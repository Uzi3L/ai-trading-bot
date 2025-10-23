from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model(df):
    features = ['ret_1','ret_5','ma_5','ma_20','vol_20']
    X = df[features]
    y = df['target']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    acc = accuracy_score(y, model.predict(X))
    print(f"Training Accuracy: {acc:.3f}")
    return model
