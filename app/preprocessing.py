from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def make_pipeline():
    # Параметры StandardScaler обучаются один раз вместе с моделью.
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=10, max_iter=1000, random_state=42)),
    ])
