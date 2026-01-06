from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def get_model(name, seed):
    if name == "logistic":
        return LogisticRegression(max_iter=1000, random_state=seed)
    elif name == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=seed)
    else:
        raise ValueError("Unsupported model")
