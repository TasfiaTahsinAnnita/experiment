import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(data_source, target, seed):
    if isinstance(data_source, str):
        df = pd.read_csv(data_source)
    else:
        df = data_source.copy()

    # Basic preprocessing: Drop rows with missing target
    df = df.dropna(subset=[target])
    
    X = df.drop(columns=[target])
    y = df[target]

    # Handle non-numeric columns in X (One-Hot Encoding)
    X = pd.get_dummies(X, drop_first=True)
    
    # Fill remaining NaNs in X with 0 (simplified for robustness)
    X = X.fillna(0)

    return train_test_split(X, y, test_size=0.2, random_state=seed)
