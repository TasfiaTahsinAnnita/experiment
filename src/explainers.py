import shap
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

def shap_explain(model, X_train, X_test):
    explainer = shap.Explainer(model, X_train)
    # check_additivity is only valid/necessary for TreeExplainer
    if "TreeExplainer" in explainer.__class__.__name__:
        shap_values = explainer(X_test, check_additivity=False)
    else:
        shap_values = explainer(X_test)
    vals = np.abs(shap_values.values)
    if vals.ndim == 3:
        vals = vals.mean(axis=-1)
    return vals.mean(axis=0)

def lime_explain(model, X_train, X_test):
    explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns,
        discretize_continuous=True
    )
    # Limit LIME to first 50 instances for speed in this demo context
    limit = min(len(X_test), 50)
    importances = np.zeros(len(X_train.columns))
    feature_map = {col: i for i, col in enumerate(X_train.columns)}
    
    for i in range(limit):
        exp = explainer.explain_instance(
            X_test.iloc[i].values,
            model.predict_proba,
            num_features=len(X_train.columns)
        )
        # Parse LIME output to map back to original features
        for feature_cond, weight in exp.as_list():
            # Find which original feature this condition corresponds to
            # This is a heuristic: match longest column name that appears in the condition
            # LIME usually returns "Feature <= 0.5" etc.
            
            # Sort columns by length descent to avoid partial matches (e.g. "Age" vs "AgeGroup")
            # We do this once outside ideally, but here is fine
            best_match = None
            for col in sorted(feature_map.keys(), key=len, reverse=True):
                if col in feature_cond:
                    best_match = col
                    break
            
            if best_match:
                importances[feature_map[best_match]] += abs(weight)
                
    return importances / limit
