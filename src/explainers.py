import shap
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

def shap_explain(model, X_train, X_test, task_type="classification"):
    # Wrapper helper to handle potential serialization issues with some models (like TF/Keras if used)
    # For sklearn/xgboost/etc, standard Explainer is usually fine.
    
    # Use KernelExplainer for models where TreeExplainer might fail or for generic support (e.g. SVM, KNN)
    # However, TreeExplainer is faster for trees.
    # Let's try flexible selection.
    
    try:
        explainer = shap.Explainer(model, X_train)
    except:
        # Fallback for models like KNN/SVM where generic Explainer might struggle finding masker
        checker = model.predict_proba if task_type == "classification" else model.predict
        explainer = shap.KernelExplainer(checker, X_train.iloc[:50, :]) # Summary background
        
    # check_additivity is only valid/necessary for TreeExplainer
    if hasattr(explainer, "explainer") and "TreeExplainer" in explainer.explainer.__class__.__name__:
         shap_values = explainer(X_test, check_additivity=False)
    elif "TreeExplainer" in explainer.__class__.__name__:
        shap_values = explainer(X_test, check_additivity=False)
    else:
        # Kernel explainer returns list for classification?
        try:
            shap_values = explainer(X_test)
        except Exception as e:
            # Fallback for KernelExplainer on huge data or specific failures
             shap_values = explainer(X_test, check_additivity=False)

    # SHAP returns Explanation object or numpy array depending on version/explainer
    if isinstance(shap_values, shap.Explanation):
        vals = np.abs(shap_values.values)
    elif isinstance(shap_values, list): # KernelExplainer for multiclass
        vals = np.abs(np.array(shap_values))
    else:
        vals = np.abs(shap_values)

    # Handle dimensions
    # Regression: (n_samples, n_features) -> ndim=2
    # Classification: (n_samples, n_features, n_classes) -> ndim=3
    # Or (n_classes, n_samples, n_features) depending on output?
    
    if vals.ndim == 3:
        # Average over classes
        vals = vals.mean(axis=-1)
    
    # Average over samples to get global feature importance
    return vals.mean(axis=0)

def lime_explain(model, X_train, X_test, task_type="classification"):
    mode = "classification" if task_type == "classification" else "regression"
    
    explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns,
        discretize_continuous=True,
        mode=mode
    )
    # Limit LIME to first 50 instances for speed in this demo context
    limit = min(len(X_test), 50)
    importances = np.zeros(len(X_train.columns))
    feature_map = {col: i for i, col in enumerate(X_train.columns)}
    
    predict_fn = model.predict_proba if task_type == "classification" else model.predict
    
    for i in range(limit):
        try:
            exp = explainer.explain_instance(
                X_test.iloc[i].values,
                predict_fn,
                num_features=len(X_train.columns)
            )
            # Parse LIME output to map back to original features
            for feature_cond, weight in exp.as_list():
                best_match = None
                for col in sorted(feature_map.keys(), key=len, reverse=True):
                    if col in feature_cond:
                        best_match = col
                        break
                
                if best_match:
                    importances[feature_map[best_match]] += abs(weight)
        except Exception:
            continue
                
    return importances / limit
