from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
import xgboost as xgb
import lightgbm as lgb

def get_model(model_name, seed, task_type="classification"):
    random_state = seed
    
    # Model definitions - Updated to 50 Epochs/Iterations where applicable
    classifiers = {
        "logistic": lambda s: LogisticRegression(random_state=s, max_iter=50),
        "random_forest": lambda s: RandomForestClassifier(random_state=s, n_estimators=50),
        "decision_tree": lambda s: DecisionTreeClassifier(random_state=s),
        "naive_bayes": lambda s: GaussianNB(),
        "gbm": lambda s: GradientBoostingClassifier(random_state=s, n_estimators=50),
        "xgboost": lambda s: xgb.XGBClassifier(random_state=s, eval_metric='logloss', n_estimators=50),
        "lightgbm": lambda s: lgb.LGBMClassifier(random_state=s, verbose=-1, n_estimators=50),
        "adaboost": lambda s: AdaBoostClassifier(random_state=s, n_estimators=50),
        "mlp": lambda s: MLPClassifier(random_state=s, max_iter=50)
    }

    regressors = {
        "linear": lambda s: LinearRegression(),
        "random_forest": lambda s: RandomForestRegressor(random_state=s, n_estimators=50),
        "decision_tree": lambda s: DecisionTreeRegressor(random_state=s),
        "gbm": lambda s: GradientBoostingRegressor(random_state=s, n_estimators=50),
        "xgboost": lambda s: xgb.XGBRegressor(random_state=s, n_estimators=50),
        "lightgbm": lambda s: lgb.LGBMRegressor(random_state=s, verbose=-1, n_estimators=50),
        "adaboost": lambda s: AdaBoostRegressor(random_state=s, n_estimators=50),
        "mlp": lambda s: MLPRegressor(random_state=s, max_iter=50)
    }

    if task_type == "classification":
        if model_name in classifiers:
            return classifiers[model_name](random_state)
    elif task_type == "regression":
        if model_name in regressors:
            return regressors[model_name](random_state)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    raise ValueError(f"Unknown model name: {model_name} for task {task_type}")
