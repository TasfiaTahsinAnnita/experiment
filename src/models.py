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
# import catboost as cb # Disabled due to build error on Windows without Visual Studio

def get_model(model_name, seed, task_type="classification"):
    random_state = seed
    
    # Model definitions
    classifiers = {
        "logistic": lambda s: LogisticRegression(random_state=s, max_iter=1000),
        "random_forest": lambda s: RandomForestClassifier(random_state=s),
        "decision_tree": lambda s: DecisionTreeClassifier(random_state=s),
        "svm": lambda s: SVC(probability=True, random_state=s),
        "knn": lambda s: KNeighborsClassifier(),
        "naive_bayes": lambda s: GaussianNB(),
        "gbm": lambda s: GradientBoostingClassifier(random_state=s),
        "xgboost": lambda s: xgb.XGBClassifier(random_state=s, eval_metric='logloss'),
        "lightgbm": lambda s: lgb.LGBMClassifier(random_state=s, verbose=-1),
        "adaboost": lambda s: AdaBoostClassifier(random_state=s),
        "extra_trees": lambda s: ExtraTreesClassifier(random_state=s),
        "mlp": lambda s: MLPClassifier(random_state=s, max_iter=1000),
        "dummy": lambda s: DummyClassifier(strategy="most_frequent", random_state=s)
    }

    regressors = {
        "linear": lambda s: LinearRegression(),
        "ridge": lambda s: Ridge(random_state=s),
        "lasso": lambda s: Lasso(random_state=s),
        "random_forest": lambda s: RandomForestRegressor(random_state=s),
        "decision_tree": lambda s: DecisionTreeRegressor(random_state=s),
        "svm": lambda s: SVR(),
        "knn": lambda s: KNeighborsRegressor(),
        "gbm": lambda s: GradientBoostingRegressor(random_state=s),
        "xgboost": lambda s: xgb.XGBRegressor(random_state=s),
        "lightgbm": lambda s: lgb.LGBMRegressor(random_state=s, verbose=-1),
        "adaboost": lambda s: AdaBoostRegressor(random_state=s),
        "extra_trees": lambda s: ExtraTreesRegressor(random_state=s),
        "mlp": lambda s: MLPRegressor(random_state=s, max_iter=1000),
        "dummy": lambda s: DummyRegressor(strategy="mean")
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
