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
import catboost as cb

def get_model(model_name, seed, task_type="classification"):
    random_state = seed
    
    # --- CLASSIFICATION ---
    if task_type == "classification":
        if model_name == "logistic":
            return LogisticRegression(random_state=random_state, max_iter=1000)
        elif model_name == "random_forest":
            return RandomForestClassifier(random_state=random_state)
        elif model_name == "decision_tree":
            return DecisionTreeClassifier(random_state=random_state)
        elif model_name == "svm":
            return SVC(probability=True, random_state=random_state)
        elif model_name == "knn":
            return KNeighborsClassifier()
        elif model_name == "naive_bayes":
            return GaussianNB()
        elif model_name == "gbm":
            return GradientBoostingClassifier(random_state=random_state)
        elif model_name == "xgboost":
            return xgb.XGBClassifier(random_state=random_state, eval_metric='logloss')
        elif model_name == "lightgbm":
            return lgb.LGBMClassifier(random_state=random_state, verbose=-1)
        elif model_name == "catboost":
            return cb.CatBoostClassifier(random_state=random_state, verbose=0)
        elif model_name == "adaboost":
            return AdaBoostClassifier(random_state=random_state)
        elif model_name == "extra_trees":
            return ExtraTreesClassifier(random_state=random_state)
        elif model_name == "mlp":
            return MLPClassifier(random_state=random_state, max_iter=1000)
        elif model_name == "dummy":
            return DummyClassifier(strategy="most_frequent", random_state=random_state)
        
    # --- REGRESSION ---
    elif task_type == "regression":
        if model_name == "linear":
            return LinearRegression()
        elif model_name == "ridge":
            return Ridge(random_state=random_state)
        elif model_name == "lasso":
            return Lasso(random_state=random_state)
        elif model_name == "random_forest":
            return RandomForestRegressor(random_state=random_state)
        elif model_name == "decision_tree":
            return DecisionTreeRegressor(random_state=random_state)
        elif model_name == "svm":
            return SVR()
        elif model_name == "knn":
            return KNeighborsRegressor()
        elif model_name == "gbm":
            return GradientBoostingRegressor(random_state=random_state)
        elif model_name == "xgboost":
            return xgb.XGBRegressor(random_state=random_state)
        elif model_name == "lightgbm":
            return lgb.LGBMRegressor(random_state=random_state, verbose=-1)
        elif model_name == "catboost":
            return cb.CatBoostRegressor(random_state=random_state, verbose=0)
        elif model_name == "adaboost":
            return AdaBoostRegressor(random_state=random_state)
        elif model_name == "extra_trees":
            return ExtraTreesRegressor(random_state=random_state)
        elif model_name == "mlp":
            return MLPRegressor(random_state=random_state, max_iter=1000)
        elif model_name == "dummy":
            return DummyRegressor(strategy="mean")

    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    raise ValueError(f"Unknown model name: {model_name} for task {task_type}")
