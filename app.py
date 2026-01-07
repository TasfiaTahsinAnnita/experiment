import streamlit as st
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import warnings

# Suppress annoying warnings from sklearn/shap interactions
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Import project modules
# We need to make sure the root dir is in sys.path if we run this from root
import sys
import os
sys.path.append(os.path.abspath('.'))

from src.data_loader import load_data
from src.models import get_model
from src.explainers import shap_explain, lime_explain
from src.perturbations import add_noise
from src.metrics import spearman_stability, top_k_jaccard
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay, auc, mean_squared_error, r2_score, accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error
from sklearn.model_selection import learning_curve
from scipy import stats
import time

def add_watermark(ax):
    ax.text(0.99, 0.01, 'Made by TasfiaTahsinAnnita', 
            transform=ax.transAxes, 
            ha='right', va='bottom', 
            fontsize=10, color='grey', alpha=0.5, weight='bold')

st.set_page_config(page_title="Explainability Consistency Analyzer", layout="wide")

st.title("Explainability Consistency Analyzer")
st.markdown("""
This tool evaluates the stability of machine learning feature importance explanations (specifically SHAP).
It trains a model, generates explanations, adds noise to the test data, and checks if the explanations remain consistent.
""")

# Sidebar Configuration
st.sidebar.header("Configuration")

uploaded_file = st.sidebar.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    # Load data immediately
    try:
        df = pd.read_csv(uploaded_file)
        
        # Initialize global state lists if needed
        if "experiment_history" not in st.session_state:
            st.session_state.experiment_history = []
        
        if "processed_df" not in st.session_state:
             st.session_state.processed_df = None

        # --- STEP 1: TARGET SELECTION & CONFIG ---
        st.write("---")
        st.header("1. Analysis Setup")
        
        col_setup1, col_setup2 = st.columns(2)
        with col_setup1:
            target_col = st.selectbox("Select Target Column", df.columns)
            st.session_state.target_col = target_col
            
            # Heuristic for task type
            is_numeric = pd.api.types.is_numeric_dtype(df[target_col])
            recommended_task = "regression" if is_numeric and df[target_col].nunique() > 20 else "classification"
            task_type = st.selectbox("Task Type", ["classification", "regression"], index=0 if recommended_task=="classification" else 1)

        with col_setup2:
            st.write("**Configuration**")
            
            # Define Model Lists
            if task_type == "classification":
                # Removed: svm, extra_trees, dummy, knn
                all_models = [
                    "logistic", "random_forest", "xgboost", "lightgbm", "decision_tree", 
                    "naive_bayes", "gbm", "adaboost", "mlp"
                ]
                # Default "Core" models (Fast & Robust)
                default_selection = ["logistic", "random_forest", "xgboost", "decision_tree"]
            else:
                # Removed: svm, knn, dummy, extra_trees, ridge, lasso
                all_models = [
                    "linear", "random_forest", "xgboost", "lightgbm", 
                    "decision_tree", "gbm", "adaboost", "mlp"
                ]
                default_selection = ["linear", "random_forest", "xgboost", "decision_tree"]
                
            # Allow User Selection to control load
            selected_models = st.multiselect("Select Models to Run", all_models, default=default_selection)
            
            # Allow Explainer Selection
            selected_explainers = st.multiselect("Select Explainers", ["SHAP", "LIME"], default=["SHAP"])
                
            run_analysis_btn = st.button("RUN ANALYSIS", type="primary", use_container_width=True)
            if len(selected_models) > 5 or len(selected_explainers) > 1:
                st.caption("⚠️ Running many models/explainers may take a long time and consume high memory.")

        # --- EXECUTION LOGIC ---
        if run_analysis_btn:
            if not selected_models:
                st.error("Please select at least one model.")
            else:
                st.session_state.has_run = True
                st.session_state.auto_task_type = task_type
                st.session_state.auto_models = selected_models
                st.session_state.auto_explainers = selected_explainers
                # Clear previous run results
                st.session_state.experiment_history = []
                st.session_state.failed_models = [] 
                st.session_state.processed_df = None

        if st.session_state.get("has_run", False):
            st.divider()
            status_container = st.empty()
            status_container.info("Status: Running Analysis... Scroll down to track progress.")
            
            # ==========================================
            # 1. AUTOMATED EDA
            # ==========================================
            with st.expander("2. Exploratory Data Analysis (EDA) ✅", expanded=False):
                st.subheader("Data Overview")
                st.dataframe(pd.DataFrame({
                    "Column": df.columns,
                    "Type": df.dtypes.astype(str),
                    "Missing": df.isnull().sum(),
                    "Unique": df.nunique()
                }))
                
                st.subheader("Visualizations")
                cols_to_viz = [c for c in df.columns if c != target_col][:6]
                for col in cols_to_viz:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.caption(f"Distribution: {col}")
                        fig, ax = plt.subplots(figsize=(5, 3))
                        if pd.api.types.is_numeric_dtype(df[col]):
                            sns.histplot(df[col], kde=True, ax=ax)
                        else:
                            sns.countplot(y=df[col], ax=ax)
                        add_watermark(ax)
                        st.pyplot(fig)
                    with c2:
                        st.caption(f"{col} vs {target_col}")
                        fig, ax = plt.subplots(figsize=(5, 3))
                        if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(df[target_col]):
                            sns.scatterplot(data=df, x=col, y=target_col, ax=ax)
                        elif not pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(df[target_col]):
                            sns.boxplot(data=df, x=col, y=target_col, ax=ax)
                        elif pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_numeric_dtype(df[target_col]):
                            sns.boxplot(data=df, x=target_col, y=col, ax=ax)
                        else:
                            try:
                                ct = pd.crosstab(df[col], df[target_col])
                                sns.heatmap(ct, annot=True, fmt='d', ax=ax)
                            except: pass
                        add_watermark(ax)
                        st.pyplot(fig)

            # ==========================================
            # 2. AUTOMATED PREPROCESSING
            # ==========================================
            if "processed_df" not in st.session_state or st.session_state.processed_df is None:
                with st.spinner("Step 3: Preprocessing Data..."):
                    # Heuristic Pipeline
                    temp_df = df.copy()
                    
                    # Drop High Cardinality ID cols
                    for col in temp_df.columns:
                        if temp_df[col].dtype == 'object' and temp_df[col].nunique() > 0.9 * len(temp_df) and col != target_col:
                            temp_df.drop(columns=[col], inplace=True)

                    # Impute
                    num_cols_p = temp_df.select_dtypes(include=np.number).columns
                    cat_cols_p = temp_df.select_dtypes(exclude=np.number).columns
                    
                    if len(num_cols_p) > 0:
                        si = SimpleImputer(strategy='mean')
                        temp_df[num_cols_p] = si.fit_transform(temp_df[num_cols_p])
                    
                    if len(cat_cols_p) > 0:
                        si_c = SimpleImputer(strategy='most_frequent')
                        temp_df[cat_cols_p] = si_c.fit_transform(temp_df[cat_cols_p])
                        # Encode
                        temp_df = pd.get_dummies(temp_df, columns=cat_cols_p, drop_first=True)
                    
                    # Scale Features
                    scaler = StandardScaler()
                    feats = [c for c in temp_df.columns if c != target_col]
                    if feats:
                        temp_df[feats] = scaler.fit_transform(temp_df[feats])
                    
                    # Fix for XGBoost/LightGBM: Ensure Classification Targets are encoded 0..N-1
                    if task_type == "classification":
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        temp_df[target_col] = le.fit_transform(temp_df[target_col])
                    
                    st.session_state.processed_df = temp_df
                    
            # ==========================================
            # 3. EXPERIMENT EXECUTION
            # ==========================================
            processed_data = st.session_state.processed_df.copy() # Work on a copy to avoid mutation issues
            
            # FORCE DATA VALIDATION FOR CLASSIFICATION
            # (Fixes XGBoost stale cache issue where Y might still have gaps)
            if st.session_state.auto_task_type == "classification":
                 try:
                     from sklearn.preprocessing import LabelEncoder
                     le_force = LabelEncoder()
                     # Fit transform on the specific column
                     y_enc = le_force.fit_transform(processed_data[target_col].astype(str)) # casting to str ensures consistent type
                     processed_data[target_col] = y_enc
                     
                     # Verify encoding
                     # valid_y = np.unique(y_enc)
                     # if len(valid_y) != valid_y.max() + 1:
                     #    st.warning(f"Label Encoding check: Gaps still detected? {valid_y}")
                     
                 except Exception as e:
                     st.error(f"Failed to encode target: {e}")

            # Define X and y
            X = processed_data.drop(columns=[target_col])
            y = processed_data[target_col]
            
            # Initialise failure list if not present
            if "failed_models" not in st.session_state:
                st.session_state.failed_models = []

            # Only start training if we haven't done it this run-session
            if not st.session_state.experiment_history and not st.session_state.failed_models:
                 with st.status("Step 4: Training Models & Generating Explanations...", expanded=True) as status:
                     models = st.session_state.auto_models
                     explainers = st.session_state.get("auto_explainers", ["SHAP"])
                     
                     seeds = [1, 2] 
                     noise = 0.05
                     
                     for m_name in models:
                         # We treat the model as the unit of failure.
                         model_failed_completely = True
                         failure_reason = ""

                         for e_name in explainers:
                             status.write(f"Processing: {m_name} + {e_name}")
                             
                             m_results = []
                             start_time = time.time()
                             success = True
                             
                             
                             # Cache for Learning Curve (only run once per model to save time)
                             lc_data = None
                             
                             for seed in seeds:
                                 try:
                                     X_tr, X_te, y_tr, y_te = load_data(processed_data, target_col, seed)
                                     model = get_model(m_name, seed, task_type=st.session_state.auto_task_type)
                                     
                                     # --- Advanced Training with History ---
                                     epoch_history = {}
                                     try:
                                         # Setup for Iterative Models (XGB, LGBM) to get Epoch Curves
                                         if "XGB" in str(type(model)) or "LGBM" in str(type(model)):
                                             eval_set = [(X_tr, y_tr), (X_te, y_te)]
                                             # Determine metric
                                             if st.session_state.auto_task_type == "classification":
                                                  eval_metric = ["logloss", "error"] # error is 1-accuracy
                                             else:
                                                  eval_metric = ["rmse", "mae"]
                                                  
                                             model.fit(X_tr, y_tr, eval_set=eval_set, eval_metric=eval_metric, verbose=False)
                                             
                                             # Extract History
                                             if hasattr(model, "evals_result"):
                                                 results = model.evals_result()
                                                 # Parse XGB/LGBM structure dict[dname][metric] = list
                                                 epoch_history["type"] = "boost"
                                                 epoch_history["data"] = results
                                                 epoch_history["metrics"] = eval_metric
                                         
                                         elif "GradientBoosting" in str(type(model)):
                                             model.fit(X_tr, y_tr)
                                             # GBM exposes train_score_ (Loss)
                                             epoch_history["type"] = "sklearn_gbm"
                                             epoch_history["loss"] = model.train_score_
                                             # Validate score not available by default in sklearn GBM without monitor, but loss is.
                                             
                                         elif "MLP" in str(type(model)):
                                             model.fit(X_tr, y_tr)
                                             epoch_history["type"] = "sklearn_mlp"
                                             epoch_history["loss"] = model.loss_curve_
                                             
                                         else:
                                             # Standard fit for others
                                             model.fit(X_tr, y_tr)
                                             
                                     except TypeError as te:
                                          # Fallback if eval_set not supported for some version/wrapper
                                          model.fit(X_tr, y_tr)
                                     except Exception as ex:
                                          raise ex # Re-raise real errors

                                     expl_fn = shap_explain if e_name == "SHAP" else lime_explain
                                     base = expl_fn(model, X_tr, X_te, task_type=st.session_state.auto_task_type)
                                     
                                     X_te_noisy = add_noise(X_te, noise)
                                     noisy = expl_fn(model, X_tr, X_te_noisy, task_type=st.session_state.auto_task_type)
                                     
                                     s_stab = spearman_stability(base, noisy)
                                     s_jac = top_k_jaccard(base, noisy)
                                     
                                     # Performance Metrics & Viz Data
                                     y_pred = model.predict(X_te)
                                     y_prob = None
                                     if st.session_state.auto_task_type == "classification" and hasattr(model, "predict_proba"):
                                         try:
                                             y_prob = model.predict_proba(X_te)[:, 1] # Binary prob for pos class
                                             # Handle multiclass later if needed, assume binary for ROC/PR simplicity now or take max
                                             if model.classes_.shape[0] > 2:
                                                 # For multiclass, we might just store all probs or skip ROC for now
                                                 y_prob = model.predict_proba(X_te)
                                         except: pass

                                     perf_metrics = {}
                                     if st.session_state.auto_task_type == "classification":
                                         perf_metrics["Accuracy"] = accuracy_score(y_te, y_pred)
                                         perf_metrics["F1 Score"] = f1_score(y_te, y_pred, average='weighted', zero_division=0)
                                         perf_metrics["Precision"] = precision_score(y_te, y_pred, average='weighted', zero_division=0)
                                         perf_metrics["Recall"] = recall_score(y_te, y_pred, average='weighted', zero_division=0)
                                     else:
                                         perf_metrics["RMSE"] = np.sqrt(mean_squared_error(y_te, y_pred))
                                         perf_metrics["R2 Score"] = r2_score(y_te, y_pred)
                                         perf_metrics["MAE"] = mean_absolute_error(y_te, y_pred)
                                     
                                     # SKLEARN Learning Curve (Sample Size) - Keep as secondary context
                                     if lc_data is None:
                                         try:
                                             train_sizes, train_scores, test_scores = learning_curve(
                                                 model, X_tr, y_tr, cv=3, n_jobs=-1, 
                                                 train_sizes=np.linspace(0.1, 1.0, 5),
                                                 scoring='accuracy' if st.session_state.auto_task_type == "classification" else 'neg_mean_squared_error'
                                             )
                                             lc_data = {
                                                 "sizes": train_sizes,
                                                 "train_mean": np.mean(train_scores, axis=1),
                                                 "train_std": np.std(train_scores, axis=1),
                                                 "test_mean": np.mean(test_scores, axis=1),
                                                 "test_std": np.std(test_scores, axis=1)
                                             }
                                         except Exception as e:
                                             lc_data = {"error": str(e)}

                                     m_results.append({
                                         "Seed": seed,
                                         "Spearman Stability": s_stab,
                                         "Top-k Jaccard": s_jac,
                                         "Base Features": X_tr.columns.tolist(),
                                         "Base Imp": base,
                                         "Performance": perf_metrics
                                     })
                                     
                                     # Cache Viz Data
                                     viz_cache = {
                                         "model": model, 
                                         "X_te": X_te, "y_te": y_te, 
                                         "y_pred": y_pred, "y_prob": y_prob,
                                         "lc_data": lc_data,
                                         "epoch_history": epoch_history
                                     }
                                     
                                 except Exception as exc:
                                     status.write(f"Failed {m_name}/{e_name}: {exc}")
                                     success = False
                                     failure_reason = str(exc)
                            
                             if success and m_results:
                                 model_failed_completely = False
                                 dur = time.time() - start_time
                                 avg_s = np.mean([x["Spearman Stability"] for x in m_results])
                                 avg_j = np.mean([x["Top-k Jaccard"] for x in m_results])
                                 
                                 # Avg Performance Metrics
                                 avg_perf = {}
                                 if st.session_state.auto_task_type == "classification":
                                     avg_perf["Accuracy"] = np.mean([x["Performance"]["Accuracy"] for x in m_results])
                                     avg_perf["F1 Score"] = np.mean([x["Performance"]["F1 Score"] for x in m_results])
                                     avg_perf["Precision"] = np.mean([x["Performance"]["Precision"] for x in m_results])
                                     avg_perf["Recall"] = np.mean([x["Performance"]["Recall"] for x in m_results])
                                 else:
                                     avg_perf["RMSE"] = np.mean([x["Performance"]["RMSE"] for x in m_results])
                                     avg_perf["R2 Score"] = np.mean([x["Performance"]["R2 Score"] for x in m_results])
                                     avg_perf["MAE"] = np.mean([x["Performance"]["MAE"] for x in m_results])

                                 run_id = f"{m_name}_{e_name}_{len(st.session_state.experiment_history)}"
                                 st.session_state.experiment_history.append({
                                     "Run ID": run_id, "Model": m_name, "Explainer": e_name,
                                     "Avg Spearman": avg_s, "Avg Jaccard": avg_j, "Duration (s)": dur,
                                     "Performance": avg_perf,
                                     "Raw Res": m_results, "Viz Cache": viz_cache
                                 })
                         
                         if model_failed_completely:
                             st.session_state.failed_models.append({"Model": m_name, "Reason": failure_reason})

                     status.update(label="Analysis Complete!", state="complete", expanded=False)
            
            status_container.success("Analysis Complete! See Results Below.")

            # ==========================================
            # 4. RESULTS DASHBOARD
            # ==========================================
            st.write("---")
            st.header("5. Analysis Results Dashboard")
            
            # Display Failed Models Warning
            if st.session_state.failed_models:
                 st.error("⚠️ Incompatible Models Detected")
                 st.caption("The following models failed to run on this dataset:")
                 fail_df = pd.DataFrame(st.session_state.failed_models)
                 st.dataframe(fail_df, hide_index=True)

            if st.session_state.experiment_history:
                res_df = pd.DataFrame(st.session_state.experiment_history)
                
                # Summary Table
                st.subheader("Performance Summary")
                st.dataframe(res_df[["Model", "Explainer", "Avg Spearman", "Avg Jaccard", "Duration (s)"]].style.background_gradient(cmap="Blues"))
                
                # Performance Metrics Table
                st.subheader("Model Accuracy Metrics")
                perf_df_list = []
                for entry in st.session_state.experiment_history:
                    row = {"Model": entry["Model"], "Explainer": entry["Explainer"]}
                    row.update(entry["Performance"])
                    perf_df_list.append(row)
                
                perf_df = pd.DataFrame(perf_df_list)
                st.dataframe(perf_df.style.background_gradient(cmap="Greens"))
                
                # Comparison Plots
                c_plot1, c_plot2 = st.columns(2)
                with c_plot1:
                    st.write("**Stability Ranking (Spearman)**")
                    fig, ax = plt.subplots()
                    sns.barplot(data=res_df, x="Model", y="Avg Spearman", hue="Explainer", ax=ax, palette="mako")
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    add_watermark(ax)
                    st.pyplot(fig)
                    
                with c_plot2:
                    st.write("**Computational Cost**")
                    fig, ax = plt.subplots()
                    sns.barplot(data=res_df, x="Model", y="Duration (s)", hue="Explainer", ax=ax, palette="Reds")
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    add_watermark(ax)
                    st.pyplot(fig)
                
                # Deep Dive for ALL Models
                st.write("---")
                st.header("Detailed Analysis by Model")
                
                if not res_df.empty:
                    # Iterate through all results
                    for idx, row in res_df.iterrows():
                        m_name = row['Model']
                        e_name = row['Explainer']
                        score = row['Avg Spearman']
                        
                        with st.expander(f"{m_name} + {e_name} (Stability: {score:.3f})", expanded=False):
                            viz_data = row["Viz Cache"]
                            if viz_data:
                                # Get Data
                                y_te = viz_data["y_te"]
                                y_pred = viz_data["y_pred"]
                                y_prob = viz_data["y_prob"]
                                lc_data = viz_data.get("lc_data", None)
                                loss_curve = viz_data.get("loss_curve", None)
                                
                                # TABS for Visuals
                                tab_perf, tab_diag, tab_train, tab_dist = st.tabs([
                                    "Feature Importance & Matrix", 
                                    "ROC & PR Curves", 
                                    "Learning & Loss Curves",
                                    "Prediction Distribution"
                                ])
                                
                                # --- TAB 1: Main Performance ---
                                with tab_perf:
                                    col_d1, col_d2 = st.columns(2)
                                    with col_d1:
                                        st.write(f"**Feature Importance ({e_name})**")
                                        f_names = row["Raw Res"][0]["Base Features"]
                                        f_imps = row["Raw Res"][0]["Base Imp"]
                                        fi_df = pd.DataFrame({"Feature": f_names, "Importance": f_imps}).sort_values("Importance", ascending=False).head(10)
                                        fig, ax = plt.subplots(figsize=(6, 4))
                                        sns.barplot(data=fi_df, y="Feature", x="Importance", ax=ax, palette="viridis")
                                        add_watermark(ax)
                                        st.pyplot(fig)
                                        
                                    with col_d2:
                                        if st.session_state.auto_task_type == "classification":
                                            st.write("**Confusion Matrix**")
                                            cm = confusion_matrix(y_te, y_pred)
                                            fig, ax = plt.subplots(figsize=(4, 4))
                                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                                            add_watermark(ax)
                                            st.pyplot(fig)
                                        else:
                                            st.write("**Predicted vs Actual**")
                                            fig, ax = plt.subplots(figsize=(4, 4))
                                            sns.scatterplot(x=y_te, y=y_pred, ax=ax)
                                            min_val = min(y_te.min(), y_pred.min())
                                            max_val = max(y_te.max(), y_pred.max())
                                            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
                                            add_watermark(ax)
                                            st.pyplot(fig)

                                # --- TAB 2: Diagnostics (ROC/PR) ---
                                with tab_diag:
                                    if st.session_state.auto_task_type == "classification" and y_prob is not None:
                                        col_r1, col_r2 = st.columns(2)
                                        # ROC Curve
                                        with col_r1:
                                            st.write("**ROC Curve**")
                                            try:
                                                # Check if binary or multiclass
                                                if y_prob.ndim == 1 or y_prob.shape[1] == 1: # Binary
                                                    fpr, tpr, _ = roc_curve(y_te, y_prob)
                                                    roc_auc = auc(fpr, tpr)
                                                    fig, ax = plt.subplots()
                                                    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
                                                    ax.plot([0, 1], [0, 1], 'k--')
                                                    ax.set_xlabel('False Positive Rate')
                                                    ax.set_ylabel('True Positive Rate')
                                                    ax.legend()
                                                    add_watermark(ax)
                                                    st.pyplot(fig)
                                                else:
                                                    st.info("Multiclass ROC not currently supported in this view.")
                                            except Exception as e:
                                                st.caption(f"Could not plot ROC: {e}")
                                        
                                        # PR Curve
                                        with col_r2:
                                            st.write("**Precision-Recall Curve**")
                                            try:
                                                if y_prob.ndim == 1 or y_prob.shape[1] == 1:
                                                    precision, recall, _ = precision_recall_curve(y_te, y_prob)
                                                    fig, ax = plt.subplots()
                                                    ax.plot(recall, precision)
                                                    ax.set_xlabel('Recall')
                                                    ax.set_ylabel('Precision')
                                                    add_watermark(ax)
                                                    st.pyplot(fig)
                                            except: pass
                                    elif st.session_state.auto_task_type == "regression":
                                        st.write("**Residual Plot**")
                                        residuals = y_te - y_pred
                                        fig, ax = plt.subplots()
                                        sns.scatterplot(x=y_pred, y=residuals, ax=ax)
                                        ax.axhline(0, color='r', linestyle='--')
                                        ax.set_xlabel("Predicted")
                                        ax.set_ylabel("Residuals")
                                        add_watermark(ax)
                                        st.pyplot(fig)
                                    else:
                                        st.info("Probabilities not available for this model/task.")

                                # --- TAB 3: Training Dynamics ---
                                with tab_train:
                                    col_t1, col_t2 = st.columns(2)
                                    
                                    with col_t1:
                                        st.write("**Learning Curve (Sample Size)**")
                                        if lc_data and "sizes" in lc_data:
                                            fig, ax = plt.subplots()
                                            ax.plot(lc_data["sizes"], lc_data["train_mean"], 'o-', color="r", label="Training score")
                                            ax.plot(lc_data["sizes"], lc_data["test_mean"], 'o-', color="g", label="CV score")
                                            ax.fill_between(lc_data["sizes"], lc_data["train_mean"] - lc_data["train_std"], 
                                                            lc_data["train_mean"] + lc_data["train_std"], alpha=0.1, color="r")
                                            ax.fill_between(lc_data["sizes"], lc_data["test_mean"] - lc_data["test_std"], 
                                                            lc_data["test_mean"] + lc_data["test_std"], alpha=0.1, color="g")
                                            ax.legend(loc="best")
                                            ax.set_xlabel("Training Set Size")
                                            ax.set_ylabel("Score")
                                            add_watermark(ax)
                                            st.pyplot(fig)
                                        else:
                                            st.info("Learning curve data not available.")
                                            if lc_data and "error" in lc_data: st.caption(lc_data["error"])

                                    with col_t2:
                                        st.write("**Model Internal Structure / Training Loss**")
                                        epoch_hist = viz_data.get("epoch_history", {})
                                        
                                        # 1. Iterative Models (Loss Curves)
                                        if epoch_hist and "type" in epoch_hist:
                                            # ... existing loss curve plotting ...
                                            fig, ax = plt.subplots()
                                            
                                            if epoch_hist["type"] == "boost":
                                                data = epoch_hist["data"]
                                                metrics = epoch_hist["metrics"]
                                                for d_name in data:
                                                    for metric in metrics:
                                                         label_name = "Train" if "0" in d_name else "Test"
                                                         if metric in data[d_name]:
                                                             vals = data[d_name][metric]
                                                             ax.plot(vals, label=f"{label_name} {metric}")
                                                ax.set_xlabel("Epochs / Iterations")
                                                ax.set_ylabel("Metric Value")
                                                ax.legend()
                                                
                                            elif epoch_hist["type"] in ["sklearn_mlp", "sklearn_gbm"]:
                                                loss = epoch_hist["loss"]
                                                ax.plot(loss, label="Training Loss", color='red')
                                                ax.set_xlabel("Iterations")
                                                ax.set_ylabel("Loss")
                                                ax.legend()
                                            
                                            add_watermark(ax)
                                            st.pyplot(fig)

                                        # 2. Linear Models (Coefficients)
                                        # Strict Type Check First to avoid ambiguity
                                        elif "Linear" in str(type(model)) or "Logistic" in str(type(model)) or "Ridge" in str(type(model)) or "Lasso" in str(type(model)):
                                            st.caption("Linear Model Weights (Coefficients)")
                                            if hasattr(model, "coef_"):
                                                coefs = model.coef_
                                                if params := getattr(model, "feature_names_in_", None):
                                                    feat_names = params
                                                elif hasattr(viz_data["X_te"], "columns"):
                                                    feat_names = viz_data["X_te"].columns
                                                else:
                                                    feat_names = [f"Feat {i}" for i in range(len(coefs.flatten()))]
                                                
                                                # Handle multi-class coefs (e.g. LogisticReg)
                                                if coefs.ndim > 1:
                                                     # Just plot the first class or average magnitude
                                                     coefs = coefs[0]
                                                
                                                fig, ax = plt.subplots(figsize=(10, 6))
                                                # Sort for better visibility
                                                indices = np.argsort(np.abs(coefs))[::-1][:15] # Top 15
                                                
                                                sns.barplot(x=coefs[indices], y=np.array(feat_names)[indices], ax=ax, palette="viridis")
                                                ax.set_title("Top Model Coefficients")
                                                add_watermark(ax)
                                                st.pyplot(fig)
                                            else:
                                                st.info("Coefficients not available.")

                                        # 3. Tree Models (Tree Viz - Single Tree)
                                        elif hasattr(model, "tree_"): 
                                            from sklearn.tree import plot_tree
                                            st.caption("Decision Tree Structure")
                                            fig, ax = plt.subplots(figsize=(20, 10))
                                            plot_tree(model, max_depth=3, feature_names=viz_data["X_te"].columns, filled=True, ax=ax, fontsize=10)
                                            ax.set_title("Tree Visualization (Depth Limited)")
                                            st.pyplot(fig)
                                            
                                        # 4. Random Forest / Extra Trees (First Tree fallback, or similar to AdaBoost?)
                                        # Keeping it simple for now as RF has many trees too.
                                        elif hasattr(model, "estimators_") and not "AdaBoost" in str(type(model)):
                                            from sklearn.tree import plot_tree
                                            st.caption("Random Forest: Visualization of the 1st Tree (from Forest)")
                                            first_tree = model.estimators_[0]
                                            fig, ax = plt.subplots(figsize=(20, 10))
                                            plot_tree(first_tree, max_depth=3, feature_names=viz_data["X_te"].columns, filled=True, ax=ax, fontsize=10)
                                            ax.set_title("Forest Tree #0")
                                            st.pyplot(fig)

                                        # ... XGB/LGBM/NB blocks follow ...

                                        # 7. AdaBoost Viz (With Slider for ALL learners)
                                        elif "AdaBoost" in str(type(model)):
                                            from sklearn.tree import plot_tree
                                            
                                            # Check if base estimators are trees
                                            if hasattr(model, "estimators_") and len(model.estimators_) > 0:
                                                n_ests = len(model.estimators_)
                                                st.caption(f"AdaBoost Ensemble: {n_ests} Weak Learners.")
                                                
                                                # SLIDER to pick tree
                                                tree_idx = st.slider(f"Select Learner (0-{n_ests-1})", 0, n_ests-1, 0, key=f"ada_slider_{m_name}")
                                                
                                                target_stump = model.estimators_[tree_idx]
                                                
                                                fig, ax = plt.subplots(figsize=(12, 8))
                                                plot_tree(target_stump, feature_names=viz_data["X_te"].columns, filled=True, ax=ax, fontsize=10)
                                                ax.set_title(f"AdaBoost Weak Learner #{tree_idx}")
                                                st.pyplot(fig)
                                            else:
                                                st.info("Estimators not accessible for visualization.")
                                        elif "XGB" in str(type(model)):
                                            import xgboost as xgb
                                            st.caption("XGBoost Tree Structure (First Tree)")
                                            fig, ax = plt.subplots(figsize=(20, 10))
                                            # Plot the 1st tree (index 0)
                                            xgb.plot_tree(model, num_trees=0, ax=ax, rankdir='LR') 
                                            ax.set_title("XGBoost Tree 0")
                                            st.pyplot(fig)
                                        
                                        elif "LGBM" in str(type(model)):
                                            import lightgbm as lgb
                                            st.caption("LightGBM Structure")
                                            try:
                                                # Try Tree Viz first (Requires Graphviz)
                                                fig, ax = plt.subplots(figsize=(20, 10))
                                                lgb.plot_tree(model, tree_index=0, ax=ax, show_info=['split_gain', 'internal_value', 'internal_count', 'leaf_count'])
                                                ax.set_title("LightGBM Tree 0")
                                                st.pyplot(fig)
                                            except Exception as e:
                                                # Fallback to Importance if Graphviz is missing
                                                st.warning(f"Tree viz unavailable (Graphviz missing?). Showing Split Importance instead.")
                                                fig, ax = plt.subplots(figsize=(10, 6))
                                                lgb.plot_importance(model, ax=ax, importance_type='split', max_num_features=15)
                                                ax.set_title("LightGBM Feature Splits (Importance)")
                                                st.pyplot(fig)
                                        
                                        # 5. Naive Bayes (Gaussian Viz)
                                        elif "GaussianNB" in str(type(model)):
                                            # ... existing NB code ...
                                            input_num_cols = viz_data["X_te"].shape[1]
                                            feat_names = viz_data["X_te"].columns
                                            
                                            st.caption("Naive Bayes: Learned Gaussian Distributions (per Class)")
                                            
                                            # GaussianNB stores mean in theta_ and variance in var_
                                            means = model.theta_ # Shape: (n_classes, n_features)
                                            vars_ = model.var_  # Shape: (n_classes, n_features)
                                            classes = model.classes_
                                            
                                            # Plot top 4 features (or less)
                                            n_feats_to_plot = min(input_num_cols, 4)
                                            
                                            fig, axes = plt.subplots(1, n_feats_to_plot, figsize=(5*n_feats_to_plot, 4))
                                            if n_feats_to_plot == 1: axes = [axes]
                                            
                                            from scipy.stats import norm
                                            
                                            for idx in range(n_feats_to_plot):
                                                ax = axes[idx]
                                                f_name = feat_names[idx]
                                                
                                                # Determine range for x-axis
                                                all_means = means[:, idx]
                                                all_stds = np.sqrt(vars_[:, idx])
                                                x_min = (all_means - 3*all_stds).min()
                                                x_max = (all_means + 3*all_stds).max()
                                                x = np.linspace(x_min, x_max, 100)
                                                
                                                for c_i, c_val in enumerate(classes):
                                                    mu = means[c_i, idx]
                                                    sigma = np.sqrt(vars_[c_i, idx])
                                                    y_pdf = norm.pdf(x, mu, sigma)
                                                    ax.plot(x, y_pdf, label=f"Class {c_val}")
                                                    ax.fill_between(x, y_pdf, alpha=0.2)
                                                
                                                ax.set_title(f"Feature: {f_name}")
                                                if idx == 0: ax.legend()
                                            
                                            st.pyplot(fig)
                                        
                                        # 6. AdaBoost Viz
                                        elif "AdaBoost" in str(type(model)):
                                            from sklearn.tree import plot_tree
                                            st.caption("AdaBoost: Visualization of the 1st Weak Learner (Tree/Stump)")
                                            
                                            # Check if base estimators are trees
                                            if hasattr(model, "estimators_") and len(model.estimators_) > 0:
                                                first_stump = model.estimators_[0]
                                                fig, ax = plt.subplots(figsize=(10, 6))
                                                plot_tree(first_stump, feature_names=viz_data["X_te"].columns, filled=True, ax=ax, fontsize=10)
                                                ax.set_title("AdaBoost Weak Learner #1")
                                                st.pyplot(fig)
                                            else:
                                                st.info("Estimators not accessible for visualization.")

                                        # 7. Fallback
                                        else:
                                            m_type = type(model).__name__
                                            if "Neighbor" in m_type:
                                                st.info(f"ℹ️ **{m_name}** is instance-based (KNN). It stores training data points directly. No internal weights or trees to visualize.")
                                            else:
                                                 st.info(f"Visual structure not available for {m_name}")

                                # --- TAB 4: Distributions ---
                                with tab_dist:
                                    col_ds1, col_ds2 = st.columns(2)
                                    with col_ds1:
                                        st.write("**Prediction Distribution**")
                                        fig, ax = plt.subplots()
                                        sns.histplot(y_pred, kde=True, ax=ax, color='purple', label='Predicted')
                                        sns.histplot(y_te, kde=True, ax=ax, color='orange', label='Actual', alpha=0.4)
                                        ax.legend()
                                        add_watermark(ax)
                                        st.pyplot(fig)
                                    with col_ds2:
                                        st.write("**Box Plot of Predictions**")
                                        fig, ax = plt.subplots()
                                        df_box = pd.DataFrame({"Actual": y_te, "Predicted": y_pred})
                                        sns.boxplot(data=df_box, ax=ax)
                                        add_watermark(ax)
                                        st.pyplot(fig)
            else:
                st.warning("No results to display.")

        else:
            st.info("Please select your Target Column above and click 'RUN FULL AUTOMATED ANALYSIS' to begin.")

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Please upload a CSV file to begin.")
