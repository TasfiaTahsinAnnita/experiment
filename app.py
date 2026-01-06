
import streamlit as st
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt

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
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay, auc, mean_squared_error, r2_score
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
    # Load data immediately to get columns
    try:
        df = pd.read_csv(uploaded_file)
        
        if "processed_df" not in st.session_state:
            st.session_state.processed_df = None
            
        if "experiment_history" not in st.session_state:
            st.session_state.experiment_history = []
            
        # Tabs for better organization
        tab1, tab2, tab3, tab4 = st.tabs([
            "Exploratory Data Analysis (EDA)", 
            "Data Preprocessing", 
            "Consistency Experiment",
            "Comparisons & Statistics"
        ])
        
        # --- TAB 1: EDA ---
        with tab1:
            st.header("1. Exploratory Data Analysis")
            
            target_col_eda = st.selectbox("Select Target Variable", df.columns, key="eda_target")
            
            # 1. Structural Analysis
            st.subheader("1.1 Structural Analysis")
            buffer = pd.DataFrame({
                "Column": df.columns,
                "Type": df.dtypes.astype(str),
                "Non-Null Count": df.count(),
                "Null Count": df.isnull().sum(),
                "Unique Values": df.nunique()
            }).reset_index(drop=True)
            st.dataframe(buffer)

            # 2. Descriptive Statistics
            st.subheader("1.2 Descriptive Statistics")
            st.dataframe(df.describe())
            
            # 3. Missing Value Analysis
            st.subheader("1.3 Missing Value Analysis")
            missing = df.isnull().sum()
            if missing.sum() > 0:
                st.bar_chart(missing[missing > 0])
            else:
                st.success("No missing values found!")

            # 4. Distribution Analysis
            st.subheader("1.4 Distribution Analysis")
            dist_col = st.selectbox("Select Feature for Distribution", df.columns, key="dist_col")
            fig, ax = plt.subplots()
            if pd.api.types.is_numeric_dtype(df[dist_col]):
                sns.histplot(df[dist_col], kde=True, ax=ax)
            else:
                df[dist_col].value_counts().plot(kind='bar', ax=ax)
            add_watermark(ax)
            st.pyplot(fig)

            # 5. Outlier Analysis (Visual)
            st.subheader("1.5 Outlier Analysis (Boxplot)")
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                outlier_col = st.selectbox("Select Feature for Boxplot", num_cols, key="out_col")
                fig, ax = plt.subplots()
                sns.boxplot(x=df[outlier_col], ax=ax)
                add_watermark(ax)
                st.pyplot(fig)

            # 6. Correlation Analysis
            st.subheader("1.6 Correlation Analysis")
            if len(num_cols) > 1:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                add_watermark(ax)
                st.pyplot(fig)
            
            # 7. Feature-Target Relationship
            st.subheader("1.7 Feature-Target Relationship")
            ft_col = st.selectbox("Select Feature to Compare with Target", df.columns, key="ft_col")
            fig, ax = plt.subplots()
            
            # Determine plot type based on dtypes
            target_is_num = pd.api.types.is_numeric_dtype(df[target_col_eda])
            feat_is_num = pd.api.types.is_numeric_dtype(df[ft_col])
            
            if target_is_num and feat_is_num:
                sns.scatterplot(x=df[ft_col], y=df[target_col_eda], ax=ax)
            elif not target_is_num and feat_is_num:
                sns.boxplot(x=df[target_col_eda], y=df[ft_col], ax=ax)
            elif target_is_num and not feat_is_num:
                sns.boxplot(x=df[ft_col], y=df[target_col_eda], ax=ax)
            else:
                # Cat vs Cat - Heatmap of contingency table
                ct = pd.crosstab(df[ft_col], df[target_col_eda])
                sns.heatmap(ct, annot=True, fmt='d', cmap="YlGnBu", ax=ax)
                
            add_watermark(ax)
            st.pyplot(fig)

            # Advanced Mining sections...
            st.markdown("---")
            st.header("Advanced Data Mining")
            # ... (PCA logic reused)
            if len(num_cols) > 0:
                X_pca = df[num_cols].fillna(df[num_cols].mean())
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_pca)
                
                col3, col4 = st.columns(2)
                with col3:
                    st.write("### PCA Projection (2D)")
                    pca = PCA(n_components=2)
                    components = pca.fit_transform(X_scaled)
                    pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
                    if target_col_eda in df.columns:
                        pca_df['Target'] = df[target_col_eda].values if len(df) == len(pca_df) else None
                        fig, ax = plt.subplots()
                        if pca_df['Target'] is not None:
                            sns.scatterplot(x='PC1', y='PC2', hue='Target', data=pca_df, ax=ax)
                        else:
                            sns.scatterplot(x='PC1', y='PC2', data=pca_df, ax=ax)
                        add_watermark(ax)
                        st.pyplot(fig)

                with col4:
                    st.write("### Outlier Detection (Isolation Forest)")
                    iso = IsolationForest(contamination=0.05, random_state=42)
                    outliers = iso.fit_predict(X_scaled)
                    pca_df['Outlier'] = outliers
                    st.metric("Potential Outliers", f"{(outliers == -1).sum()}")
                    fig, ax = plt.subplots()
                    sns.scatterplot(x='PC1', y='PC2', hue='Outlier', data=pca_df, palette={1: 'green', -1: 'red'}, ax=ax)
                    add_watermark(ax)
                    st.pyplot(fig)
        
        # --- TAB 2: PREPROCESSING ---
        with tab2:
            st.header("2. Data Preprocessing Pipeline")
            st.markdown("Configure how to process your data before running the experiment.")
            
            prep_df = df.copy() # Start with fresh copy for preview
            
            # A. Missing Value Imputation
            st.subheader("2.1 Missing Value Imputation")
            impute_strategy = st.selectbox("Numeric Imputation Strategy", ["mean", "median", "most_frequent", "constant (0)"])
            
            # B. Categorical Encoding
            st.subheader("2.2 Categorical Encoding")
            cat_strategy = st.radio("Encoding Method", ["One-Hot Encoding (get_dummies)", "Label Encoding"])
            
            # C. Outlier Treatment
            st.subheader("2.3 Outlier Treatment")
            outlier_method = st.checkbox("Remove Outliers (Isolation Forest filtering)?")
            
            # D. Feature Engineering
            st.subheader("2.4 Feature Engineering")
            poly_features = st.checkbox("Add Polynomial Features (Degree 2)?")
            
            # E. Scaling
            st.subheader("2.5 Feature Scaling")
            scaler_option = st.selectbox("Scaling Method", ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"])
            
            # F. Feature Selection
            st.subheader("2.6 Feature Selection")
            selection_k = st.number_input("Select K Best Features (0 = Keep All)", min_value=0, max_value=len(df.columns), value=0)

            if st.button("Apply Preprocessing & Save for Experiment"):
                with st.spinner("Processing..."):
                    # 1. Target Separation (Don't process target yet)
                    target_col = target_col_eda # Use the one selected in EDA
                    if target_col not in prep_df.columns:
                        st.error(f"Target column {target_col} not found.")
                    else:
                        prep_df = prep_df.dropna(subset=[target_col])
                        y_prep = prep_df[target_col]
                        X_prep = prep_df.drop(columns=[target_col])
                        
                        # 2. Imputation (Numeric)
                        num_cols_prep = X_prep.select_dtypes(include=[np.number]).columns
                        cat_cols_prep = X_prep.select_dtypes(exclude=[np.number]).columns
                        
                        if impute_strategy == "constant (0)":
                             X_prep[num_cols_prep] = X_prep[num_cols_prep].fillna(0)
                        else:
                             imp = SimpleImputer(strategy=impute_strategy)
                             if len(num_cols_prep) > 0:
                                X_prep[num_cols_prep] = imp.fit_transform(X_prep[num_cols_prep])
                        
                        # Fill categorical NaNs with 'Missing'
                        X_prep[cat_cols_prep] = X_prep[cat_cols_prep].fillna("Missing")

                        # 3. Categorical Encoding
                        if cat_strategy.startswith("One-Hot"):
                            X_prep = pd.get_dummies(X_prep, columns=cat_cols_prep, drop_first=True)
                        else:
                            le = LabelEncoder()
                            for col in cat_cols_prep:
                                X_prep[col] = le.fit_transform(X_prep[col].astype(str))
                        
                        # 4. Outlier Removal (Rows)
                        if outlier_method:
                            iso = IsolationForest(contamination=0.05, random_state=42)
                            # Fit on current X
                            preds = iso.fit_predict(X_prep)
                            mask = preds != -1
                            X_prep = X_prep[mask]
                            y_prep = y_prep[mask]
                            st.info(f"Removed {(~mask).sum()} outlier rows.")

                        # 5. Feature Engineering
                        if poly_features:
                            poly = PolynomialFeatures(degree=2, include_bias=False)
                            # Only apply to numeric columns to verify size? Or all? Poly on one-hot is huge.
                            # Just apply to everything (safe) or limit. Let's apply.
                            X_poly = poly.fit_transform(X_prep)
                            feat_names = poly.get_feature_names_out(X_prep.columns)
                            X_prep = pd.DataFrame(X_poly, columns=feat_names, index=X_prep.index)
                        
                        # 6. Scaling
                        if scaler_option != "None":
                            if scaler_option == "StandardScaler": s = StandardScaler()
                            elif scaler_option == "MinMaxScaler": s = MinMaxScaler()
                            else: s = RobustScaler()
                            
                            X_scaled = s.fit_transform(X_prep)
                            X_prep = pd.DataFrame(X_scaled, columns=X_prep.columns, index=X_prep.index)

                        # 7. Feature Selection
                        if selection_k > 0 and selection_k < X_prep.shape[1]:
                            # Needs target. Check task type.
                            is_regression = pd.api.types.is_numeric_dtype(y_prep) and len(y_prep.unique()) > 20
                            score_func = f_regression if is_regression else f_classif
                            
                            selector = SelectKBest(score_func=score_func, k=selection_k)
                            X_new = selector.fit_transform(X_prep, y_prep)
                            selected_indices = selector.get_support(indices=True)
                            selected_cols = X_prep.columns[selected_indices]
                            X_prep = pd.DataFrame(X_new, columns=selected_cols, index=X_prep.index)
                            st.info(f"Selected top {selection_k} features: {selected_cols.tolist()}")

                        # Recombine for storage (optional, or just store X and y)
                        # We store X_prep and y_prep separate or combined?
                        # load_data currently expects a dataframe or path.
                        # It splits X and y itself. So let's recombine.
                        processed_data = X_prep.copy()
                        processed_data[target_col] = y_prep.values
                        
                        st.session_state.processed_df = processed_data
                        st.session_state.target_col = target_col
                        
                        st.success("Preprocessing Complete! Data saved for Experiment.")
                        st.write("### Processed Data Preview")
                        st.dataframe(processed_data.head())
                        st.write(f"Shape: {processed_data.shape}")

        # --- TAB 4: COMPARISONS & STATISTICS ---
        with tab4:
            st.header("4. Comparisons & Statistics")
            
            if not st.session_state.experiment_history:
                st.info("No experiments found in history. Run some experiments in the 'Consistency Experiment' tab first.")
            else:
                history_df = pd.DataFrame(st.session_state.experiment_history)
                st.write("### Experiment History")
                st.dataframe(history_df[["Run ID", "Model", "Explainer", "Avg Spearman", "Avg Jaccard", "Duration (s)"]])
                
                # 5. Compare Results sections
                st.markdown("---")
                st.header("5. Compare Results")
                
                comp_col1, comp_col2 = st.columns(2)
                
                with comp_col1:
                    st.subheader("Model Performance Comparison")
                    metric_to_plot = st.selectbox("Select Metric to Compare", ["Avg Spearman", "Avg Jaccard", "Duration (s)"])
                    fig, ax = plt.subplots()
                    sns.barplot(x="Model", y=metric_to_plot, hue="Explainer", data=history_df, ax=ax)
                    plt.xticks(rotation=45)
                    add_watermark(ax)
                    st.pyplot(fig)
                    
                with comp_col2:
                    st.subheader("Computational Complexity")
                    fig, ax = plt.subplots()
                    sns.barplot(x="Model", y="Duration (s)", data=history_df, ax=ax, palette="viridis")
                    plt.xticks(rotation=45)
                    plt.ylabel("Time (seconds)")
                    add_watermark(ax)
                    st.pyplot(fig)

                # 6. Statistical Validation
                st.markdown("---")
                st.header("6. Statistical Validation")
                st.write("Perform hypothesis testing to check if the difference in stability between two models is statistically significant.")
                
                stat_col1, stat_col2 = st.columns(2)
                
                with stat_col1:
                    model_a_id = st.selectbox("Select Model A (Control/Baseline)", history_df["Run ID"].unique(), key="mod_a")
                    model_b_id = st.selectbox("Select Model B (Treatment/Proposed)", history_df["Run ID"].unique(), key="mod_b")
                    
                    target_metric_stat = st.selectbox("Metric for Testing", ["Spearman Stability", "Top-k Jaccard"])
                    
                with stat_col2:
                    if st.button("Run Statistical Test"):
                        # Retrieve raw data
                        entry_a = next(item for item in st.session_state.experiment_history if item["Run ID"] == model_a_id)
                        entry_b = next(item for item in st.session_state.experiment_history if item["Run ID"] == model_b_id)
                        
                        scores_a = [x[target_metric_stat] for x in entry_a["Raw Res"]]
                        scores_b = [x[target_metric_stat] for x in entry_b["Raw Res"]]
                        
                        # T-Test
                        t_stat, p_val = stats.ttest_ind(scores_a, scores_b)
                        
                        # Effect Size (Cohen's d)
                        mean_diff = np.mean(scores_a) - np.mean(scores_b)
                        pooled_std = np.sqrt((np.std(scores_a)**2 + np.std(scores_b)**2) / 2)
                        cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0
                        
                        st.write(f"### Results ({target_metric_stat})")
                        st.metric("P-Value", f"{p_val:.5f}")
                        st.metric("T-Statistic", f"{t_stat:.3f}")
                        st.metric("Cohen's d (Effect Size)", f"{cohens_d:.3f}")
                        
                        if p_val < 0.05:
                            st.success("Statistically Significant Difference (p < 0.05)!")
                        else:
                            st.warning("No Statistically Significant Difference (p >= 0.05).")
                            
                        # Confidence Intervals
                        st.write("#### 95% Confidence Intervals")
                        ci_a = stats.t.interval(0.95, len(scores_a)-1, loc=np.mean(scores_a), scale=stats.sem(scores_a))
                        ci_b = stats.t.interval(0.95, len(scores_b)-1, loc=np.mean(scores_b), scale=stats.sem(scores_b))
                        
                        st.write(f"**{model_a_id}**: {ci_a}")
                        st.write(f"**{model_b_id}**: {ci_b}")

                # Robustness / Cross Model Evaluation
                st.markdown("---")
                st.subheader("Cross-Model Evaluation Matrix")
                pivot_df = history_df.pivot_table(index="Model", columns="Explainer", values="Avg Spearman", aggfunc="mean")
                st.dataframe(pivot_df)
                fig, ax = plt.subplots()
                sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", ax=ax)
                plt.title("Average Spearman Stability Matrix")
                add_watermark(ax)
                st.pyplot(fig)
        # --- TAB 3: EXPERIMENT ---
        with tab3:
            st.header("3. Consistency Experiment")
            
            if st.session_state.processed_df is None:
                st.warning("Please go to the 'Data Preprocessing' tab and click 'Apply' first.")
                exp_df = df # Fallback to raw
                exp_target = df.columns[0]
            else:
                st.success("Using Preprocessed Data")
                exp_df = st.session_state.processed_df
                exp_target = st.session_state.target_col

            # Config inputs
            st.info(f"Target Variable: **{exp_target}**")
            target_col = exp_target # Override
            
            # --- MODEL SELECTION ---
            task_type = st.sidebar.selectbox("Task Type", ["classification", "regression"])
            
            if task_type == "classification":
                model_options = [
                    "logistic", "random_forest", "decision_tree", "svm", "knn", "naive_bayes", 
                    "gbm", "xgboost", "lightgbm", "adaboost", "extra_trees", "mlp", "dummy"
                ]
            else:
                model_options = [
                    "linear", "ridge", "lasso", "random_forest", "decision_tree", "svm", "knn", 
                    "gbm", "xgboost", "lightgbm", "adaboost", "extra_trees", "mlp", "dummy"
                ]
            
            model_type = st.sidebar.selectbox("Model Type", model_options)
            explainer_type = st.sidebar.selectbox("Explanation Method", ["SHAP", "LIME"])
            
            noise_level = st.sidebar.slider("Noise Level", 0.0, 0.2, 0.02, 0.01)
            num_seeds = st.sidebar.slider("Number of Random Seeds", 1, 10, 5)

            run_btn = st.sidebar.button("Run Analysis")
            
            st.sidebar.markdown("---")
            st.sidebar.caption("Made by TasfiaTahsinAnnita")

            if run_btn:
                st.warning("Running experiment... This may take a while depending on dataset size.")
                
                results_data = []
                
                progress_bar = st.progress(0)
                
                seeds = list(range(1, num_seeds + 1))
                
                start_time_all = time.time()
                for i, seed in enumerate(seeds):
                    try:
                        loop_start = time.time()
                        # Load and split data
                        X_train, X_test, y_train, y_test = load_data(
                            exp_df,
                            target_col,
                            seed
                        )
                        
                        # Train model
                        model = get_model(model_type, seed, task_type=task_type)
                        model.fit(X_train, y_train)

                        # Select Explainer
                        explain_func = shap_explain if explainer_type == "SHAP" else lime_explain

                        # Base Explanation
                        # Pass task_type to explainer
                        base_exp = explain_func(model, X_train, X_test, task_type=task_type)

                        # Noisy Explanation
                        X_test_noisy = add_noise(X_test, noise_level)
                        noisy_exp = explain_func(model, X_train, X_test_noisy, task_type=task_type)

                        # Calculate Metrics
                        stability = spearman_stability(base_exp, noisy_exp)
                        jaccard = top_k_jaccard(base_exp, noisy_exp)

                        if i == 0:
                            st.write(f"### Feature Importance ({explainer_type}) - Seed {seed} (Base vs Noisy)")
                            
                            # Create comparison dataframe
                            feature_names = X_train.columns
                            importance_df = pd.DataFrame({
                                "Feature": feature_names,
                                "Base Importance": base_exp,
                                "Noisy Importance": noisy_exp
                            }).sort_values(by="Base Importance", ascending=False).head(10)
                            
                            st.dataframe(importance_df)
                            
                            # Plot comparison
                            fig, ax = plt.subplots(figsize=(10, 6))
                            importance_df.set_index("Feature").plot(kind="bar", ax=ax)
                            plt.title(f"Top 10 Feature Importances ({explainer_type})")
                            plt.ylabel("Mean Absolute Value")
                            add_watermark(ax)
                            st.pyplot(fig)

                            # --- NEW: Model Evaluation Visualizations ---
                            st.subheader(f"Model Performance Analysis (Seed {seed})")
                            col_eval1, col_eval2 = st.columns(2)
                            
                            if task_type == "classification":
                                y_pred = model.predict(X_test)
                                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
                                
                                # 1. Confusion Matrix
                                with col_eval1:
                                    st.write("#### Confusion Matrix")
                                    cm = confusion_matrix(y_test, y_pred)
                                    fig, ax = plt.subplots()
                                    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax, cmap='Blues')
                                    add_watermark(ax)
                                    st.pyplot(fig)

                                # 2. ROC & Precision-Recall (if proba available)
                                with col_eval2:
                                    if y_proba is not None:
                                        st.write("#### ROC Curve")
                                        fpr, tpr, _ = roc_curve(y_test, y_proba)
                                        roc_auc = auc(fpr, tpr)
                                        fig, ax = plt.subplots()
                                        ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
                                        ax.plot([0, 1], [0, 1], 'k--')
                                        ax.set_xlabel('False Positive Rate')
                                        ax.set_ylabel('True Positive Rate')
                                        ax.legend(loc='lower right')
                                        add_watermark(ax)
                                        st.pyplot(fig)
                                        
                                        st.write("#### Precision-Recall Curve")
                                        precision, recall, _ = precision_recall_curve(y_test, y_proba)
                                        fig, ax = plt.subplots()
                                        ax.plot(recall, precision)
                                        ax.set_xlabel('Recall')
                                        ax.set_ylabel('Precision')
                                        add_watermark(ax)
                                        st.pyplot(fig)
                                    else:
                                        st.info("Probability scores not available for ROC/PR curves.")
                            else: # Regression
                                from sklearn.metrics import mean_squared_error, r2_score
                                y_pred = model.predict(X_test)
                                mse = mean_squared_error(y_test, y_pred)
                                r2 = r2_score(y_test, y_pred)
                                
                                with col_eval1:
                                    st.metric("MSE", f"{mse:.4f}")
                                    st.metric("R2 Score", f"{r2:.4f}")
                                
                                with col_eval2:
                                    st.write("#### Predicted vs Actual")
                                    fig, ax = plt.subplots()
                                    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
                                    min_val = min(y_test.min(), y_pred.min())
                                    max_val = max(y_test.max(), y_pred.max())
                                    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
                                    ax.set_xlabel("Actual")
                                    ax.set_ylabel("Predicted")
                                    add_watermark(ax)
                                    st.pyplot(fig)

                            # 3. Learning Curve (Computationally expensive, use subset/cv=3)
                            st.write("#### Learning Curve (Sample)")
                            train_sizes, train_scores, test_scores = learning_curve(
                                model, X_train, y_train, cv=3, n_jobs=-1, 
                                train_sizes=np.linspace(0.1, 1.0, 5),
                                scoring='accuracy' if task_type == 'classification' else 'neg_mean_squared_error'
                            )
                            
                            train_mean = np.mean(train_scores, axis=1)
                            test_mean = np.mean(test_scores, axis=1)
                            
                            fig, ax = plt.subplots()
                            ax.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
                            ax.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
                            ax.set_xlabel("Training examples")
                            ax.set_ylabel("Score")
                            ax.legend(loc="best")
                            add_watermark(ax)
                            st.pyplot(fig)

                        # Collect importance data for comprehensive plots
                        results_data.append({
                            "Seed": seed,
                            "Spearman Stability": stability,
                            "Top-k Jaccard": jaccard,
                            "Base Importance": base_exp,
                            "Noisy Importance": noisy_exp,
                            "Features": X_train.columns.tolist()
                        })
                    
                    except Exception as e:
                        st.error(f"Error on seed {seed}: {e}")
                
                # Update progress
                progress_bar.progress((i + 1) / len(seeds))

                # Display Results
                if results_data:
                    results_df = pd.DataFrame(results_data)
                    
                    st.write(f"### Results per Seed ({explainer_type})")
                    st.dataframe(results_df[["Seed", "Spearman Stability", "Top-k Jaccard"]])

                    avg_stability = results_df["Spearman Stability"].mean()
                    avg_jaccard = results_df["Top-k Jaccard"].mean()

                    col1, col2 = st.columns(2)
                    col1.metric("Average Spearman Stability", f"{avg_stability:.3f}")
                    col2.metric("Average Top-k Jaccard", f"{avg_jaccard:.3f}")

                    duration = time.time() - start_time_all
                    
                    # --- Save to History ---
                    # Create a summary entry
                    run_id = f"{model_type}_{explainer_type}_{len(st.session_state.experiment_history)}"
                    history_entry = {
                        "Run ID": run_id,
                        "Model": model_type,
                        "Explainer": explainer_type,
                        "Task": task_type,
                        "Noise": noise_level,
                        "Avg Spearman": avg_stability,
                        "Avg Jaccard": avg_jaccard,
                        "Duration (s)": duration,
                        "Raw Res": results_data
                    }
                    st.session_state.experiment_history.append(history_entry)
                    st.success(f"Experiment saved to history as {run_id}")

                    # Visualization Area
                    st.write("---")
                    st.header("Detailed Analysis")

                    # 1. Consistency Metrics Plot
                    st.write("### 1. Stability Metrics per Seed")
                    fig, ax = plt.subplots()
                    results_df.set_index("Seed")[["Spearman Stability", "Top-k Jaccard"]].plot(kind="bar", ax=ax)
                    plt.ylim(0, 1.1)
                    add_watermark(ax)
                    st.pyplot(fig)
                    
                    # Prepare data for aggregated plots
                    # We use the first valid seed's feature names
                    feature_names = results_data[0]["Features"]
                    
                    # 2. Scatter Plot: Base vs Noisy Importance (All Seeds)
                    st.write("### 2. Feature Stability Scatter (Base vs Noisy)")
                    st.markdown("Points on the diagonal line indicate perfect stability. Deviations show instability.")
                    
                    scatter_data = []
                    for res in results_data:
                        for f, base, noisy in zip(res["Features"], res["Base Importance"], res["Noisy Importance"]):
                            scatter_data.append({"Feature": f, "Base": base, "Noisy": noisy, "Seed": res["Seed"]})
                    
                    scatter_df = pd.DataFrame(scatter_data)
                    
                    fig, ax = plt.subplots(figsize=(8, 8))
                    sns.scatterplot(data=scatter_df, x="Base", y="Noisy", hue="Feature", style="Seed", ax=ax)
                    
                    # Add diagonal line
                    max_val = max(scatter_df["Base"].max(), scatter_df["Noisy"].max())
                    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
                    ax.set_aspect('equal')
                    add_watermark(ax)
                    st.pyplot(fig)

                    # 3. Rank Variation Plot (Bump Chart-ish)
                    st.write("### 3. Feature Rank Changes (Base -> Noisy) [First Seed]")
                    
                    # Use only first seed for clarity
                    first_seed_data = results_data[0]
                    base_ranks = pd.Series(first_seed_data["Base Importance"], index=feature_names).rank(ascending=False)
                    noisy_ranks = pd.Series(first_seed_data["Noisy Importance"], index=feature_names).rank(ascending=False)
                    
                    rank_df = pd.DataFrame({"Feature": feature_names, "Rank Base": base_ranks, "Rank Noisy": noisy_ranks})
                    rank_df = rank_df.sort_values("Rank Base")
                    
                    # Take top 10 features for cleaner plot
                    top_10_rank_df = rank_df.nsmallest(10, "Rank Base")

                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Draw lines connecting ranks
                    for idx, row in top_10_rank_df.iterrows():
                        ax.plot([0, 1], [row["Rank Base"], row["Rank Noisy"]], marker='o', label=row["Feature"])
                        
                    ax.set_xticks([0, 1])
                    ax.set_xticklabels(["Original Data", "Noisy Data"])
                    ax.set_ylabel("Feature Rank (Lower is Better)")
                    ax.invert_yaxis() # Rank 1 at top
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.title("Rank Stability of Top 10 Features")
                    add_watermark(ax)
                    st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Please upload a CSV file to begin.")
