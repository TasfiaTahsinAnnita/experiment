
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
from sklearn.preprocessing import StandardScaler

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
        
        # Tabs for better organization
        tab1, tab2 = st.tabs(["Data Analysis", "Consistency Experiment"])
        
        with tab1:
            st.header("Exploratory Data Analysis")
            
            st.write("### Dataset Statistics")
            st.dataframe(df.describe())
            
            st.write("### Missing Values")
            missing = df.isnull().sum()
            st.bar_chart(missing)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### Numeric Feature Correlation")
                # Select only numeric columns
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                    add_watermark(ax)
                    st.pyplot(fig)
                else:
                    st.warning("No numeric columns found for correlation matrix.")
            
            with col2:
                target_col_eda = st.selectbox("Select Target to Visualize", df.columns, key="eda_target")
                st.write(f"### Distribution of {target_col_eda}")
                fig, ax = plt.subplots()
                if pd.api.types.is_numeric_dtype(df[target_col_eda]):
                    df[target_col_eda].hist(ax=ax, bins=20)
                else:
                    df[target_col_eda].value_counts().plot(kind='bar', ax=ax)
                add_watermark(ax)
                st.pyplot(fig)

            st.markdown("---")
            st.header("Advanced Data Mining")
            
            # Use numeric features for PCA/Clustering
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                # Handle NaNs for PCA
                X_pca = df[numeric_cols].fillna(df[numeric_cols].mean())
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_pca)
                
                col3, col4 = st.columns(2)
                
                with col3:
                    st.write("### PCA Projection (2D)")
                    pca = PCA(n_components=2)
                    components = pca.fit_transform(X_scaled)
                    pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
                    
                    # Color by target if selected (and if it matches length)
                    if target_col_eda in df.columns and len(df) == len(pca_df):
                        pca_df['Target'] = df[target_col_eda].values
                        fig, ax = plt.subplots()
                        sns.scatterplot(x='PC1', y='PC2', hue='Target', data=pca_df, ax=ax)
                    else:
                        fig, ax = plt.subplots()
                        sns.scatterplot(x='PC1', y='PC2', data=pca_df, ax=ax)
                    
                    add_watermark(ax)
                    st.pyplot(fig)
                    st.caption(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")

                with col4:
                    st.write("### Outlier Detection using Isolation Forest")
                    iso = IsolationForest(contamination=0.05, random_state=42)
                    outliers = iso.fit_predict(X_scaled)
                    outlier_count = (outliers == -1).sum()
                    
                    st.metric("Potential Outliers Detected", f"{outlier_count} (approx 5%)")
                    
                    # Visualizing outliers on PCA
                    pca_df['Outlier'] = outliers
                    fig, ax = plt.subplots()
                    sns.scatterplot(x='PC1', y='PC2', hue='Outlier', data=pca_df, palette={1: 'green', -1: 'red'}, ax=ax)
                    plt.title("Outliers (Red) in PCA Space")
                    add_watermark(ax)
                    st.pyplot(fig)
            else:
                st.warning("Not enough numeric data for advanced mining.")

        with tab2:
            st.header("Consistency Experiment")
            
            # Config inputs
            target_col = st.sidebar.selectbox("Select Target Column", df.columns, index=0 if len(df.columns) > 0 else 0)
            
            model_type = st.sidebar.selectbox("Model Type", ["random_forest", "logistic"])
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
                
                for i, seed in enumerate(seeds):
                    try:
                        # Load and split data
                        X_train, X_test, y_train, y_test = load_data(
                            df,
                            target_col,
                            seed
                        )
                        
                        # Train model
                        model = get_model(model_type, seed)
                        model.fit(X_train, y_train)

                        # Select Explainer
                        explain_func = shap_explain if explainer_type == "SHAP" else lime_explain

                        # Base Explanation
                        base_exp = explain_func(model, X_train, X_test)

                        # Noisy Explanation
                        X_test_noisy = add_noise(X_test, noise_level)
                        noisy_exp = explain_func(model, X_train, X_test_noisy)

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
