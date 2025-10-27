# ---------------------------------------------------------------
# 🌍 AI for SDG 3: Predicting Disease Outbreak Risk (Model Comparison)
# Approach: Compare K-Means vs DBSCAN (Unsupervised Learning)
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit Page Setup
st.set_page_config(page_title="Disease Outbreak Risk Predictor", page_icon="🌍")
st.title("🌡️ AI for SDG 3: Predicting Disease Outbreak Risk")
st.write("Using **Unsupervised Learning** (K-Means & DBSCAN) to identify high-risk disease outbreak regions.")

# File Uploader
uploaded_file = st.file_uploader("📁 Upload your dataset (climate_disease_dataset.csv)", type="csv")

if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.subheader("📄 Preview of Uploaded Data")
    st.dataframe(data.head())

    # Select features
    features = ['year', 'avg_temp_c', 'precipitation_mm',
                'population_density', 'malaria_cases', 'dengue_cases']
    data = data.dropna(subset=features)
    X = data[features]

    # Normalize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Choose Algorithm
    algo = st.selectbox("🤖 Choose a Clustering Algorithm", ["K-Means", "DBSCAN"])

    if algo == "K-Means":
        n_clusters = st.slider("Select number of clusters (K)", 2, 8, 4)
        model = KMeans(n_clusters=n_clusters, random_state=42)
        data['Cluster'] = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, data['Cluster'])
        st.success(f"✅ K-Means Silhouette Score: {score:.3f}")

    elif algo == "DBSCAN":
        eps = st.slider("Set DBSCAN eps value", 0.1, 5.0, 1.0)
        min_samples = st.slider("Set DBSCAN min_samples", 2, 10, 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        data['Cluster'] = model.fit_predict(X_scaled)

        # DBSCAN may assign some points to -1 (noise)
        if len(set(data['Cluster'])) > 1:
            score = silhouette_score(X_scaled, data[data['Cluster'] != -1]['Cluster'])
            st.success(f"✅ DBSCAN Silhouette Score (excluding noise): {score:.3f}")
        else:
            st.warning("⚠️ All points are noise (-1). Try adjusting eps or min_samples.")

    # Visualize clusters
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X_scaled)
    sns.set(style="whitegrid", palette="viridis")
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=reduced[:,0], y=reduced[:,1], hue=data['Cluster'], palette='viridis', s=70)
    plt.title(f"Disease Outbreak Risk Clusters ({algo})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    st.pyplot(plt)

    # Show cluster summary
    st.subheader("📊 Cluster Summary")
    cluster_summary = data.groupby('Cluster')[features].mean().round(2)
    st.dataframe(cluster_summary)

    # Download clustered data
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Clustered Data", data=csv, file_name=f"clustered_disease_data_{algo}.csv", mime="text/csv")

else:
    st.info("👆 Please upload your dataset to start.")
