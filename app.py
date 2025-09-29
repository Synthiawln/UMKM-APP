#Import Library
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

# ======================================================
# 1. FUNGSI BANTUAN
# ======================================================
def compute_sse(data, centroids, labels):
    return np.sum((data - centroids[labels])**2)

def objective_function(position, k, d, data):
    centroids = position.reshape((k, d))
    distances = cdist(data, centroids)
    labels = np.argmin(distances, axis=1)
    return compute_sse(data, centroids, labels)

def GWO(objective_func, lb, ub, dim, k, d, data, pop_size=50, epochs=200):
    alpha, beta, delta = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    alpha_score, beta_score, delta_score = np.inf, np.inf, np.inf
    population = np.random.uniform(lb, ub, (pop_size, dim))

    for t in range(epochs):
        for i in range(pop_size):
            fitness = objective_func(population[i], k, d, data)
            if fitness < alpha_score:
                alpha_score, alpha = fitness, population[i].copy()
            elif fitness < beta_score:
                beta_score, beta = fitness, population[i].copy()
            elif fitness < delta_score:
                delta_score, delta = fitness, population[i].copy()

        a = 2 - t * (2 / epochs)
        for i in range(pop_size):
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                A1, C1 = 2 * a * r1 - a, 2 * r2
                D_alpha = abs(C1 * alpha[j] - population[i][j])
                X1 = alpha[j] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2, C2 = 2 * a * r1 - a, 2 * r2
                D_beta = abs(C2 * beta[j] - population[i][j])
                X2 = beta[j] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3, C3 = 2 * a * r1 - a, 2 * r2
                D_delta = abs(C3 * delta[j] - population[i][j])
                X3 = delta[j] - A3 * D_delta

                population[i][j] = (X1 + X2 + X3) / 3

        population = np.clip(population, lb, ub)

    return alpha, alpha_score

def plot_elbow(sse_values, K_range):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(K_range, sse_values, 'bo-')
    ax.set_xlabel("Jumlah Cluster (k)")
    ax.set_ylabel("SSE (Sum of Squared Error)")
    ax.set_title("Metode Elbow untuk Menentukan k Optimal")
    return fig

def plot_clusters_side_by_side(data, labels1, centroids1, labels2, centroids2):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # KMeans standar
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels1,
                    palette='Set2', s=50, ax=axes[0])
    axes[0].scatter(centroids1[:, 0], centroids1[:, 1],
                    c='red', s=100, marker='X', label='Centroid')
    axes[0].set_title("KMeans Clustering")
    axes[0].legend()

    # GWO-KMeans
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels2,
                    palette='Set2', s=50, ax=axes[1])
    axes[1].scatter(centroids2[:, 0], centroids2[:, 1],
                    c='red', s=100, marker='X', label='Centroid')
    axes[1].set_title("GWO-KMeans Clustering")
    axes[1].legend()

    plt.tight_layout()
    return fig

# ======================================================
# 2. STREAMLIT APP
# ======================================================

# Judul aplikasi
st.title("Analisis Clustering UMKM Kuliner")

# Upload dataset
uploaded_file = st.file_uploader("Upload dataset Excel (.xlsx)", type=["xlsx"])

# --- 1. Load Dataset ---
if uploaded_file is not None:
    kuliner_data = pd.read_excel(uploaded_file, sheet_name="UKM Kuliner")
    st.info("✅ Dataset berhasil diupload!")
else:
    kuliner_data = pd.read_excel("Data Set UMKM.xlsx", sheet_name="UKM Kuliner")
    st.info("📂 Tidak ada upload. Menggunakan dataset bawaan.")

    # Tampilkan preview dataset
    st.subheader("Preview Dataset")
    st.dataframe(kuliner_data.head(20))

    # Info dataset
    st.subheader("Informasi Dataset")
    st.write("Jumlah baris dan kolom:", kuliner_data.shape)
    st.write("Nama kolom:", kuliner_data.columns.tolist())


 # --- Cleaning ---
    kuliner_data = kuliner_data.dropna().drop_duplicates()
    st.write("Jumlah data setelah cleaning:", kuliner_data.shape)

    # --- Label Encoding ---
    label_encoder = LabelEncoder()
    kuliner_data['Jenis_Kelamin'] = label_encoder.fit_transform(kuliner_data['Jenis Kelamin'])
    kuliner_data['Pendidikan']    = label_encoder.fit_transform(kuliner_data['Pendidikan Terakhir'])
    kuliner_data['Omset']         = label_encoder.fit_transform(kuliner_data['Omset per-Tahun'])
    kuliner_data['Kepemilikan']   = label_encoder.fit_transform(kuliner_data['Status Kepemilkan Tanah/Bangunan'])
    kuliner_data['Sarana_Media']  = label_encoder.fit_transform(kuliner_data['Sarana Media Elektronik'])

    st.subheader("Hasil Label Encoding")
    st.dataframe(kuliner_data.head(20))

    # --- Normalisasi ---
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(
        kuliner_data[['Jenis_Kelamin','Pendidikan','Omset','Kepemilikan','Sarana_Media']]
    )

    data_scaled_df = pd.DataFrame(
        data_scaled,
        columns=['Jenis_Kelamin','Pendidikan','Omset','Kepemilikan','Sarana_Media']
    )
    st.subheader("Hasil Normalisasi (MinMaxScaler)")
    st.dataframe(data_scaled_df.head(20))

    # --- Elbow Method ---
    sse_values = []
    K_range = range(2, 11)
    for k_test in K_range:
        kmeans = KMeans(n_clusters=k_test, random_state=42)
        kmeans.fit(data_scaled)
        sse_values.append(kmeans.inertia_)

    st.subheader("📉 Elbow Method")
    st.pyplot(plot_elbow(sse_values, K_range))

    # --- Pilih jumlah cluster ---
    k = st.slider("Pilih jumlah cluster (k)", 2, 10, 4)

    # --- GWO Optimizer ---
    d = data_scaled.shape[1]
    dim = k * d
    lb = np.tile(np.min(data_scaled, axis=0), k)
    ub = np.tile(np.max(data_scaled, axis=0), k)

    best_position, best_sse = GWO(objective_function, lb, ub, dim, k, d, data_scaled)
    best_centroids = best_position.reshape((k, d))

    # --- KMeans Standard ---
    kmeans_std = KMeans(n_clusters=k, random_state=42)
    kmeans_labels = kmeans_std.fit_predict(data_scaled)
    sse_kmeans = compute_sse(data_scaled, kmeans_std.cluster_centers_, kmeans_labels)
    sil_kmeans = silhouette_score(data_scaled, kmeans_labels)

    # --- KMeans + GWO ---
    kmeans_gwo = KMeans(n_clusters=k, init=best_centroids, n_init=1, random_state=42)
    gwo_labels = kmeans_gwo.fit_predict(data_scaled)
    sse_gwo = compute_sse(data_scaled, kmeans_gwo.cluster_centers_, gwo_labels)
    sil_gwo = silhouette_score(data_scaled, gwo_labels)

    # --- Perbandingan Hasil ---
    st.subheader("📊 Perbandingan Evaluasi Klaster")
    st.dataframe(pd.DataFrame({
        'Metode': ['KMeans', 'GWO-KMeans'],
        'SSE': [round(sse_kmeans, 4), round(sse_gwo, 4)],
        'Silhouette': [round(sil_kmeans, 4), round(sil_gwo, 4)]
    }))

    # --- Visualisasi Klaster ---
    st.subheader("📈 Visualisasi Clustering")
    fig = plot_clusters_side_by_side(
        data_scaled,
        kmeans_labels, kmeans_std.cluster_centers_,
        gwo_labels, kmeans_gwo.cluster_centers_
    )
    st.pyplot(fig)

    # --- Hasil Final ---
    kmeans = KMeans(n_clusters=k, init=best_centroids, n_init=1, random_state=42)
    kmeans_labels = kmeans.fit_predict(data_scaled)
    sse_final = compute_sse(data_scaled, kmeans.cluster_centers_, kmeans_labels)
    sil_final = silhouette_score(data_scaled, kmeans_labels)

    st.success(f"✅ SSE Final: {round(sse_final, 4)} | Silhouette Final: {round(sil_final, 4)}")

