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
from sklearn.decomposition import PCA

# FUNGSI BANTUAN
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

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Clustering UMKM Kuliner",
    page_icon="🍽️",
    layout="wide"
)

# SIDEBAR
st.sidebar.title("Tentang Penelitian")
st.sidebar.markdown("""
### Latar Belakang
UMKM kuliner memiliki peran penting dalam perekonomian, namun sering menghadapi tantangan dalam menentukan strategi pemasaran.  

Penelitian ini mengoptimalkan klastering dengan **K-Means** dan **Grey Wolf Optimizer (GWO)** agar segmen UMKM lebih jelas, sehingga strategi pemasaran lebih efisien.
""")

# Upload dataset di sidebar
st.sidebar.subheader("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload dataset Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    kuliner_data = pd.read_excel(uploaded_file, sheet_name="UKM Kuliner")
    st.sidebar.success("✅ Dataset berhasil diupload!")
else:
    kuliner_data = pd.read_excel("C:/UMKM_CODING/Data Set UMKM.xlsx", sheet_name="UKM Kuliner")
    st.sidebar.info("📂 Menggunakan dataset bawaan dari folder.")

    
# Navigasi Sederhana
menu = st.sidebar.radio("Menu Utama", ["🏠 Beranda", "📊 Dataset", "🔎 Hasil Analisis"])

# MAIN PAGE
# Judul aplikasi
st.set_page_config(page_title="Analisis Clustering UMKM Kuliner", layout="wide")

# MENU: BERANDA
if menu == "🏠 Beranda":

    # --- Header (logo + judul sejajar dan di tengah) ---
    col_space1, col_center, col_space2 = st.columns([1, 6, 1])
    with col_center:
        header_col1, header_col2 = st.columns([1, 10])

        with header_col1:
            st.image("animal-track.png", width=60)

        with header_col2:
            st.markdown("""
                <h1 style='color: #F8F9FA; margin-bottom: 0;'>Analisis Clustering UMKM Kuliner</h1>
                <h4 style='color: #A0AEC0; margin-top: 0;'>Mengoptimalkan K-Means dengan Grey Wolf Optimizer (GWO)</h4>
            """, unsafe_allow_html=True)

    st.markdown("---")
    
    col1, col2 = st.columns([2, 2])

    with col1:
        st.subheader("Tentang Aplikasi")
        st.write("""
        Aplikasi ini dikembangkan untuk membantu analisis data **UMKM Kuliner** 
        menggunakan kombinasi metode **K-Means** dan **Grey Wolf Optimizer (GWO)**.  
        Tujuannya agar hasil segmentasi lebih akurat dan mendukung efisiensi strategi pemasaran.
        """)
        st.info("💡 Gunakan menu di sebelah kiri untuk mengunggah dataset dan menjalankan analisis.")

    with col2:
        st.markdown("### ⚙️ Fitur Utama Aplikasi")
        st.markdown("""
        - 📂 Upload dataset UMKM  
        - 🔍 Optimasi centroid awal dengan GWO  
        - 🧩 Proses klasterisasi otomatis  
        - 📊 Visualisasi hasil clustering  
        - 📈 Evaluasi kinerja (SSE dan Silhouette)
        """)
        st.markdown("### 📘 Tujuan Penelitian")
        st.write("""
        Dengan analisis ini, diharapkan pelaku UMKM dapat memahami karakteristik segmen usahanya,
        sehingga strategi pemasaran yang diambil lebih terarah dan efisien.
        """)

    st.markdown("---")
    st.markdown("<p style='text-align:center; color:gray;'>© 2025 Aplikasi Analisis UMKM Kuliner</p>", unsafe_allow_html=True)

# MENU: DATASET
# Tampilkan preview dataset
elif menu == "📊 Dataset":
    st.subheader("📊Preview Dataset")
    st.dataframe(kuliner_data.head(20))

    # Info dataset
    st.subheader("ℹ️Informasi Dataset")
    st.write("Jumlah baris dan kolom:", kuliner_data.shape)
    st.write("Nama kolom:", kuliner_data.columns.tolist())

    st.markdown("---")
    st.markdown("<p style='text-align:center; color:gray;'>© 2025 Aplikasi Analisis UMKM Kuliner</p>", unsafe_allow_html=True)
    
# MENU: HASIL ANALISIS
elif menu == "🔎 Hasil Analisis":
    st.title("📌 Hasil Analisis Clustering")

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
    st.markdown(
    "<h3 style='text-align: center; color:#F8F9FA;'>Elbow Method (SSE untuk k = 2..10)</h3>",
    unsafe_allow_html=True
    )
    sse_values = []
    k_range = range(2, 11)
    for k_test in k_range:
        kmeans_tmp = KMeans(n_clusters=k_test, random_state=42)
        kmeans_tmp.fit(data_scaled)
        sse_values.append(kmeans_tmp.inertia_)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        fig, ax = plt.subplots(figsize=(5,3))
        ax.plot(list(k_range), sse_values, marker='o', color='blue', linewidth=2)
        ax.set_title("Metode Elbow untuk Menentukan k Optimal", fontsize=12)
        ax.set_xlabel("Jumlah Cluster (k)")
        ax.set_ylabel("SSE (Sum of Squared Error)")
        ax.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig)

    # --- Pilih k ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        k = st.slider("Pilih jumlah cluster (k)", min_value=2, max_value=10, value=4, step=1)

    # --- Jalankan GWO dan KMeans ---
    d = data_scaled.shape[1]
    dim = k * d
    lb = np.tile(np.min(data_scaled, axis=0), k)
    ub = np.tile(np.max(data_scaled, axis=0), k)

    with st.spinner("Menjalankan GWO untuk cari centroid inisialisasi..."):
        best_position, best_sse = GWO(objective_function, lb, ub, dim, k, d, data_scaled, pop_size=40, epochs=150)
        best_centroids = best_position.reshape((k, d))

    # KMeans standar
    kmeans_std = KMeans(n_clusters=k, random_state=42)
    kmeans_labels = kmeans_std.fit_predict(data_scaled)
    sse_kmeans = compute_sse(data_scaled, kmeans_std.cluster_centers_, kmeans_labels)
    sil_kmeans = silhouette_score(data_scaled, kmeans_labels) if len(set(kmeans_labels)) > 1 else np.nan

    # KMeans dengan inisialisasi GWO
    kmeans_gwo = KMeans(n_clusters=k, init=best_centroids, n_init=1, random_state=42)
    gwo_labels = kmeans_gwo.fit_predict(data_scaled)
    sse_gwo = compute_sse(data_scaled, kmeans_gwo.cluster_centers_, gwo_labels)
    sil_gwo = silhouette_score(data_scaled, gwo_labels) if len(set(gwo_labels)) > 1 else np.nan

    # --- Evaluasi & hasil ---
    eval_df = pd.DataFrame({
        'Metode': ['KMeans', 'GWO-KMeans'],
        'SSE': [round(sse_kmeans, 4), round(sse_gwo, 4)],
        'Silhouette': [round(sil_kmeans, 4) if not np.isnan(sil_kmeans) else "NA",
                       round(sil_gwo, 4) if not np.isnan(sil_gwo) else "NA"]
    })
    st.subheader("Perbandingan Evaluasi Klaster")
    st.dataframe(eval_df)

    # --- Visualisasi side-by-side ---
    st.markdown(
        "<h4 style='text-align: center; color: #F8F9FA;'>Visualisasi Klaster (dua dimensi pertama kolom ter-normalisasi)</h4>",
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        fig = plot_clusters_side_by_side(data_scaled, kmeans_labels, kmeans_std.cluster_centers_,
                                         gwo_labels, kmeans_gwo.cluster_centers_)
        fig.set_size_inches(10, 5)
        st.pyplot(fig)

    # --- Scatter Pendidikan vs Omset ---
    st.markdown(
    "<h4 style='text-align: center; color: #F8F9FA;'>Visualisasi (Pendidikan vs Omset) — GWO-KMeans</h4>",
    unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        plt.figure(figsize=(5,3))
        sns.scatterplot(x=data_scaled_df['Pendidikan'], y=data_scaled_df['Omset'],
                        hue=gwo_labels, palette='Set2', s=60, legend='brief')
        centroids = kmeans_gwo.cluster_centers_
        plt.scatter(centroids[:, 1], centroids[:, 2], c='red', s=150, marker='X', label='Centroid')
        plt.title("Pendidikan vs Omset (GWO-KMeans)", fontsize=11)
        plt.xlabel("Pendidikan")
        plt.ylabel("Omset")
        plt.legend()
        st.pyplot(plt)

    # --- PCA 2D ---
    st.markdown(
    "<h4 style='text-align: center; color: #F8F9FA;'>Visualisasi PCA (2D) — GWO-KMeans</h4>",
    unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data_scaled)
        plt.figure(figsize=(5,3))
        sns.scatterplot(x=data_pca[:,0], y=data_pca[:,1],
                        hue=gwo_labels, palette='Set2', s=60, legend='brief')
        centroids_pca = pca.transform(kmeans_gwo.cluster_centers_)
        plt.scatter(centroids_pca[:,0], centroids_pca[:,1],
                    c='red', s=150, marker='X', label='Centroid')
        plt.title("PCA 2D - GWO-KMeans", fontsize=11)
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.legend()
        st.pyplot(plt)

    # --- Evaluasi Akhir ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
     st.success(f"✅ SSE (KMeans): {round(sse_kmeans,4)} | SSE (GWO-KMeans): {round(sse_gwo,4)}")

    st.markdown("---")
    st.markdown("<p style='text-align:center; color:gray;'>© 2025 Aplikasi Analisis UMKM Kuliner</p>", unsafe_allow_html=True)
