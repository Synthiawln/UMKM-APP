# =========================================================
# Analisis Clustering UMKM Kuliner - K-Means + GWO + ACO
# Disinkronkan agar SEMUA bagian analisis SAMA PERSIS dengan
# notebook Colab skripsi (encoding, filtering, parameter,
# perbandingan 3 metode, PCA, uji Friedman, profil cluster).
# =========================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from scipy.stats import friedmanchisquare
from sklearn.decomposition import PCA

# =========================================================
# KONFIGURASI HALAMAN (hanya dipanggil SEKALI, di paling atas)
# =========================================================
st.set_page_config(
    page_title="Analisis Clustering UMKM Kuliner",
    page_icon="🍽️",
    layout="wide"
)

DEFAULT_DATA_PATH = "Data Set UMKM.xlsx"
RANDOM_SEED = 42          # seed tetap agar hasil GWO/ACO/KMeans reproducible
K_FINAL = 4                # k hasil metode Elbow (sesuai skripsi)

# =========================================================
# MAPPING ORDINAL DETERMINISTIK (SAMA PERSIS DENGAN COLAB)
# =========================================================
MAP_JENIS_KELAMIN = {'L': 0, 'P': 1}

MAP_PENDIDIKAN = {
    'BELUM TAMAT SD/SEDERAJAT'           : 1,
    'SD'                                  : 2,
    'TAMAT SD/SEDERAJAT'                  : 2,
    'SMP'                                 : 3,
    'SMA'                                 : 4,
    'SMK'                                 : 4,
    'SLTA/SEDERAJAT'                      : 4,
    'D1'                                  : 5,
    'D2'                                  : 5,
    'D3'                                  : 5,
    'AKADEMI/DIPLOMA III/SARJANA MUDA'    : 5,
    'D4'                                  : 6,
    'DIPLOMA IV/STRATA I'                 : 6,
    'S1'                                  : 6,
    'S2'                                  : 7
}

MAP_OMSET = {
    'Kurang dari 10 juta'    : 1,
    '10 juta s/d 25 juta'    : 2,
    '25 juta s/d 40 juta'    : 3,
    '40 juta s/d 55 juta'    : 4,
    '55 juta s/d 70 juta'    : 5,
    '70 juta s/d 85 juta'    : 6,
    '85 juta s/d 100 juta'   : 7,
    '100 juta s/d 120 juta'  : 8,
    '120 juta s/d 150 juta'  : 9,
    'Lebih dari 150 juta'    : 10
}

MAP_KEPEMILIKAN = {
    'Sewa'             : 1,
    'Magersari (adat)' : 2,
    'Lainnya'          : 3,
    'Milik sendiri'    : 4
}

ENCODED_COLS = ['Jenis_Kelamin', 'Pendidikan', 'Omset', 'Kepemilikan', 'Sarana_Media']


def count_sarana(val):
    if pd.isna(val) or str(val).strip() in ['-', '']:
        return 0
    return len([x.strip() for x in str(val).split(',') if x.strip()])


# =========================================================
# LOAD & PREPROCESS (di-cache)
# =========================================================
@st.cache_data(show_spinner=False)
def load_raw_data(uploaded_bytes=None):
    if uploaded_bytes is not None:
        return pd.read_excel(uploaded_bytes, sheet_name="UKM Kuliner")
    if not os.path.exists(DEFAULT_DATA_PATH):
        raise FileNotFoundError(
            f"File '{DEFAULT_DATA_PATH}' tidak ditemukan. "
            "Pastikan file excel sudah ada satu folder dengan app.py di repo GitHub."
        )
    return pd.read_excel(DEFAULT_DATA_PATH, sheet_name="UKM Kuliner")


@st.cache_data(show_spinner=False)
def preprocess_data(df: pd.DataFrame):
    data = df.dropna().drop_duplicates()

    enc = data.copy()
    enc = enc[enc['Jenis Kelamin'].isin(['L', 'P'])]
    enc = enc[enc['Pendidikan Terakhir'].isin(MAP_PENDIDIKAN.keys())]
    enc = enc.dropna(subset=[
        'Omset per-Tahun', 'Status Kepemilkan Tanah/Bangunan', 'Sarana Media Elektronik'
    ])

    enc['Jenis_Kelamin'] = enc['Jenis Kelamin'].map(MAP_JENIS_KELAMIN)
    enc['Pendidikan']    = enc['Pendidikan Terakhir'].map(MAP_PENDIDIKAN)
    enc['Omset']         = enc['Omset per-Tahun'].map(MAP_OMSET)
    enc['Kepemilikan']   = enc['Status Kepemilkan Tanah/Bangunan'].map(MAP_KEPEMILIKAN)
    enc['Sarana_Media']  = enc['Sarana Media Elektronik'].apply(count_sarana)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(enc[ENCODED_COLS])
    data_scaled_df = pd.DataFrame(data_scaled, columns=ENCODED_COLS)

    return data, enc, data_scaled, data_scaled_df


# =========================================================
# ALGORITMA INTI (identik dengan Colab)
# =========================================================
def compute_sse(data, centroids, labels):
    return np.sum((data - centroids[labels]) ** 2)


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


def ACO(objective_func, lb, ub, dim, k, d, data,
        n_ants=50, epochs=200, evaporation=0.5, alpha=1.0, q=1.0):
    ants = np.random.uniform(lb, ub, (n_ants, dim))
    pheromones = np.ones(n_ants)

    best_pos = ants[0].copy()
    best_score = objective_func(best_pos, k, d, data)

    for t in range(epochs):
        scores = np.array([objective_func(ant, k, d, data) for ant in ants])

        idx = np.argmin(scores)
        if scores[idx] < best_score:
            best_score = scores[idx]
            best_pos = ants[idx].copy()

        pheromones = (1 - evaporation) * pheromones
        for i in range(n_ants):
            pheromones[i] += q / (1 + scores[i])

        prob = pheromones ** alpha
        prob /= prob.sum()

        new_ants = np.zeros_like(ants)
        for i in range(n_ants):
            guide_idx = np.random.choice(n_ants, p=prob)
            noise = np.random.uniform(-0.1, 0.1, dim) * (ub - lb)
            new_ants[i] = np.clip(ants[guide_idx] + noise, lb, ub)

        ants = new_ants

    return best_pos, best_score


@st.cache_data(show_spinner=False)
def run_all_clustering(data_scaled: np.ndarray, k: int, seed: int):
    """KMeans standar + GWO-KMeans + ACO-KMeans, seed tetap -> hasil konsisten."""
    d = data_scaled.shape[1]
    dim = k * d
    lb = np.tile(np.min(data_scaled, axis=0), k)
    ub = np.tile(np.max(data_scaled, axis=0), k)

    # --- KMeans standar ---
    kmeans_std = KMeans(n_clusters=k, random_state=seed, n_init=10)
    kmeans_labels = kmeans_std.fit_predict(data_scaled)
    sse_kmeans = compute_sse(data_scaled, kmeans_std.cluster_centers_, kmeans_labels)
    sil_kmeans = silhouette_score(data_scaled, kmeans_labels) if len(set(kmeans_labels)) > 1 else np.nan

    # --- GWO ---
    np.random.seed(seed)
    best_pos_gwo, _ = GWO(objective_function, lb, ub, dim, k, d, data_scaled, pop_size=50, epochs=200)
    best_centroids_gwo = best_pos_gwo.reshape((k, d))

    dist_gwo_raw = cdist(data_scaled, best_centroids_gwo)
    labels_gwo_raw = np.argmin(dist_gwo_raw, axis=1)
    sse_gwo_raw = compute_sse(data_scaled, best_centroids_gwo, labels_gwo_raw)

    kmeans_gwo = KMeans(n_clusters=k, init=best_centroids_gwo, n_init=1, random_state=seed)
    gwo_labels = kmeans_gwo.fit_predict(data_scaled)
    sse_gwo = compute_sse(data_scaled, kmeans_gwo.cluster_centers_, gwo_labels)
    sil_gwo = silhouette_score(data_scaled, gwo_labels) if len(set(gwo_labels)) > 1 else np.nan

    # --- ACO ---
    np.random.seed(seed)
    best_pos_aco, _ = ACO(objective_function, lb, ub, dim, k, d, data_scaled, n_ants=50, epochs=200)
    best_centroids_aco = best_pos_aco.reshape((k, d))

    dist_aco_raw = cdist(data_scaled, best_centroids_aco)
    labels_aco_raw = np.argmin(dist_aco_raw, axis=1)
    sse_aco_raw = compute_sse(data_scaled, best_centroids_aco, labels_aco_raw)

    kmeans_aco = KMeans(n_clusters=k, init=best_centroids_aco, n_init=1, random_state=seed)
    aco_labels = kmeans_aco.fit_predict(data_scaled)
    sse_aco = compute_sse(data_scaled, kmeans_aco.cluster_centers_, aco_labels)
    sil_aco = silhouette_score(data_scaled, aco_labels) if len(set(aco_labels)) > 1 else np.nan

    return {
        "kmeans_labels": kmeans_labels, "kmeans_centroids": kmeans_std.cluster_centers_,
        "sse_kmeans": sse_kmeans, "sil_kmeans": sil_kmeans,

        "gwo_labels": gwo_labels, "gwo_centroids": kmeans_gwo.cluster_centers_,
        "sse_gwo": sse_gwo, "sil_gwo": sil_gwo, "sse_gwo_raw": sse_gwo_raw,

        "aco_labels": aco_labels, "aco_centroids": kmeans_aco.cluster_centers_,
        "sse_aco": sse_aco, "sil_aco": sil_aco, "sse_aco_raw": sse_aco_raw,
    }


@st.cache_data(show_spinner=False)
def run_friedman_30(data_scaled: np.ndarray, k: int, n_runs: int = 30):
    """Uji statistik 30-run untuk membandingkan KMeans vs GWO vs ACO (Friedman Test)."""
    d = data_scaled.shape[1]
    dim = k * d
    lb = np.tile(np.min(data_scaled, axis=0), k)
    ub = np.tile(np.max(data_scaled, axis=0), k)

    records = []
    for run in range(n_runs):
        seed = run * 7

        km = KMeans(n_clusters=k, init='random', n_init=1, random_state=seed)
        km_labels = km.fit_predict(data_scaled)
        sse_km = compute_sse(data_scaled, km.cluster_centers_, km_labels)
        sil_km = silhouette_score(data_scaled, km_labels)

        np.random.seed(seed)
        best_pos_gwo, _ = GWO(objective_function, lb, ub, dim, k, d, data_scaled, pop_size=50, epochs=200)
        c_gwo = best_pos_gwo.reshape((k, d))
        lbl_gwo = np.argmin(cdist(data_scaled, c_gwo), axis=1)
        sse_gwo_r = compute_sse(data_scaled, c_gwo, lbl_gwo)
        sil_gwo_r = silhouette_score(data_scaled, lbl_gwo)

        np.random.seed(seed)
        best_pos_aco, _ = ACO(objective_function, lb, ub, dim, k, d, data_scaled, n_ants=50, epochs=200)
        c_aco = best_pos_aco.reshape((k, d))
        lbl_aco = np.argmin(cdist(data_scaled, c_aco), axis=1)
        sse_aco_r = compute_sse(data_scaled, c_aco, lbl_aco)
        sil_aco_r = silhouette_score(data_scaled, lbl_aco)

        records.append({
            'No': run + 1,
            'SSE_KMeans': round(sse_km, 2), 'Sil_KMeans': round(sil_km, 4),
            'SSE_GWO': round(sse_gwo_r, 2), 'Sil_GWO': round(sil_gwo_r, 4),
            'SSE_ACO': round(sse_aco_r, 2), 'Sil_ACO': round(sil_aco_r, 4),
        })

    df30 = pd.DataFrame(records)
    stat, p = friedmanchisquare(df30['SSE_KMeans'], df30['SSE_GWO'], df30['SSE_ACO'])
    return df30, stat, p


# =========================================================
# FUNGSI VISUALISASI
# =========================================================
def plot_clusters_side_by_side(data, labels1, centroids1, labels2, centroids2, title2="GWO-KMeans"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels1, palette='Set2', s=50, ax=axes[0])
    axes[0].scatter(centroids1[:, 0], centroids1[:, 1], c='red', s=100, marker='X', label='Centroid')
    axes[0].set_title("KMeans Clustering (Standard)")
    axes[0].legend()

    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels2, palette='Set2', s=50, ax=axes[1])
    axes[1].scatter(centroids2[:, 0], centroids2[:, 1], c='red', s=100, marker='X', label='Centroid')
    axes[1].set_title(f"{title2} Clustering")
    axes[1].legend()
    plt.tight_layout()
    return fig


def plot_comparison_bars(sse_values, sil_values, metode):
    colors = ['#aec6cf', '#b5e4b5', '#f4a9a8']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle('Perbandingan Metode Clustering (KMeans, GWO, ACO)', fontsize=13, fontweight='bold')

    bars1 = axes[0].bar(metode, sse_values, color=colors, edgecolor='gray', linewidth=0.8)
    min_idx = int(np.argmin(sse_values))
    bars1[min_idx].set_edgecolor('red')
    bars1[min_idx].set_linewidth(2.5)
    for bar, val in zip(bars1, sse_values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(sse_values) * 0.01,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[0].set_title('SSE (lebih kecil = lebih baik)', fontsize=10)
    axes[0].set_ylabel('SSE')
    axes[0].spines[['top', 'right']].set_visible(False)
    axes[0].yaxis.grid(True, linestyle='--', alpha=0.4)
    axes[0].set_axisbelow(True)

    bars2 = axes[1].bar(metode, sil_values, color=colors, edgecolor='gray', linewidth=0.8)
    max_idx = int(np.argmax(sil_values))
    bars2[max_idx].set_edgecolor('green')
    bars2[max_idx].set_linewidth(2.5)
    for bar, val in zip(bars2, sil_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(sil_values) * 0.01,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[1].set_title('Silhouette Score (lebih besar = lebih baik)', fontsize=10)
    axes[1].set_ylabel('Silhouette Score')
    axes[1].spines[['top', 'right']].set_visible(False)
    axes[1].yaxis.grid(True, linestyle='--', alpha=0.4)
    axes[1].set_axisbelow(True)

    plt.tight_layout()
    return fig


def plot_pca_all_methods(data, all_labels, all_centroids, titles, k):
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data)
    n = len(titles)
    palette = sns.color_palette("pastel", k)

    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    fig.suptitle('PCA 2D – Perbandingan Semua Metode Clustering', fontsize=12, fontweight='bold')

    for idx, (labels, centroids, title) in enumerate(zip(all_labels, all_centroids, titles)):
        pca_c = pca.transform(centroids)
        sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=labels, palette=palette,
                        ax=axes[idx], s=55, edgecolor='gray', linewidth=0.4, legend=False)
        axes[idx].scatter(pca_c[:, 0], pca_c[:, 1], c='darkred', marker='X', s=160,
                          edgecolor='black', zorder=5)
        axes[idx].set_title(title, fontsize=10)
        axes[idx].grid(True, linestyle='--', alpha=0.3)
        axes[idx].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    return fig


def plot_cluster_distribution(cluster_series):
    cluster_count = cluster_series.value_counts().sort_index()
    labels = [f"Cluster {i}" for i in cluster_count.index]
    sizes = cluster_count.values.tolist()
    total = sum(sizes)
    persentase = [s / total * 100 for s in sizes]

    base_colors = ["#aec6cf", "#b5e4b5", "#f5d895", "#f4a9a8",
                   "#c9b1d9", "#f7b89e", "#a8d8b9", "#f9e4b7"]
    colors = base_colors[:len(labels)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("white")

    ax_pie = axes[0]
    wedges, texts, autotexts = ax_pie.pie(
        sizes, labels=None, colors=colors, autopct="%1.1f%%", startangle=140,
        wedgeprops=dict(linewidth=1.5, edgecolor="white"), pctdistance=0.70,
    )
    for at in autotexts:
        at.set_fontsize(10); at.set_fontweight("bold"); at.set_color("black")

    legend_labels = [f"{lbl}\n(n={sz})" for lbl, sz in zip(labels, sizes)]
    patches = [mpatches.Patch(color=colors[i], label=legend_labels[i]) for i in range(len(labels))]
    ax_pie.legend(handles=patches, loc="upper left", bbox_to_anchor=(-0.25, 1.05), fontsize=8, frameon=False)
    ax_pie.set_title("Distribusi Jumlah Data per Cluster", fontsize=12, fontweight="bold", pad=12)

    ax_bar = axes[1]
    x = np.arange(len(labels))
    bars = ax_bar.bar(x, sizes, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    for bar, sz in zip(bars, sizes):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.005,
                    str(sz), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(labels, fontsize=9)
    ax_bar.set_xlabel("Cluster", fontsize=10); ax_bar.set_ylabel("Jumlah Data", fontsize=10)
    ax_bar.set_title("Jumlah Anggota Tiap Cluster", fontsize=12, fontweight="bold", pad=12)
    ax_bar.set_ylim(0, max(sizes) * 1.15)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax_bar.set_axisbelow(True)

    plt.tight_layout(pad=2.5)
    return fig, labels, sizes, persentase, total


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("Tentang Penelitian")
st.sidebar.markdown("""
### Latar Belakang
UMKM kuliner memiliki peran penting dalam perekonomian, namun sering menghadapi
tantangan dalam menentukan strategi pemasaran.

Penelitian ini mengoptimalkan klastering dengan **K-Means**, **Grey Wolf Optimizer (GWO)**,
dan **Ant Colony Optimization (ACO)** sebagai pembanding, agar segmen UMKM lebih jelas.
""")

st.sidebar.subheader("Dataset")
st.sidebar.caption("Aplikasi otomatis membaca dataset bawaan dari repo. Upload opsional untuk mencoba dataset lain.")
uploaded_file = st.sidebar.file_uploader("Ganti dataset (opsional)", type=["xlsx"])

try:
    kuliner_data_raw = load_raw_data(uploaded_file)
    if uploaded_file is not None:
        st.sidebar.success("✅ Menggunakan dataset yang kamu upload.")
    else:
        st.sidebar.info(f"📂 Menggunakan dataset bawaan: `{DEFAULT_DATA_PATH}`")
except FileNotFoundError as e:
    st.sidebar.error(str(e))
    st.error(str(e))
    st.stop()

menu = st.sidebar.radio("Menu Utama", ["🏠 Beranda", "📊 Dataset", "🔎 Hasil Analisis"])

# =========================================================
# MENU: BERANDA
# =========================================================
if menu == "🏠 Beranda":
    col_space1, col_center, col_space2 = st.columns([1, 6, 1])
    with col_center:
        header_col1, header_col2 = st.columns([1, 10])
        with header_col1:
            if os.path.exists("animal-track.png"):
                st.image("animal-track.png", width=60)
        with header_col2:
            st.markdown("""
                <h1 style='color: #F8F9FA; margin-bottom: 0;'>Analisis Clustering UMKM Kuliner</h1>
                <h4 style='color: #A0AEC0; margin-top: 0;'>K-Means dioptimasi dengan Grey Wolf Optimizer (GWO), dibandingkan dengan Ant Colony Optimization (ACO)</h4>
            """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns([2, 2])
    with col1:
        st.subheader("Tentang Aplikasi")
        st.write("""
        Aplikasi ini dikembangkan untuk membantu analisis data **UMKM Kuliner**
        menggunakan kombinasi metode **K-Means** dan **Grey Wolf Optimizer (GWO)**,
        dengan **ACO** sebagai metode pembanding.
        Tujuannya agar hasil segmentasi lebih akurat dan mendukung efisiensi strategi pemasaran.
        """)
        st.info("💡 Gunakan menu di sebelah kiri untuk melihat dataset dan hasil analisis lengkap.")
    with col2:
        st.markdown("### ⚙️ Fitur Utama Aplikasi")
        st.markdown("""
        - 📂 Dataset UMKM otomatis terbaca dari repo
        - 🔍 Optimasi centroid awal dengan GWO & ACO
        - 🧩 Perbandingan 3 metode clustering
        - 📊 Visualisasi PCA & profil tiap cluster
        - 📈 Evaluasi SSE, Silhouette, dan Uji Friedman
        """)

    st.markdown("---")
    st.markdown("<p style='text-align:center; color:gray;'>© 2025 Aplikasi Analisis UMKM Kuliner</p>",
                unsafe_allow_html=True)

# =========================================================
# MENU: DATASET
# =========================================================
elif menu == "📊 Dataset":
    st.subheader("📊 Preview Dataset")
    st.dataframe(kuliner_data_raw.head(20))
    st.subheader("ℹ️ Informasi Dataset")
    st.write("Jumlah baris dan kolom:", kuliner_data_raw.shape)
    st.write("Nama kolom:", kuliner_data_raw.columns.tolist())
    st.markdown("---")
    st.markdown("<p style='text-align:center; color:gray;'>© 2025 Aplikasi Analisis UMKM Kuliner</p>",
                unsafe_allow_html=True)

# =========================================================
# MENU: HASIL ANALISIS
# =========================================================
elif menu == "🔎 Hasil Analisis":
    st.title("📌 Hasil Analisis Clustering")

    data_clean, kuliner_enc, data_scaled, data_scaled_df = preprocess_data(kuliner_data_raw)

    st.write("Jumlah data setelah cleaning & filtering:", kuliner_enc.shape)
    st.caption(
        "Filtering mengikuti metodologi skripsi: hanya baris dengan Jenis Kelamin (L/P), "
        "Pendidikan Terakhir valid, dan tanpa nilai kosong pada Omset/Kepemilikan/Sarana Media."
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1️⃣ Data & Elbow", "2️⃣ Perbandingan Metode", "3️⃣ PCA & Visualisasi",
        "4️⃣ Uji Statistik (Friedman)", "5️⃣ Profil Cluster"
    ])

    # ---------- TAB 1: DATA & ELBOW ----------
    with tab1:
        st.subheader("Hasil Encoding Ordinal (Deterministik)")
        preview_cols = [
            'Jenis Kelamin', 'Jenis_Kelamin', 'Pendidikan Terakhir', 'Pendidikan',
            'Omset per-Tahun', 'Omset', 'Status Kepemilkan Tanah/Bangunan', 'Kepemilikan',
            'Sarana Media Elektronik', 'Sarana_Media'
        ]
        st.dataframe(kuliner_enc[preview_cols].head(20))

        st.subheader("Hasil Normalisasi (MinMaxScaler)")
        st.dataframe(data_scaled_df.head(20))

        st.markdown(
            "<h3 style='text-align: center;'>Elbow Method (SSE untuk k = 2..10)</h3>",
            unsafe_allow_html=True
        )
        sse_values_elbow = []
        k_range = range(2, 11)
        for k_test in k_range:
            kmeans_tmp = KMeans(n_clusters=k_test, random_state=RANDOM_SEED, n_init=10)
            kmeans_tmp.fit(data_scaled)
            sse_values_elbow.append(kmeans_tmp.inertia_)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(list(k_range), sse_values_elbow, marker='o', color='blue', linewidth=2)
            ax.set_title("Metode Elbow untuk Menentukan k Optimal", fontsize=12)
            ax.set_xlabel("Jumlah Cluster (k)")
            ax.set_ylabel("SSE (Sum of Squared Error)")
            ax.grid(True, linestyle="--", alpha=0.6)
            st.pyplot(fig)

        st.success(f"✅ Berdasarkan metode Elbow, dipilih **k = {K_FINAL}** (sesuai skripsi).")

        with st.expander("🔧 Eksperimen: coba nilai k lain (opsional, di luar hasil resmi skripsi)"):
            k_experiment = st.slider("Nilai k eksperimen", min_value=2, max_value=10, value=K_FINAL, step=1)
            if k_experiment != K_FINAL:
                st.warning("⚠️ Hasil di tab lain tetap memakai k=4 (resmi skripsi). Ini hanya area eksperimen.")

    # ---------- Jalankan clustering utama (k tetap = 4, resmi skripsi) ----------
    with st.spinner("Menjalankan KMeans, GWO, dan ACO..."):
        result = run_all_clustering(data_scaled, K_FINAL, RANDOM_SEED)

    kmeans_labels = result["kmeans_labels"]
    gwo_labels = result["gwo_labels"]
    aco_labels = result["aco_labels"]

    metode = ['KMeans\n(Standard)', 'GWO-KMeans', 'ACO-KMeans']
    sse_all = [result["sse_kmeans"], result["sse_gwo"], result["sse_aco"]]
    sil_all = [result["sil_kmeans"], result["sil_gwo"], result["sil_aco"]]

    # ---------- TAB 2: PERBANDINGAN METODE ----------
    with tab2:
        st.subheader("Perbandingan Evaluasi Klaster (KMeans vs GWO-KMeans vs ACO-KMeans)")

        eval_df = pd.DataFrame({
            'Metode': ['KMeans (Standard)', 'GWO murni (pre-KMeans)', 'GWO-KMeans (post)',
                       'ACO murni (pre-KMeans)', 'ACO-KMeans (post)'],
            'SSE': [round(result["sse_kmeans"], 4), round(result["sse_gwo_raw"], 4),
                    round(result["sse_gwo"], 4), round(result["sse_aco_raw"], 4), round(result["sse_aco"], 4)]
        })
        st.dataframe(eval_df, use_container_width=True)

        eval_summary = pd.DataFrame({
            'Metode': metode,
            'SSE': [round(v, 4) for v in sse_all],
            'Silhouette Score': [round(v, 4) if not np.isnan(v) else "NA" for v in sil_all]
        })
        st.dataframe(eval_summary, use_container_width=True)

        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            fig_bar = plot_comparison_bars(sse_all, sil_all, metode)
            st.pyplot(fig_bar)

        best_sse_method = metode[int(np.argmin(sse_all))].replace('\n', ' ')
        best_sil_method = metode[int(np.argmax(sil_all))].replace('\n', ' ')
        st.info(f"📌 Border **merah** = SSE terkecil → **{best_sse_method}**")
        st.info(f"📌 Border **hijau** = Silhouette terbesar → **{best_sil_method}**")

    # ---------- TAB 3: PCA & VISUALISASI ----------
    with tab3:
        st.markdown("<h4 style='text-align: center;'>Visualisasi Klaster: KMeans vs GWO-KMeans</h4>",
                    unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            fig_side = plot_clusters_side_by_side(
                data_scaled, kmeans_labels, result["kmeans_centroids"], gwo_labels, result["gwo_centroids"]
            )
            fig_side.set_size_inches(10, 5)
            st.pyplot(fig_side)

        st.markdown("<h4 style='text-align: center;'>PCA 2D – Perbandingan Semua Metode</h4>",
                    unsafe_allow_html=True)
        fig_pca = plot_pca_all_methods(
            data_scaled,
            all_labels=[kmeans_labels, gwo_labels, aco_labels],
            all_centroids=[result["kmeans_centroids"], result["gwo_centroids"], result["aco_centroids"]],
            titles=['KMeans (Standard)', 'GWO-KMeans', 'ACO-KMeans'],
            k=K_FINAL
        )
        st.pyplot(fig_pca)

        # st.markdown("<h4 style='text-align: center;'>Pendidikan vs Omset — GWO-KMeans</h4>",
        #             unsafe_allow_html=True)
        # col1, col2, col3 = st.columns([1, 2, 1])
        # with col2:
        #     fig2, ax2 = plt.subplots(figsize=(5, 3))
        #     sns.scatterplot(x=data_scaled_df['Pendidikan'], y=data_scaled_df['Omset'],
        #                      hue=gwo_labels, palette='Set2', s=60, legend='brief', ax=ax2)
        #     centroids = result["gwo_centroids"]
        #     ax2.scatter(centroids[:, 1], centroids[:, 2], c='red', s=150, marker='X', label='Centroid')
        #     ax2.set_title("Pendidikan vs Omset (GWO-KMeans)", fontsize=11)
        #     ax2.legend()
        #     st.pyplot(fig2)

    # ---------- TAB 4: UJI STATISTIK FRIEDMAN ----------
    with tab4:
        st.subheader("Uji Statistik 30-Run (Friedman Test)")
        st.warning(
            "⏳ Proses ini menjalankan GWO & ACO sebanyak **30 kali** (masing-masing 200 iterasi) — "
            "bisa memakan waktu beberapa menit. Hasil akan di-cache setelah selesai sekali dijalankan."
        )
        run_test = st.button("▶️ Jalankan Uji 30-Run")

        if run_test:
            with st.spinner("Menjalankan 30 run... harap tunggu, ini proses berat"):
                df30, stat, p = run_friedman_30(data_scaled, K_FINAL, n_runs=30)
            st.session_state["df30"] = df30
            st.session_state["friedman_stat"] = stat
            st.session_state["friedman_p"] = p

        if "df30" in st.session_state:
            df30 = st.session_state["df30"]
            stat = st.session_state["friedman_stat"]
            p = st.session_state["friedman_p"]

            st.markdown("#### Tabel Perbandingan Evaluasi Cluster (30 Run)")
            st.dataframe(df30, use_container_width=True)

            st.markdown("#### 📊 Rata-rata 30 Run")
            st.write(f"- **K-Means** → SSE: {df30['SSE_KMeans'].mean():.2f} | Silhouette: {df30['Sil_KMeans'].mean():.4f}")
            st.write(f"- **GWO-KMeans** → SSE: {df30['SSE_GWO'].mean():.2f} | Silhouette: {df30['Sil_GWO'].mean():.4f}")
            st.write(f"- **ACO-KMeans** → SSE: {df30['SSE_ACO'].mean():.2f} | Silhouette: {df30['Sil_ACO'].mean():.4f}")

            st.markdown("#### Hasil Uji Friedman")
            st.write(f"- Friedman Statistic: **{stat:.2f}**")
            st.write(f"- P-value: **{p:.2f}**")
            if p < 0.05:
                st.success("✅ Terdapat perbedaan signifikan antar metode (p < 0.05) → GWO/ACO terbukti lebih baik.")
            else:
                st.warning("⚠️ Tidak ada perbedaan signifikan antar metode (p ≥ 0.05).")
        else:
            st.caption("Klik tombol di atas untuk menjalankan uji statistik.")

    # ---------- TAB 5: PROFIL CLUSTER ----------
    with tab5:
        st.subheader("Profil Cluster")
        label_source = st.radio(
            "Gunakan label cluster dari:",
            ["GWO-KMeans (sesuai tujuan penelitian)", "KMeans Standard"],
            horizontal=True
        )
        chosen_labels = gwo_labels if label_source.startswith("GWO") else kmeans_labels
        st.caption(
            "Catatan: di notebook Colab, bagian profil cluster memakai variabel `kmeans_labels` "
            "meski komentarnya menyebut 'GWO-KMeans' — ini tampaknya salah penamaan variabel. "
            "Default di sini memakai label GWO-KMeans sesuai tujuan penelitian; kamu bisa ganti "
            "ke KMeans Standard di atas kalau ingin menyamai persis output asli notebook."
        )

        profile_df = data_scaled_df.copy()
        profile_df['cluster'] = chosen_labels

        st.markdown("#### Rata-rata Tiap Cluster")
        cluster_mean = profile_df.groupby('cluster').mean()
        st.dataframe(cluster_mean.round(3), use_container_width=True)

        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            fig_mean, ax_mean = plt.subplots(figsize=(9, 5))
            cluster_mean.T.plot(kind='bar', ax=ax_mean)
            ax_mean.set_title('Perbandingan Rata-rata Tiap Cluster')
            ax_mean.set_xlabel('Variabel')
            ax_mean.set_ylabel('Nilai Rata-rata')
            ax_mean.legend(title='Cluster')
            ax_mean.grid()
            plt.xticks(rotation=45)
            st.pyplot(fig_mean)

        st.markdown("#### Distribusi Jumlah Data per Cluster")
        fig_dist, labels_d, sizes_d, persentase_d, total_d = plot_cluster_distribution(profile_df['cluster'])
        st.pyplot(fig_dist)

        summary_dist = pd.DataFrame({
            "Cluster": labels_d, "Jumlah": sizes_d,
            "Persentase": [f"{p:.1f}%" for p in persentase_d]
        })
        st.dataframe(summary_dist, use_container_width=True)

        st.markdown("#### Contoh Data Asli Tiap Cluster")
        kuliner_hasil = kuliner_enc.copy()
        kuliner_hasil['cluster'] = chosen_labels
        kolom_asli = [
            'Jenis Kelamin', 'Pendidikan Terakhir', 'Omset per-Tahun',
            'Status Kepemilkan Tanah/Bangunan', 'Sarana Media Elektronik'
        ]
        for c in sorted(kuliner_hasil['cluster'].unique()):
            anggota = (kuliner_hasil['cluster'] == c).sum()
            st.markdown(f"**▶ Cluster {c}** (total: {anggota} anggota)")
            st.dataframe(
                kuliner_hasil[kuliner_hasil['cluster'] == c][kolom_asli].head(5).reset_index(drop=True),
                use_container_width=True
            )

    st.markdown("---")
    st.markdown("<p style='text-align:center; color:gray;'>© 2025 Aplikasi Analisis UMKM Kuliner</p>",
                unsafe_allow_html=True)
