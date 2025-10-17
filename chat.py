# playlist_analysis_pipeline.py
# Exécuter dans un notebook (Jupyter/Colab) ou en tant que script.
# Le fichier CSV attendu : "/mnt/data/playlist_dataset.csv" (ou change path)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import os
import warnings
warnings.filterwarnings("ignore")

# ---- CONFIG ----
CSV_PATH = "playlist_dataset.csv"   # adapte ce chemin si besoin
OUT_CSV = "playlist_analysis_results.csv"
RANDOM_STATE = 42
MAX_TSNE_SAMPLES = 1000

# ---- 1) LOAD ----
df = pd.read_csv(CSV_PATH, sep=",")
print("Loaded:", df.shape)
print(df.head())

# ---- 2) EXPLORATION & STATS (Descriptive) ----
print("\n--- Basic info ---")
print(df.info())
print("\nMissing counts per column:")
print(df.isna().sum().sort_values(ascending=False).head(20))

numeric = df.select_dtypes(include=[np.number])
print("\nNumeric features:", numeric.columns.tolist())
print(numeric.describe().T)

categorical = df.select_dtypes(include=['object','category'])
if not categorical.empty:
    print("\nCategorical sample/value counts:")
    for c in categorical.columns:
        print(f" - {c}: {categorical[c].nunique()} unique, top: {categorical[c].mode().iat[0] if not categorical[c].mode().empty else 'NA'}")

# ---- 3) FEATURE ENGINEERING (suggestions appliquées) ----
df_fe = df.copy()
# Example common in playlists: duration in seconds if duration_ms present
if "duration_ms" in df_fe.columns:
    df_fe["duration_sec"] = df_fe["duration_ms"] / 1000.0

# Create interaction energy*valence if both exist
if set(["energy","valence"]).issubset(df_fe.columns):
    df_fe["energy_valence"] = df_fe["energy"] * df_fe["valence"]

# Normalized tempo if present
if "tempo" in df_fe.columns:
    df_fe["tempo_norm"] = (df_fe["tempo"] - df_fe["tempo"].mean())/df_fe["tempo"].std()

# Print new cols
fe_cols = [c for c in ["duration_sec","energy_valence","tempo_norm"] if c in df_fe.columns]
print("\nFeature engineered columns:", fe_cols)

# ---- 4) SELECT NUMERIC FEATURES FOR ANALYSIS ----
# Prefer typical audio features (adapt if your columns differ)
candidate_audio = ["danceability","energy","loudness","speechiness","acousticness",
                   "instrumentalness","liveness","valence","tempo","duration_sec",
                   "energy_valence","tempo_norm"]
numeric_cols = [c for c in candidate_audio if c in df_fe.columns]

# If none found, fallback to all numeric columns
if len(numeric_cols) == 0:
    numeric_cols = numeric.columns.tolist()

print("\nNumeric cols used for PCA/clustering:", numeric_cols)

X = df_fe[numeric_cols].copy()

# ---- 5) MISSING VALUE HANDLING & SCALING ----
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)

# ---- 6) PCA (analyse factorielle au sens PCA) ----
n_pca = min(6, X_scaled.shape[1])
pca = PCA(n_components=n_pca, random_state=RANDOM_STATE)
pca_components = pca.fit_transform(X_scaled)
explained_ratio = pca.explained_variance_ratio_
print("\nPCA explained variance ratio:", np.round(explained_ratio, 4))

# Save PCA components into df
for i in range(pca_components.shape[1]):
    df_fe[f"PC{i+1}"] = pca_components[:, i]

# ---- 7) Factor Analysis (autre forme d'analyse factorielle) ----
n_fa = min(6, X_scaled.shape[1])
fa = FactorAnalysis(n_components=n_fa, random_state=RANDOM_STATE)
fa_components = fa.fit_transform(X_scaled)
for i in range(fa_components.shape[1]):
    df_fe[f"FA{i+1}"] = fa_components[:, i]

# ---- 8) KMEANS clustering (minimum demandé) ----
# Try several k and choose by silhouette (k from 2 to 6 or less if few samples)
max_k_try = min(6, max(2, X_scaled.shape[0]-1))
best_k = None
best_score = -1
best_kmeans = None
for k in range(2, max_k_try+1):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    if len(set(labels)) > 1 and X_scaled.shape[0] >= k+1:
        s = silhouette_score(X_scaled, labels)
    else:
        s = -1
    print(f"K={k} silhouette={s:.4f}")
    if s > best_score:
        best_score = s
        best_k = k
        best_kmeans = km

print(f"Chosen K for KMeans: {best_k} (silhouette {best_score:.4f})")
df_fe["kmeans_label"] = best_kmeans.predict(X_scaled)

# ---- 9) GMM (approfondissement) ----
gmm = GaussianMixture(n_components=best_k if best_k is not None else 2, random_state=RANDOM_STATE)
df_fe["gmm_label"] = gmm.fit_predict(X_scaled)
df_fe["gmm_prob_max"] = gmm.predict_proba(X_scaled).max(axis=1)

# ---- 10) DBSCAN (approfondissement) ----
db_labels = None
if X_scaled.shape[0] >= 5:
    neigh = NearestNeighbors(n_neighbors=5).fit(X_scaled)
    distances, _ = neigh.kneighbors(X_scaled)
    eps_guess = float(np.median(distances[:, -1]))
    print("DBSCAN eps guess:", eps_guess)
    db = DBSCAN(eps=eps_guess, min_samples=5)
    df_fe["dbscan_label"] = db.fit_predict(X_scaled)
else:
    df_fe["dbscan_label"] = -1

# ---- 11) t-SNE pour visualisation (approfondissement) ----
# For speed, limit to MAX_TSNE_SAMPLES
n = X_scaled.shape[0]
if n > MAX_TSNE_SAMPLES:
    idx = np.random.RandomState(RANDOM_STATE).choice(np.arange(n), size=MAX_TSNE_SAMPLES, replace=False)
    X_tsne_input = X_scaled.iloc[idx]
    subset_idx = idx
else:
    X_tsne_input = X_scaled
    subset_idx = np.arange(n)

perplexity = 30 if X_tsne_input.shape[0] > 30 else max(5, X_tsne_input.shape[0] // 3)
tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=perplexity, init='pca', learning_rate='auto')
tsne_res = tsne.fit_transform(X_tsne_input)
tsne_df = pd.DataFrame(tsne_res, columns=["TSNE1","TSNE2"], index=subset_idx)
# Attach t-SNE coordinates back to df_fe (NaN where not computed)
df_fe["TSNE1"] = np.nan
df_fe["TSNE2"] = np.nan
df_fe.loc[tsne_df.index, "TSNE1"] = tsne_df["TSNE1"].values
df_fe.loc[tsne_df.index, "TSNE2"] = tsne_df["TSNE2"].values

# ---- 12) CLUSTER PROFILES (interprétation) ----
profile = df_fe.groupby("kmeans_label")[numeric_cols].agg(["mean","median","count"])
print("\nCluster profiles (KMeans):")
print(profile)

# ---- 13) METRICS & OUTPUTS ----
if n >= 3 and len(set(df_fe["kmeans_label"])) > 1:
    sil = silhouette_score(X_scaled, df_fe["kmeans_label"])
    print(f"\nKMeans silhouette score (final): {sil:.4f}")

print("\nKMeans counts:\n", df_fe["kmeans_label"].value_counts())
print("GMM counts:\n", df_fe["gmm_label"].value_counts())
print("DBSCAN counts:\n", df_fe["dbscan_label"].value_counts())

# Save results
df_fe.to_csv(OUT_CSV, index=False)
print("\nSaved analysis results to:", OUT_CSV)

# ---- 14) PLOTS ----
# PCA explained variance
plt.figure(figsize=(6,4))
plt.plot(np.arange(1, len(explained_ratio)+1), np.cumsum(explained_ratio), marker='o')
plt.title("Cumulative explained variance (PCA)")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.grid(True)
plt.show()

# t-SNE coloured by KMeans
plt.figure(figsize=(7,5))
subset_mask = ~df_fe["TSNE1"].isna()
for lab in sorted(df_fe["kmeans_label"].dropna().unique()):
    mask = subset_mask & (df_fe["kmeans_label"] == lab)
    plt.scatter(df_fe.loc[mask,"TSNE1"], df_fe.loc[mask,"TSNE2"], label=f"Cluster {lab}", alpha=0.7, s=20)
plt.title("t-SNE 2D projection (sample) colored by KMeans clusters")
plt.xlabel("TSNE1")
plt.ylabel("TSNE2")
plt.legend(loc="best")
plt.grid(True)
plt.show()

# ---- 15) EXECUTIVE SUMMARY (texte imprimé) ----
print("\n=== Executive summary ===")
print(f"Rows: {df.shape[0]}, Features used: {len(numeric_cols)}")
print("Key PCA variance per component:", np.round(explained_ratio, 4))
print("Best K (KMeans):", best_k, "Silhouette:", round(best_score,4))
print("GMM components:", gmm.n_components)
if 'eps_guess' in locals():
    print("DBSCAN eps guess:", eps_guess, "Clusters (excl. noise):", len(set(df_fe['dbscan_label'])) - (1 if -1 in set(df_fe['dbscan_label']) else 0))
print("Saved full results to:", OUT_CSV)

# End