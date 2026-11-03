"""
prepare_data.py — Menyiapkan data dashboard Ruang Baca dari hasil bagian A.

Jalankan SEKALI (di laptop atau Google Colab) setiap kali data bersih berubah:

    python prepare_data.py Book_Ratings_Clean.csv

Hasilnya folder `data/` yang di-commit ke GitHub bersama app.py.

Logika modeling di file ini SAMA dengan notebook Colab bagian B (B1–B5 dan B8):
inti CF (pengguna >= 5 rating, buku >= 10 rating), pool Content-Based (buku >= 3 rating),
TF-IDF judul + penulis + penerbit, item-based CF cosine pola co-rating dengan 20 tetangga,
dan weighted rating (IMDb, m = 25) untuk pengguna baru. Hasil evaluasi (5-fold CV,
kurva alpha, uji cold-start) disalin dari folder `hasil_bagian_B` keluaran notebook.
"""
import html
import json
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize

# ---------- KONFIGURASI (identik dengan notebook B1) ----------
SEED = 42
USER_MIN, BOOK_MIN, POOL_MIN = 5, 10, 3
KNN = 20
W_TITLE, W_AUTHOR, W_PUBLISHER = 1.0, 1.0, 0.3
ALPHA_BEST = 0.6          # hasil pemilihan alpha pada notebook B6
M_MIN = 25                # weighted rating untuk pengguna baru (B8)

SRC = sys.argv[1] if len(sys.argv) > 1 else "Book_Ratings_Clean.csv"
EVAL_DIR = sys.argv[2] if len(sys.argv) > 2 else "hasil_bagian_B"
OUT = "data"
os.makedirs(OUT, exist_ok=True)

# ---------- B1. Muat data ----------
df = pd.read_csv(SRC, dtype={"ISBN": str}, low_memory=False)
df = df[df["Book-Rating"].between(1, 10)].drop_duplicates(["User-ID", "ISBN"]).copy()
df["User-ID"] = df["User-ID"].astype(int)

books_meta = (df.drop_duplicates("ISBN").set_index("ISBN")
              [["Book-Title", "Book-Author", "Year-Of-Publication", "Publisher"]].copy())
for kol in ["Book-Title", "Book-Author", "Publisher"]:
    books_meta[kol] = books_meta[kol].map(lambda x: html.unescape(x) if isinstance(x, str) else x)


def _kunci_karya(judul, penulis):
    t = str(judul).lower().replace("&", " and ").split(" / ")[-1]
    t = re.sub(r"\(.*?\)|\[.*?\]", " ", t).split(":")[0].strip()
    t = re.sub(r"^(.*),\s*(the|a|an)$", r"\2 \1", t)
    t = re.sub(r"[^a-z0-9]+", "", re.sub(r"^(the|a|an)\s+", "", t))
    return t + "|" + re.sub(r"[^a-z]", "", str(penulis).lower())


books_meta["work"] = [_kunci_karya(t, a) for t, a in zip(books_meta["Book-Title"], books_meta["Book-Author"])]

# ---------- B2. Inti CF & pool ----------
core = df[["User-ID", "ISBN", "Book-Rating"]].copy()
while True:
    n0 = len(core)
    core = core[core.groupby("User-ID")["ISBN"].transform("size") >= USER_MIN]
    core = core[core.groupby("ISBN")["User-ID"].transform("size") >= BOOK_MIN]
    if len(core) == n0:
        break
core_users = np.sort(core["User-ID"].unique())
core_items = np.sort(core["ISBN"].unique())
u_idx = {u: k for k, u in enumerate(core_users)}
i_idx = {b: k for k, b in enumerate(core_items)}
nU, nI = len(core_users), len(core_items)

n_per_book = df.groupby("ISBN").size()
pool_items = np.array(sorted(set(n_per_book[n_per_book >= POOL_MIN].index) | set(core_items)))

# ---------- B3. Fitur Content-Based ----------
def _norm_name(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def _onehot(values):
    cats, inv = np.unique(values, return_inverse=True)
    return sp.csr_matrix((np.ones(len(values)), (np.arange(len(values)), inv)), shape=(len(values), len(cats)))


all_items = books_meta.index.values
a_idx = {b: k for k, b in enumerate(all_items)}
judul = (books_meta["Book-Title"].fillna("").astype(str).str.lower().str.replace(r"\(.*?\)", " ", regex=True))
tfidf = TfidfVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2), sublinear_tf=True)
X_title = normalize(tfidf.fit_transform(judul))
penulis = books_meta["Book-Author"].map(_norm_name)
penerbit = books_meta["Publisher"].map(_norm_name)
penulis = np.where(penulis.isin(["", "nan"]), "na_" + books_meta.index, penulis)
penerbit = np.where(penerbit.isin(["", "nan"]), "na_" + books_meta.index, penerbit)
X_all = normalize(sp.hstack([X_title * W_TITLE, _onehot(penulis) * W_AUTHOR,
                             _onehot(penerbit) * W_PUBLISHER]).tocsr()).astype(np.float32)

# ---------- B4/B8. Item-based CF (co-rating, KNN = 20) ----------
core["u"], core["i"] = core["User-ID"].map(u_idx), core["ISBN"].map(i_idx)
R_full = sp.csr_matrix((core["Book-Rating"].astype(np.float32), (core["u"], core["i"])), shape=(nU, nI))
M = R_full.copy(); M.data[:] = 1.0
norm = np.sqrt(np.asarray(M.multiply(M).sum(0)).ravel())
S = (M.T @ M).toarray().astype(np.float32)
S /= np.outer(np.maximum(norm, 1e-9), np.maximum(norm, 1e-9))
np.fill_diagonal(S, 0)
thr = -np.sort(-S, axis=0)[KNN - 1]
S = np.where(S >= thr[None, :], S, 0)
S_cf = sp.csr_matrix(np.clip(S, 0, None).astype(np.float32))
del S, M

# ---------- B8. Weighted rating (pengguna baru) ----------
C_mean = core["Book-Rating"].mean()
stt = core.groupby("ISBN")["Book-Rating"].agg(["mean", "count"])
stt["WR"] = stt["count"] / (stt["count"] + M_MIN) * stt["mean"] + M_MIN / (stt["count"] + M_MIN) * C_mean

# ---------- Ekspor ----------
agg = df.groupby("ISBN")["Book-Rating"].agg(["size", "mean"])
codes = {w: k for k, w in enumerate(books_meta["work"].unique())}
year = pd.to_numeric(books_meta["Year-Of-Publication"], errors="coerce")
year = year.where(year.between(1000, 2026))
books = pd.DataFrame({
    "isbn": all_items,
    "title": books_meta["Book-Title"].fillna("(tanpa judul)").astype(str).values,
    "author": books_meta["Book-Author"].fillna("—").astype(str).values,
    "year": year.astype("Int16").values,
    "publisher": books_meta["Publisher"].fillna("—").astype(str).values,
    "work": books_meta["work"].map(codes).astype(np.int32).values,
    "n_ratings": agg["size"].reindex(all_items).astype(np.int32).values,
    "avg_rating": agg["mean"].reindex(all_items).astype(np.float32).round(2).values,
    "in_core": np.isin(all_items, core_items),
    "in_pool": np.isin(all_items, pool_items),
})
books["wr"] = stt["WR"].reindex(all_items).astype(np.float32).values
books.to_parquet(f"{OUT}/books.parquet", index=False)

ratings = pd.DataFrame({"user_id": df["User-ID"].astype(np.int32).values,
                        "book": df["ISBN"].map(a_idx).astype(np.int32).values,
                        "rating": df["Book-Rating"].astype(np.int8).values})
ratings.to_parquet(f"{OUT}/ratings.parquet", index=False)

# Pengguna: usia, negara, persona K-Means (label ditentukan dari centroid, bukan urutan acak klaster)
users = (df.groupby("User-ID").agg(n_ratings=("Book-Rating", "size"), avg_rating=("Book-Rating", "mean"),
                                   age=("Age", "first"), location=("Location", "first")).reset_index()
         .rename(columns={"User-ID": "user_id"}))
users["country"] = (users["location"].fillna("").str.split(",").str[-1].str.strip()
                    .replace({"": "tidak diketahui", "n/a": "tidak diketahui"}).str.title()
                    .replace({"Usa": "USA", "Tidak Diketahui": "Tidak diketahui"}))
users["in_core"] = users["user_id"].isin(core_users)
users["persona"] = pd.NA
ada = users["age"].between(5, 100)
X = StandardScaler().fit_transform(np.column_stack([np.log1p(users.loc[ada, "n_ratings"]),
                                                    users.loc[ada, "avg_rating"], users.loc[ada, "age"]]))
lab = KMeans(n_clusters=3, random_state=SEED, n_init=10).fit_predict(X)
prof = pd.DataFrame({"k": lab, "n": users.loc[ada, "n_ratings"].values,
                     "r": users.loc[ada, "avg_rating"].values}).groupby("k").agg(n=("n", "mean"), r=("r", "mean"))
prof["age"] = pd.Series(users.loc[ada, "age"].values).groupby(lab).mean()
aktif = prof["n"].idxmax()                      # klaster dengan rating terbanyak
sisa = prof.drop(aktif).sort_values("age")      # dua klaster sisanya dibedakan oleh usia
nama = {aktif: "Kutu Buku Aktif", sisa.index[0]: "Pembaca Muda Kasual", sisa.index[1]: "Pembaca Senior Kasual"}
users.loc[ada, "persona"] = [nama[k] for k in lab]
users = users.drop(columns="location")
users["age"] = users["age"].where(ada)
users.to_parquet(f"{OUT}/users.parquet", index=False)

np.savez_compressed(
    f"{OUT}/model.npz",
    X_data=X_all.data, X_indices=X_all.indices, X_indptr=X_all.indptr, X_shape=X_all.shape,
    Scf_data=S_cf.data, Scf_indices=S_cf.indices, Scf_indptr=S_cf.indptr, Scf_shape=S_cf.shape,
    R_data=R_full.data, R_indices=R_full.indices, R_indptr=R_full.indptr, R_shape=R_full.shape,
    core_users=core_users.astype(np.int32),
    core_books=np.array([a_idx[b] for b in core_items], dtype=np.int32),
    pool_books=np.array([a_idx[b] for b in pool_items], dtype=np.int32),
)

meta = {"alpha_best": ALPHA_BEST, "user_min": USER_MIN, "book_min": BOOK_MIN, "pool_min": POOL_MIN,
        "knn": KNN, "m_min": M_MIN, "n_core_users": int(nU), "n_core_books": int(nI),
        "n_pool_books": int(len(pool_items)), "n_users": int(df["User-ID"].nunique()),
        "n_books": int(df["ISBN"].nunique()), "n_works": int(books_meta["work"].nunique()),
        "n_ratings": int(len(df)), "n_core_ratings": int(len(core))}
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)

for f in ["eval_results.csv", "alpha_sweep.csv", "cold_start_eval.csv"]:
    if os.path.exists(os.path.join(EVAL_DIR, f)):
        shutil.copy(os.path.join(EVAL_DIR, f), os.path.join(OUT, f))
    else:
        print(f"PERINGATAN: {EVAL_DIR}/{f} tidak ditemukan — tab Evaluasi akan kosong untuk bagian ini.")

print(json.dumps(meta, indent=1))
for f in sorted(os.listdir(OUT)):
    print(f"{f:22s} {os.path.getsize(os.path.join(OUT, f)) / 1e6:6.2f} MB")
