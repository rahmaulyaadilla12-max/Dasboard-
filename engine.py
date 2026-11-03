"""
engine.py — Mesin rekomendasi Ruang Baca.

Port langsung dari fungsi notebook Colab bagian B (B5 & B8): `minmax_rows`, `top_n`,
`rekomendasi()` dan `buku_mirip()`. Skor CF, CB, dan Hybrid dihitung ulang secara
langsung dari matriks yang disiapkan `prepare_data.py`, sehingga slider α di dashboard
memberi hasil yang sama persis dengan notebook untuk α berapa pun.

Switching hybrid:
  pengguna inti CF          -> Weighted Hybrid  α·CF + (1−α)·CB
  punya rating, di luar inti -> Content-Based
  pengguna baru              -> buku populer (weighted rating ala IMDb)
"""
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp


class Engine:
    def __init__(self, data_dir="data"):
        p = lambda f: os.path.join(data_dir, f)
        self.meta = json.load(open(p("meta.json")))
        self.alpha_best = float(self.meta["alpha_best"])
        self.books = pd.read_parquet(p("books.parquet"))
        self.users = pd.read_parquet(p("users.parquet")).set_index("user_id")
        r = pd.read_parquet(p("ratings.parquet")).sort_values(["user_id", "rating"], ascending=[True, False])
        self.r_user = r["user_id"].to_numpy()
        self.r_book = r["book"].to_numpy()
        self.r_val = r["rating"].to_numpy().astype(np.float32)
        self.ratings = r

        z = np.load(p("model.npz"))
        csr = lambda k: sp.csr_matrix((z[f"{k}_data"], z[f"{k}_indices"], z[f"{k}_indptr"]),
                                      shape=tuple(z[f"{k}_shape"]))
        self.X_all, self.S_cf, self.R = csr("X"), csr("Scf"), csr("R")
        self.core_users = z["core_users"]
        self.core_books = z["core_books"]
        self.pool_books = z["pool_books"]
        self.nI = len(self.core_books)

        work = self.books["work"].to_numpy()
        self.work = work
        self.w_core = work[self.core_books]
        self.w_pool = work[self.pool_books]
        self.X_core = self.X_all[self.core_books]
        self.X_coreT = self.X_core.T.tocsr()
        self.X_poolT = self.X_all[self.pool_books].T.tocsr()
        self.u_pos = {int(u): k for k, u in enumerate(self.core_users)}
        self.core_pos = np.full(len(self.books), -1, dtype=np.int64)
        self.core_pos[self.core_books] = np.arange(self.nI)

        # SAME_WORK[i, j] = 1 jika dua buku inti adalah karya yang sama (edisi lain)
        _, wc = np.unique(self.w_core, return_inverse=True)
        g = sp.csr_matrix((np.ones(self.nI), (np.arange(self.nI), wc)), shape=(self.nI, wc.max() + 1))
        self.SAME_WORK = (g @ g.T).tocsr()

        # Daftar populer: weighted rating (dihitung dari inti), satu edisi per karya
        pop = self.books[self.books["wr"].notna()].sort_values("wr", ascending=False)
        self.popular = pop[~pop["work"].duplicated()].index.to_numpy()

        # Indeks pencarian: satu baris per karya, edisi wakil = edisi inti / paling banyak dirating
        b = self.books.assign(_c=self.books["in_core"].astype(int), _i=np.arange(len(self.books)))
        rep = b.sort_values(["_c", "n_ratings"], ascending=False).drop_duplicates("work")
        tot = b.groupby("work")["n_ratings"].sum()
        self.works = pd.DataFrame({"book": rep["_i"].to_numpy(), "work": rep["work"].to_numpy(),
                                   "n_total": tot.reindex(rep["work"]).to_numpy()})
        self.works["key"] = (rep["title"].str.lower() + " | " + rep["author"].str.lower()).to_numpy()
        self.works = self.works.sort_values("n_total", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------ util
    @staticmethod
    def minmax_rows(M, exclude):
        M = np.array(M, dtype=np.float32, copy=True)
        M[exclude] = np.nan
        lo, hi = np.nanmin(M, axis=1, keepdims=True), np.nanmax(M, axis=1, keepdims=True)
        return np.nan_to_num((M - lo) / np.where(hi - lo > 0, hi - lo, 1), nan=0.0)

    @staticmethod
    def top_n(scores, exclude, n, works=None):
        s = np.array(scores, dtype=np.float32, copy=True)
        s[exclude] = -np.inf
        k = min(s.shape[1] - 1, n * 4 if works is not None else n)
        top = np.argpartition(-s, k, axis=1)[:, :k]
        top = np.take_along_axis(top, np.argsort(-np.take_along_axis(s, top, axis=1), axis=1), axis=1)
        if works is None:
            return top[:, :n]
        out = np.empty((len(top), n), dtype=top.dtype)
        for r, baris in enumerate(top):
            _, pertama = np.unique(works[baris], return_index=True)
            unik = baris[np.sort(pertama)][:n]
            if len(unik) < n:
                unik = np.concatenate([unik, baris[~np.isin(baris, unik)][: n - len(unik)]])
            out[r] = unik
        return out

    @staticmethod
    def bobot_profil(nilai):
        nilai = np.asarray(nilai, dtype=np.float32)
        b = np.clip(nilai - nilai.mean(), 0, None)
        return b if b.sum() > 0 else nilai

    def history(self, user_id):
        """(indeks buku, rating) milik pengguna, terurut dari rating tertinggi."""
        a, b = np.searchsorted(self.r_user, [user_id, user_id + 1])
        return self.r_book[a:b], self.r_val[a:b]

    def segment(self, user_id):
        if int(user_id) in self.u_pos:
            return "inti"
        return "luar" if len(self.history(user_id)[0]) else "baru"

    def _frame(self, books_idx, cf, cb, hyb, dominant, because):
        out = self.books.iloc[books_idx][["isbn", "title", "author", "year", "publisher",
                                          "n_ratings", "avg_rating"]].copy()
        out["book"] = books_idx
        out["cf"], out["cb"], out["hybrid"] = cf, cb, hyb
        out["dominant"], out["because"] = dominant, because
        return out.reset_index(drop=True)

    # ------------------------------------------------ rekomendasi per pengguna
    def recommend_user(self, user_id, n=10, alpha=None):
        alpha = self.alpha_best if alpha is None else float(alpha)
        user_id = int(user_id)
        seg = self.segment(user_id)

        if seg == "inti":                                   # 1) WEIGHTED HYBRID
            r = self.R[self.u_pos[user_id]]
            B = r.copy(); B.data[:] = 1
            sudah = (B @ self.SAME_WORK).toarray() > 0
            cf_n = self.minmax_rows(np.asarray((r @ self.S_cf).todense()), sudah)
            cnt = np.diff(r.indptr)
            w = r.copy()
            w.data = np.clip(w.data - np.repeat(np.asarray(w.sum(1)).ravel() / np.maximum(cnt, 1), cnt), 0, None)
            if w.sum() == 0:
                w = r.copy()
            cb_raw = np.asarray(((w @ self.X_core) @ self.X_coreT).todense())
            cb_n = self.minmax_rows(cb_raw, sudah)
            h = alpha * cf_n + (1 - alpha) * cb_n
            top = self.top_n(h, sudah, n, works=self.w_core)[0]
            dibaca, nilai = r.indices, r.data
            bobot = w.toarray().ravel()[dibaca]
            kcf = nilai[:, None] * self.S_cf[dibaca][:, top].toarray()
            kcb = bobot[:, None] * (self.X_core[dibaca] @ self.X_coreT[:, top]).toarray()
            dom = np.where(alpha * cf_n[0, top] >= (1 - alpha) * cb_n[0, top], "CF", "CB")
            because = []
            for k in range(len(top)):
                kon = kcf[:, k] if dom[k] == "CF" else kcb[:, k]
                if kon.max() <= 0:
                    kon = kcf[:, k] + kcb[:, k]
                because.append(int(self.core_books[dibaca[kon.argmax()]]))
            df = self._frame(self.core_books[top], cf_n[0, top], cb_n[0, top], h[0, top], dom, because)
            return f"Weighted Hybrid (α = {alpha:.1f})", seg, df

        if seg == "luar":                                   # 2) SWITCHING -> CONTENT-BASED
            idx, nilai = self.history(user_id)
            bobot = self.bobot_profil(nilai)
            Xh = self.X_all[idx]
            skor = np.asarray((sp.csr_matrix(bobot) @ Xh @ self.X_poolT).todense()).ravel()
            tolak = np.isin(self.w_pool, self.work[idx])
            top = self.top_n(skor[None, :], tolak[None, :], n, works=self.w_pool)[0]
            mx = skor[top].max()
            skor_n = skor[top] / mx if mx > 0 else skor[top]
            sim = (Xh @ self.X_poolT[:, top]).toarray()
            because = [int(idx[(bobot[:, None] * sim)[:, k].argmax()]) for k in range(len(top))]
            df = self._frame(self.pool_books[top], np.nan, skor_n, skor_n, "CB", because)
            return "Switching → Content-Based", seg, df

        top = self.popular[:n]                              # 3) SWITCHING -> POPULER
        wr = self.books["wr"].to_numpy()[top] / 10
        df = self._frame(top, np.nan, np.nan, wr, "POP", -1)
        return "Switching → Populer (weighted rating)", seg, df

    # --------------------------------------------------- buku mirip buku
    def similar_books(self, book, n=10, alpha=None):
        alpha = self.alpha_best if alpha is None else float(alpha)
        j = int(self.core_pos[book])
        if j >= 0:                                          # buku inti -> hybrid item-to-item
            cf = self.S_cf[:, j].toarray().ravel()
            cf = cf / max(cf.max(), 1e-9)
            cb = (self.X_core @ self.X_core[j].T).toarray().ravel()
            cb[j] = 0
            cb = cb / max(cb.max(), 1e-9)
            h = alpha * cf + (1 - alpha) * cb
            top = self.top_n(h[None, :], (self.w_core == self.w_core[j])[None, :], n, works=self.w_core)[0]
            dom = np.where(alpha * cf[top] >= (1 - alpha) * cb[top], "CF", "CB")
            df = self._frame(self.core_books[top], cf[top], cb[top], h[top], dom, -1)
            return f"Hybrid item-to-item (α = {alpha:.1f})", True, df
        sim = np.asarray((self.X_all[book] @ self.X_poolT).todense()).ravel()
        sama = self.w_pool == self.work[book]
        top = self.top_n(sim[None, :], sama[None, :], n, works=self.w_pool)[0]
        df = self._frame(self.pool_books[top], np.nan, sim[top], sim[top], "CB", -1)
        return "Content-Based (buku di luar inti CF)", False, df

    # --------------------------------------------------------- pencarian
    def search(self, query, limit=30):
        q = str(query).strip().lower()
        if not q:
            return self.works["book"].to_numpy()[:limit]
        hit = self.works["key"].str.contains(q, regex=False)
        return self.works.loc[hit, "book"].to_numpy()[:limit]
