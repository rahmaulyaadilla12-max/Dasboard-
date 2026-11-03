# 🌿 Ruang Baca — Sistem Rekomendasi Buku Hybrid

> *Tempat menenangkan pikiran melalui harmoni data dan sastra.*

Dashboard Streamlit untuk sistem rekomendasi buku Book-Crossing dengan metode **Hybrid (Item-Based Collaborative Filtering + Content-Based)**. Logika modelnya **sama persis** dengan notebook Google Colab bagian B; dashboard menghitung skor CF, CB, dan Hybrid secara langsung, sehingga slider α memberi hasil yang identik dengan notebook.

## Struktur

```
book-ratings-main/
├─ app.py               # tampilan dashboard (5 halaman)
├─ engine.py            # mesin rekomendasi (port fungsi notebook B5 & B8)
├─ prepare_data.py      # menyiapkan folder data/ dari Book_Ratings_Clean.csv
├─ data/                # hasil prepare_data.py (±17 MB) — WAJIB ikut di-commit
├─ .streamlit/config.toml   # tema terang (mencegah angka grafik berwarna putih)
├─ assets/background.jpg
└─ requirements.txt
```

## Halaman

| Halaman | Isi |
|---|---|
| 🏠 Beranda | Pencarian judul/penulis, koleksi terpopuler (weighted rating), baca sesuai mood |
| 🎯 Kurator Personal | Pilih satu buku → buku mirip. Buku inti CF: **Hybrid item-to-item** (slider α); buku lain: **Content-Based** |
| 👤 Rekomendasi Pengguna | Masukkan User-ID → **switching hybrid**: inti → Weighted Hybrid, di luar inti → Content-Based, baru → populer. Ada skor CF/CB/Hybrid dan alasan "Karena menyukai …" |
| 📊 Estetika Data | KPI, tren tahun terbit, sebaran rating, ekor panjang rating per buku, usia, negara, persona K-Means |
| 🧪 Evaluasi Model | Hasil 5-fold CV notebook: Precision/Recall/HitRate@10, kurva pemilihan α, uji cold-start, cakupan |

## Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Memperbarui data (hanya jika data bersih bagian A berubah)

```bash
pip install scikit-learn
python prepare_data.py Book_Ratings_Clean.csv hasil_bagian_B
```

`hasil_bagian_B` adalah folder keluaran notebook bagian B (berisi `eval_results.csv`, `alpha_sweep.csv`, `cold_start_eval.csv`). Commit ulang folder `data/` setelahnya.

## Catatan metodologis

- Hanya rating eksplisit (1–10). Inti CF: pengguna ≥ 5 rating, buku ≥ 10 rating (4.743 × 3.057). Pool Content-Based: buku ≥ 3 rating (27.804).
- CF = cosine pada pola "dirating bersama" (Amazon item-to-item), 20 tetangga terdekat. CB = TF-IDF judul + penulis + penerbit (bobot 0,3).
- Skor CF dan CB dinormalkan min-max 0–1 per pengguna sebelum digabung: `Hybrid = α·CF + (1−α)·CB`, α = 0,6 (hasil 5-fold CV).
- Edisi berbeda dari karya yang sama (hardcover, paperback, large print) digabung, jadi tidak muncul berulang.
- Label persona K-Means ditentukan dari profil klaster, bukan dari nomor klaster acak.
