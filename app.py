# ==========================================================
# RUANG BACA — Dashboard Sistem Rekomendasi Buku (Hybrid)
# Tema: Tyrian Purple × Citron × Cream
# Model: port langsung dari notebook Colab bagian B (lihat engine.py)
# ==========================================================
import html
import json
import os
import zlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import Engine

st.set_page_config(page_title="Ruang Baca", page_icon="🌿", layout="wide", initial_sidebar_state="auto")

# ===== PALET =====
PURPLE, CITRON, CREAM = "#66023C", "#CAD183", "#FFFDD0"
INK, SOFT, MUTED = "#3D0A22", "#F6F3C8", "#8A6A78"
C_CF, C_CB, C_HYB, C_POP, C_RND = "#2F5D8A", "#5E7A1F", "#66023C", "#C98A12", "#9E9E9E"

# ===== CSS =====
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;0,800;1,400;1,600&family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400;1,500&display=swap');

html, body, .stApp { color: __INK__; }
.stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stWidgetLabel"] {
    font-family: 'Cormorant Garamond', Georgia, serif; font-size: 17px;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { font-family: 'Playfair Display', Georgia, serif; color: __PURPLE__; }
/* Ikon bawaan Streamlit HARUS tetap pakai font ikon — tanpa ini muncul teks "keyboard_double_arrow_left" */
[data-testid="stIconMaterial"], .material-symbols-rounded, span[class*="material"] {
    font-family: 'Material Symbols Rounded' !important;
}
.stApp { background: __CREAM__; }
[data-testid="stHeader"] { background: transparent; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background-color: __PURPLE__; border-right: 2px solid __CITRON__; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span,
[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] { color: __CREAM__; }
[data-testid="stSidebar"] [role="radiogroup"] label { border-radius: 8px; padding: 6px 10px; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(202,209,131,.15); }
[data-testid="stSidebar"] [data-testid="stIconMaterial"] { color: __CREAM__; }

/* ---------- Input ---------- */
[data-testid="stTextInput"] div[data-baseweb="input"], [data-testid="stNumberInput"] div[data-baseweb="input"],
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: #FFFEF0; border: 2px solid __PURPLE__; border-radius: 10px;
}
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input { color: __INK__; font-size: 17px; }
[data-testid="stSelectbox"] div[data-baseweb="select"] * { color: __INK__; }
[data-baseweb="popover"] li { font-family: 'Cormorant Garamond', serif; font-size: 16px; color: __INK__; }

/* ---------- Tombol ---------- */
[data-testid="stButton"] button {
    background-color: __PURPLE__; border: none; border-radius: 8px; padding: 6px 14px;
}
[data-testid="stButton"] button p { color: __CREAM__; font-family: 'Playfair Display', serif; font-weight: 700;
    font-size: 13px; letter-spacing: .8px; }
[data-testid="stButton"] button:hover { background-color: __INK__; }

/* ---------- Header & judul ---------- */
.top-header { background: __PURPLE__; padding: 30px 36px; border-radius: 18px; display: flex;
    justify-content: space-between; align-items: center; gap: 16px; margin-bottom: 26px; box-shadow: 0 8px 0 __CITRON__; }
.header-title { font-family: 'Playfair Display', serif; font-size: 34px; font-weight: 700; color: __CREAM__; line-height: 1.15; }
.header-title em { color: __CITRON__; }
.header-subtitle { font-family: 'Cormorant Garamond', serif; font-size: 19px; font-style: italic; color: __CREAM__; opacity: .9; margin-top: 6px; }
.header-badge { background: __CITRON__; color: __PURPLE__; font-family: 'Playfair Display', serif; font-size: 11px;
    font-weight: 700; padding: 7px 16px; border-radius: 999px; letter-spacing: 2px; white-space: nowrap; }
.section-title { font-family: 'Playfair Display', serif; font-size: 30px; font-weight: 600; color: __PURPLE__;
    text-align: center; margin: 40px 0 6px; font-style: italic; }
.section-title::after { content: ''; display: block; width: 60px; height: 3px; background: __CITRON__; margin: 10px auto 0; border-radius: 2px; }
.section-sub { font-family: 'Cormorant Garamond', serif; text-align: center; color: __PURPLE__; opacity: .8;
    font-size: 14px; letter-spacing: 3px; margin-bottom: 22px; text-transform: uppercase; font-style: italic; }
.chart-title { font-family: 'Playfair Display', serif; color: __PURPLE__; font-style: italic; font-size: 21px;
    text-align: center; margin: 14px 0 0; }
.chart-sub { font-family: 'Cormorant Garamond', serif; color: __INK__; font-size: 13px; letter-spacing: 2px;
    text-align: center; margin-bottom: 4px; text-transform: uppercase; }
.info-box { background: #FFFEF0; border: 1.5px solid __PURPLE__; border-left: 6px solid __CITRON__; padding: 14px 20px;
    border-radius: 8px; margin: 6px 0 18px; font-family: 'Cormorant Garamond', serif; font-size: 17px; line-height: 1.6; color: __INK__; }
.info-box b { color: __PURPLE__; }

/* ---------- KPI ---------- */
.kpi-card { background: #FFFEF0; border-radius: 14px; padding: 20px 16px; text-align: center; border: 2px solid __PURPLE__;
    box-shadow: 5px 5px 0 __PURPLE__; height: 100%; }
.kpi-icon { font-size: 24px; margin-bottom: 8px; }
.kpi-title { font-family: 'Cormorant Garamond', serif; font-size: 12.5px; color: __PURPLE__; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; }
.kpi-value { font-family: 'Playfair Display', serif; font-size: 31px; font-weight: 700; color: __PURPLE__; line-height: 1.1; margin-top: 4px; }
.kpi-value small { font-family: 'Cormorant Garamond', serif; font-size: 16px; color: __INK__; font-style: italic; }
.kpi-note { font-family: 'Cormorant Garamond', serif; font-size: 14px; color: __INK__; margin-top: 4px; font-style: italic; }

/* ---------- Mood ---------- */
.mood-card { background: #FFFEF0; border-radius: 14px; padding: 20px 12px; border: 2px solid __PURPLE__; box-shadow: 5px 5px 0 __PURPLE__;
    text-align: center; margin-bottom: 10px; }
.mood-card.active { background: __CITRON__; }
.mood-icon { font-size: 34px; margin-bottom: 8px; }
.mood-label { font-family: 'Playfair Display', serif; font-size: 17px; font-weight: 700; color: __PURPLE__; font-style: italic; }

/* ---------- Rak buku (kartu) ---------- */
.rb-shelf { display: grid; grid-template-columns: repeat(auto-fill, minmax(168px, 1fr)); gap: 20px 18px; margin: 6px 0 10px; }
.rb-card { background: #FFFEF0; border: 2px solid __PURPLE__; border-radius: 12px; box-shadow: 5px 5px 0 __PURPLE__;
    display: flex; flex-direction: column; overflow: hidden; transition: transform .2s, box-shadow .2s; position: relative; }
.rb-card:hover { transform: translate(-2px,-2px); box-shadow: 7px 7px 0 __CITRON__, 7px 7px 0 2px __PURPLE__; }
.rb-cover { position: relative; aspect-ratio: 2 / 3; max-height: 250px; background: __SOFT__; display: flex;
    align-items: center; justify-content: center; border-bottom: 2px solid __PURPLE__; }
.rb-cover img { width: 100%; height: 100%; object-fit: cover; }
.rb-ph { display: none; position: absolute; inset: 0; align-items: center; justify-content: center; text-align: center;
    padding: 14px; color: __CREAM__; font-family: 'Playfair Display', serif; font-weight: 700; font-size: 15px; line-height: 1.3; }
.rb-rank { position: absolute; top: 8px; left: 8px; background: __CITRON__; color: __PURPLE__; font-family: 'Playfair Display', serif;
    font-weight: 800; font-size: 12px; padding: 3px 9px; border-radius: 20px; border: 1.5px solid __PURPLE__; z-index: 2; }
.rb-tag { position: absolute; top: 8px; right: 8px; font-family: 'Cormorant Garamond', serif; font-weight: 700; font-size: 12px;
    padding: 2px 8px; border-radius: 20px; z-index: 2; border: 1px solid rgba(0,0,0,.15); }
.rb-over { position: absolute; inset: 0; background: rgba(61,10,34,.9); color: __CREAM__; opacity: 0; transition: opacity .2s;
    padding: 16px 14px; font-family: 'Cormorant Garamond', serif; font-size: 15px; line-height: 1.45; z-index: 3;
    display: flex; flex-direction: column; justify-content: center; }
.rb-over b { color: __CITRON__; font-family: 'Playfair Display', serif; }
.rb-card:hover .rb-over { opacity: 1; }
.rb-body { padding: 10px 12px 12px; display: flex; flex-direction: column; gap: 3px; flex: 1; }
.rb-title { font-family: 'Playfair Display', serif; font-weight: 700; font-size: 15px; color: __PURPLE__; line-height: 1.25;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; min-height: 2.5em; }
.rb-author { font-family: 'Cormorant Garamond', serif; font-style: italic; font-size: 15px; color: __INK__;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rb-meta { font-family: 'Cormorant Garamond', serif; font-size: 14px; color: __INK__; }
.rb-bar { display: flex; align-items: center; gap: 6px; font-family: 'Cormorant Garamond', serif; font-size: 13.5px; color: __INK__; }
.rb-bar .l { width: 44px; font-weight: 700; }
.rb-bar .t { flex: 1; height: 7px; background: #E9E5C0; border-radius: 4px; overflow: hidden; }
.rb-bar .f { height: 100%; border-radius: 4px; }
.rb-bar .v { width: 30px; text-align: right; font-weight: 700; }
.rb-why { font-family: 'Cormorant Garamond', serif; font-size: 14px; color: __INK__; font-style: italic; margin-top: 4px;
    border-top: 1px dashed rgba(102,2,60,.3); padding-top: 5px; line-height: 1.3;
    display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
.rb-why b { color: __PURPLE__; font-style: normal; }
.rb-pill { display: inline-block; background: __CITRON__; color: __PURPLE__; font-family: 'Playfair Display', serif; font-weight: 700;
    font-size: 12px; padding: 2px 10px; border-radius: 999px; margin-top: 3px; width: fit-content; }

/* ---------- Kartu referensi (hero) ---------- */
.rb-hero { display: flex; gap: 22px; align-items: center; background: #FFFEF0; border: 2px solid __PURPLE__; border-radius: 14px;
    box-shadow: 6px 6px 0 __CITRON__, 6px 6px 0 2px __PURPLE__; padding: 16px 20px; margin: 8px 0 18px; }
.rb-hero .rb-cover { width: 110px; min-width: 110px; border: 2px solid __PURPLE__; border-radius: 6px; overflow: hidden; }
.rb-hero h4 { margin: 0 0 4px; font-size: 22px; }
.rb-hero .rb-meta { font-size: 16px; }
.rb-status { display: inline-block; margin-top: 8px; padding: 4px 12px; border-radius: 999px; font-family: 'Cormorant Garamond', serif;
    font-weight: 700; font-size: 15px; }
.rb-legend { display: flex; flex-wrap: wrap; gap: 16px; font-family: 'Cormorant Garamond', serif; font-size: 15px; color: __INK__; margin: 2px 0 10px; }
.rb-legend i { display: inline-block; width: 22px; height: 8px; border-radius: 4px; margin-right: 6px; vertical-align: middle; }
.rb-table { overflow-x: auto; margin: 4px 0 12px; }
.rb-table table { width: 100%; border-collapse: collapse; font-family: 'Cormorant Garamond', serif; font-size: 16px; color: __INK__;
    background: #FFFEF0; border: 2px solid __PURPLE__; border-radius: 10px; overflow: hidden; }
.rb-table th, .rb-table td { padding: 7px 12px; border-bottom: 1px solid rgba(102,2,60,.15); text-align: right; white-space: nowrap; }
.rb-table thead th { background: __PURPLE__; color: __CREAM__; font-family: 'Playfair Display', serif; font-size: 14px; text-align: right; }
.rb-table tbody th { text-align: left; color: __PURPLE__; font-weight: 700; }
.rb-table thead th:first-child { text-align: left; }
.rb-table tbody tr:nth-child(even) { background: rgba(202,209,131,.18); }
@media (max-width: 640px) {
  .top-header { flex-direction: column; align-items: flex-start; padding: 22px; }
  .header-title { font-size: 26px; }
  .rb-shelf { grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 14px; }
  .rb-hero { flex-direction: column; align-items: flex-start; }
}
</style>
"""
for k, v in {"__PURPLE__": PURPLE, "__CITRON__": CITRON, "__CREAM__": CREAM, "__INK__": INK, "__SOFT__": SOFT}.items():
    CSS = CSS.replace(k, v)
st.markdown(CSS, unsafe_allow_html=True)

# Script fallback sampul: Amazon -> Open Library -> kartu warna berisi judul.
# (Atribut onerror di st.markdown tidak dijalankan Streamlit, jadi dipasang lewat script.)
COVER_JS = """<script>
document.querySelectorAll('img.rb-img:not([data-bound])').forEach(function(im){
  im.dataset.bound = '1';
  var gagal = function(){
    var fb = im.dataset.fb;
    if (fb) { im.dataset.fb = ''; im.src = fb; }
    else { im.style.display = 'none'; var ph = im.parentElement.querySelector('.rb-ph'); if (ph) ph.style.display = 'flex'; }
  };
  var cek = function(){ if (im.naturalWidth < 10) gagal(); };   // Amazon mengirim gambar 1x1 bila sampul tak ada
  im.addEventListener('error', gagal);
  im.addEventListener('load', cek);
  if (im.complete) { if (im.naturalWidth === 0) gagal(); else cek(); }
});
</script>"""


# ===== DATA & MODEL =====
@st.cache_resource(show_spinner="Menyiapkan perpustakaan Ruang Baca…")
def get_engine():
    return Engine("data")


@st.cache_data
def load_eval():
    def rd(f):
        p = os.path.join("data", f)
        return pd.read_csv(p) if os.path.exists(p) else None
    return rd("eval_results.csv"), rd("alpha_sweep.csv"), rd("cold_start_eval.csv")


E = get_engine()
BOOKS = E.books
META = E.meta
ALPHA = E.alpha_best


# ===== HELPER TAMPILAN =====
esc = lambda s: html.escape(str(s), quote=True)


def fmt_year(y):
    return "—" if pd.isna(y) else str(int(y))


def cover_html(isbn, title):
    up = str(isbn).upper()
    hue = zlib.crc32(up.encode()) % 360
    amazon = f"https://images.amazon.com/images/P/{up}.01.LZZZZZZZ.jpg"
    ol = f"https://covers.openlibrary.org/b/isbn/{up}-M.jpg?default=false"
    return (f'<img class="rb-img" loading="lazy" src="{amazon}" data-fb="{ol}" alt="">'
            f'<div class="rb-ph" style="background:linear-gradient(150deg,{PURPLE},hsl({hue},45%,38%))">{esc(title)}</div>')


def bar(label, val, color):
    if val is None or pd.isna(val):
        return f'<div class="rb-bar"><span class="l">{label}</span><span class="t"></span><span class="v">—</span></div>'
    w = max(0.0, min(1.0, float(val))) * 100
    return (f'<div class="rb-bar"><span class="l">{label}</span><span class="t"><span class="f" '
            f'style="display:block;width:{w:.0f}%;background:{color}"></span></span><span class="v">{float(val):.2f}</span></div>')


TAGS = {"CF": ("#DCE7F2", C_CF, "dominan CF"), "CB": ("#E6ECCB", "#4B6118", "dominan CB"),
        "POP": (CITRON, PURPLE, "populer")}


def shelf(rows, ranked=True, bars=False, tag=None, why=False, pill=None):
    """Render satu rak buku. rows: DataFrame dengan kolom isbn, title, author, year, publisher, n_ratings, avg_rating
    (+ cf, cb, hybrid, dominant, because untuk mode rekomendasi)."""
    cards = []
    for k, r in enumerate(rows.itertuples(index=False), start=1):
        rating = f"★ {r.avg_rating:.1f} · {int(r.n_ratings):,} rating" if r.n_ratings else "belum dirating"
        t = None
        if tag == "dominant":
            t = TAGS.get(getattr(r, "dominant", ""), None)
        elif tag:
            t = (CITRON, PURPLE, tag)
        tag_html = f'<span class="rb-tag" style="background:{t[0]};color:{t[1]}">{t[2]}</span>' if t else ""
        rank_html = f'<span class="rb-rank">#{k}</span>' if ranked else ""
        body = [f'<div class="rb-title" title="{esc(r.title)}">{esc(r.title)}</div>',
                f'<div class="rb-author">{esc(r.author)}</div>',
                f'<div class="rb-meta">{fmt_year(r.year)} · {rating}</div>']
        if pill is not None:
            body.append(f'<span class="rb-pill">{esc(pill(r))}</span>')
        if bars:
            body += [bar("CF", r.cf, C_CF), bar("CB", r.cb, C_CB), bar("Hybrid", r.hybrid, C_HYB)]
        if why and getattr(r, "because", -1) is not None and int(r.because) >= 0:
            body.append(f'<div class="rb-why">Karena menyukai <b>{esc(BOOKS.title.iat[int(r.because)])}</b></div>')
        over = (f'<div class="rb-over"><b>{esc(r.title)}</b><br>{esc(r.author)}<br><br>Penerbit: {esc(r.publisher)}'
                f'<br>Terbit: {fmt_year(r.year)}<br>Rating komunitas: {rating}<br>ISBN: {esc(r.isbn)}</div>')
        cards.append(f'<div class="rb-card"><div class="rb-cover">{rank_html}{tag_html}{cover_html(r.isbn, r.title)}{over}</div>'
                     f'<div class="rb-body">{"".join(body)}</div></div>')
    st.html(f'<div class="rb-shelf">{"".join(cards)}</div>{COVER_JS}', unsafe_allow_javascript=True)


def legend():
    st.html(f'<div class="rb-legend"><span><i style="background:{C_CF}"></i>CF — pola rating pembaca serupa</span>'
            f'<span><i style="background:{C_CB}"></i>CB — kemiripan judul, penulis, penerbit</span>'
            f'<span><i style="background:{C_HYB}"></i>Hybrid — campuran α·CF + (1−α)·CB</span>'
            f'<span style="opacity:.85">Skor 0–1 bersifat relatif: 1 = kandidat terkuat dalam daftar ini.</span></div>')


def books_frame(idx):
    return BOOKS.iloc[np.asarray(idx, dtype=int)][["isbn", "title", "author", "year", "publisher", "n_ratings", "avg_rating"]]


def title_block(title, sub):
    st.markdown(f'<div class="section-title">{title}</div><div class="section-sub">{sub}</div>', unsafe_allow_html=True)


def kpi(col, icon, title, value, note=""):
    small = ' style="font-size:22px"' if len(str(value)) > 12 and "<" not in str(value) else ""
    col.markdown(f'<div class="kpi-card"><div class="kpi-icon">{icon}</div><div class="kpi-title">{title}</div>'
                 f'<div class="kpi-value"{small}>{value}</div><div class="kpi-note">{note}</div></div>', unsafe_allow_html=True)


def style_fig(fig, height=340, **kw):
    base = dict(
        height=height, plot_bgcolor="rgba(255,254,240,0.9)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Cormorant Garamond, Georgia, serif", color=INK, size=15),
        margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
        hoverlabel=dict(bgcolor=CREAM, bordercolor=PURPLE, font=dict(color=INK, family="Cormorant Garamond", size=15)),
        legend=dict(font=dict(color=INK, size=14), bgcolor="rgba(0,0,0,0)"))
    base.update(kw)                       # argumen khusus per grafik menimpa bawaan
    fig.update_layout(**base)
    fig.update_xaxes(showgrid=False, tickfont=dict(color=INK, size=14), title_font=dict(color=INK, size=15),
                     linecolor="rgba(102,2,60,.4)", automargin=True)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(102,2,60,.12)", tickfont=dict(color=INK, size=14),
                     title_font=dict(color=INK, size=15), zeroline=False, automargin=True)
    return fig


def show(fig):
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False, "responsive": True})


def table(df, fmt=None):
    """Tabel HTML bertema (st.dataframe memotong kolom dan memakai warna bawaan)."""
    fmt = fmt or {}
    head = "".join(f"<th>{esc(c)}</th>" for c in [df.index.name or ""] + list(df.columns))
    rows = ""
    for i, r in df.iterrows():
        cells = "".join(f"<td>{fmt.get(c, '{}').format(v) if not isinstance(v, str) else esc(v)}</td>" for c, v in r.items())
        rows += f"<tr><th>{esc(i)}</th>{cells}</tr>"
    st.html(f'<div class="rb-table"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>')


def chart_head(title, sub):
    st.markdown(f'<div class="chart-title">{title}</div><div class="chart-sub">{sub}</div>', unsafe_allow_html=True)


# ===== SIDEBAR =====
PAGES = ["🏠 Beranda", "🎯 Kurator Personal", "👤 Rekomendasi Pengguna", "📊 Estetika Data", "🧪 Evaluasi Model"]
with st.sidebar:
    st.markdown('<div style="text-align:center;font-size:46px;margin-bottom:2px">🌿</div>'
                '<h1 style="text-align:center;margin:0;font-family:\'Playfair Display\',serif;font-weight:700;'
                f'font-style:italic;font-size:34px;color:{CREAM}">Ruang Baca</h1>'
                f'<p style="text-align:center;font-size:12px;letter-spacing:3px;margin:6px 0 22px;color:{CITRON}">'
                'TEMPAT MENENANGKAN PIKIRAN</p>', unsafe_allow_html=True)
    page = st.radio("Navigasi", PAGES)
    if page in ("🎯 Kurator Personal", "👤 Rekomendasi Pengguna"):
        st.markdown("<hr style='border:0;border-top:1px solid rgba(202,209,131,.35);margin:16px 0'>", unsafe_allow_html=True)
        n_reco = st.select_slider("Jumlah rekomendasi", options=[5, 10, 15, 20], value=10)
    else:
        n_reco = 10
    st.markdown(f'<div style="margin-top:40px;font-size:13px;text-align:center;letter-spacing:1.5px;color:{CITRON};line-height:1.6">'
                f'HYBRID CF + CB · α = {ALPHA}<br>{META["n_ratings"]:,} rating · {META["n_users"]:,} pembaca</div>',
                unsafe_allow_html=True)


# ==========================================
# HALAMAN 1: BERANDA
# ==========================================
MOODS = {
    "🍵 Menenangkan Pikiran": r"\b(?:spirit\w*|philosoph\w*|meditat\w*|zen|peace\w*|soul|poem|poems|poetry|inspir\w*|wisdom|buddh\w*)\b",
    "✨ Petualangan Magis": r"\b(?:fantasy|magic\w*|wizard\w*|dragon\w*|quest|sword\w*|elf|elves|myth\w*|sorcer\w*|witch\w*)\b",
    "🌱 Pengembangan Diri": r"\b(?:self|success\w*|habits?|leader\w*|motivat\w*|productiv\w*|psycholog\w*|mind|business|happiness)\b",
    "💌 Romansa Klasik": r"\b(?:romance|love|lover\w*|bride|passion\w*|heart|kiss|wedding|duke|regency|courtship)\b",
}


@st.cache_data
def mood_books(pattern, n=12):
    b = BOOKS[BOOKS["in_pool"]]
    hit = b["title"].str.contains(pattern, case=False, regex=True)
    if "romance" in pattern:
        hit |= b["publisher"].str.contains(r"harlequin|silhouette|avon", case=False, regex=True)
    b = b[hit].sort_values("n_ratings", ascending=False)
    return b[~b["work"].duplicated()].index[:n].to_numpy()


if page == "🏠 Beranda":
    st.markdown(f"""<div class="top-header"><div>
        <div class="header-title">Selamat datang di <em>Ruang Baca</em>.</div>
        <div class="header-subtitle">Temukan buku yang menenangkan pikiranmu hari ini — {META['n_works']:,} karya,
        {META['n_users']:,} pembaca.</div></div><div class="header-badge">EST. 2026</div></div>""",
                unsafe_allow_html=True)

    q = st.text_input("Pencarian Buku", placeholder="🔍 Cari judul atau penulis favoritmu…", label_visibility="collapsed")
    if q.strip():
        hasil = E.search(q, limit=12)
        if len(hasil):
            st.markdown(f'<div class="info-box">Menemukan <b>{len(hasil)}</b> karya teratas untuk “{esc(q)}”, '
                        'diurutkan dari yang paling banyak dibaca. Arahkan kursor ke sampul untuk detail; '
                        'buka <b>Kurator Personal</b> untuk mencari buku yang mirip.</div>', unsafe_allow_html=True)
            shelf(books_frame(hasil), ranked=False, tag="HASIL")
        else:
            st.warning("Buku tidak ditemukan. Coba kata kunci lain (judul dalam bahasa Inggris atau nama penulis).")

    title_block("Koleksi Terpopuler", "weighted rating — disukai banyak pembaca")
    shelf(books_frame(E.popular[:12]), ranked=True, tag="TERPOPULER")

    title_block("Baca Sesuai Mood", "pilih suasana hatimu")
    mood_cols = st.columns(4)
    sel = st.session_state.get("selected_mood")
    for i, mood in enumerate(MOODS):
        icon, label = mood.split(" ", 1)
        with mood_cols[i]:
            st.markdown(f'<div class="mood-card{" active" if sel == mood else ""}"><div class="mood-icon">{icon}</div>'
                        f'<div class="mood-label">{label}</div></div>', unsafe_allow_html=True)
            if st.button("✓ TERPILIH" if sel == mood else "PILIH", key=f"mood_{i}", width="stretch"):
                st.session_state["selected_mood"] = None if sel == mood else mood
                st.rerun()
    if sel:
        idx = mood_books(MOODS[sel])
        title_block(f"Untuk suasana “{sel.split(' ', 1)[1]}”", "kata kunci judul · diurutkan dari yang paling banyak dibaca")
        if len(idx):
            shelf(books_frame(idx), ranked=False, tag="MOOD")
        else:
            st.info("Belum ada buku yang cocok untuk mood ini.")

# ==========================================
# HALAMAN 2: KURATOR PERSONAL (buku -> buku)
# ==========================================
elif page == "🎯 Kurator Personal":
    title_block("Kurasi Spesial Untukmu", "pilih satu buku, temukan yang senada")
    st.markdown('<div class="info-box">Ketik judul atau penulis, lalu pilih bukunya. Untuk buku yang cukup sering dirating '
                '(inti CF, 3.057 buku) sistem memakai <b>Hybrid item-to-item</b>: gabungan pola “pembaca buku ini juga '
                'membaca…” (CF) dan kemiripan isi (CB). Untuk buku lain sistem beralih ke <b>Content-Based</b>.</div>',
                unsafe_allow_html=True)

    q = st.text_input("Cari buku", placeholder="Ketik judul atau penulis, mis. harry potter, grisham, discworld…")
    opsi = E.search(q, limit=40)
    if q.strip() and len(opsi) == 0:
        st.warning("Tidak ada judul atau penulis yang cocok. Coba kata kunci yang lebih pendek.")
        st.stop()

    def label(b):
        r = BOOKS.iloc[int(b)]
        return f"{r.title} — {r.author} ({fmt_year(r.year)}) · {int(r.n_ratings):,} rating"

    pilih = st.selectbox("Pilih buku referensimu" + ("" if q.strip() else " (atau ketik di kotak pencarian)"),
                         opsi, format_func=label, index=0 if q.strip() else None,
                         placeholder="Pilih dari 40 karya paling banyak dibaca…")
    if pilih is None:
        st.stop()

    ref = BOOKS.iloc[int(pilih)]
    inti = bool(ref.in_core)
    status = (("Inti CF → Hybrid item-to-item", "#DCE7F2", C_CF) if inti
              else ("Di luar inti CF → Content-Based", "#E6ECCB", "#4B6118"))
    st.html(f'<div class="rb-hero"><div class="rb-cover">{cover_html(ref.isbn, ref.title)}</div><div>'
            f'<h4>{esc(ref.title)}</h4><div class="rb-meta"><i>{esc(ref.author)}</i> · {esc(ref.publisher)} · {fmt_year(ref.year)}</div>'
            f'<div class="rb-meta">★ {ref.avg_rating:.1f} dari {int(ref.n_ratings):,} rating</div>'
            f'<span class="rb-status" style="background:{status[1]};color:{status[2]}">{status[0]}</span></div></div>{COVER_JS}',
            unsafe_allow_javascript=True)

    alpha = ALPHA
    if inti:
        alpha = st.slider("Bobot α — kiri: mirip isi (CB) · kanan: mirip selera pembaca (CF)",
                          0.0, 1.0, ALPHA, 0.1, help=f"α = {ALPHA} adalah nilai terbaik hasil 5-fold CV di notebook.", key="alpha_item")
    metode, _, rec = E.similar_books(int(pilih), n=n_reco, alpha=alpha)
    st.markdown(f"<h3 style='font-style:italic;margin:6px 0 2px'>✨ Pembaca buku ini juga menyukai…</h3>"
                f"<div class='rb-meta' style='margin-bottom:6px'>Metode: <b>{metode}</b></div>", unsafe_allow_html=True)
    if inti:
        legend()
    shelf(rec, ranked=True, bars=inti, tag="dominant" if inti else "CB")

# ==========================================
# HALAMAN 3: REKOMENDASI PENGGUNA (User-ID)
# ==========================================
elif page == "👤 Rekomendasi Pengguna":
    title_block("Rak Pribadi Pembaca", "masukkan user-id, lihat racikan untuknya")
    st.markdown('<div class="info-box">Sistem memilih jalurnya sendiri (<b>switching hybrid</b>): pengguna inti '
                f'(≥ {META["user_min"]} rating pada buku inti) → <b>Weighted Hybrid</b>; punya rating tetapi di luar inti → '
                '<b>Content-Based</b>; pengguna baru tanpa riwayat → <b>buku populer</b>.</div>', unsafe_allow_html=True)

    if "uid" not in st.session_state:
        st.session_state["uid"] = 46671

    def set_uid(v):
        st.session_state["uid"] = v

    c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1], vertical_alignment="bottom")
    uid = int(c1.number_input("User-ID", min_value=0, step=1, key="uid"))
    c2.button("Contoh: pengguna inti", on_click=set_uid, args=(46671,), width="stretch")
    c3.button("Contoh: di luar inti", on_click=set_uid, args=(209919,), width="stretch")
    c4.button("Pengguna baru", on_click=set_uid, args=(0,), width="stretch")

    seg = E.segment(uid)
    hist_b, hist_r = E.history(uid)
    if uid in E.users.index:
        u = E.users.loc[uid]
        k1, k2, k3, k4 = st.columns(4)
        kpi(k1, "📚", "Buku dinilai", f"{int(u.n_ratings):,}")
        kpi(k2, "⭐", "Rata-rata rating", f"{u.avg_rating:.1f}<small> / 10</small>")
        kpi(k3, "🌍", "Negara", esc(u.country))
        kpi(k4, "🧭", "Persona", esc(u.persona) if pd.notna(u.persona) else "—",
            f"usia {int(u.age)}" if pd.notna(u.age) else "usia tidak diisi")
    elif uid != 0:
        st.markdown(f'<div class="info-box">User-ID <b>{uid}</b> tidak ada di data, jadi diperlakukan sebagai '
                    '<b>pengguna baru</b>.</div>', unsafe_allow_html=True)

    if len(hist_b):
        st.markdown("<h3 style='font-style:italic;margin:22px 0 6px'>📖 Buku yang paling ia sukai</h3>", unsafe_allow_html=True)
        top_h = pd.Series(hist_r, index=hist_b).head(6)
        hb = books_frame(top_h.index).assign(rr=top_h.values)
        shelf(hb, ranked=False, pill=lambda r: f"rating darinya: {int(r.rr)}/10")

    alpha = ALPHA
    if seg == "inti":
        alpha = st.slider("Bobot α — kiri: mirip isi (CB) · kanan: mirip selera pembaca (CF)",
                          0.0, 1.0, ALPHA, 0.1, help=f"α = {ALPHA} adalah nilai terbaik hasil 5-fold CV di notebook.", key="alpha_user")
        sample = E.users[E.users.in_core].sort_values("n_ratings", ascending=False).index[:400:40].tolist()
        st.caption("User-ID inti lain untuk dicoba: " + ", ".join(map(str, sample)))
    metode, seg, rec = E.recommend_user(uid, n=n_reco, alpha=alpha)
    st.markdown(f"<h3 style='font-style:italic;margin:14px 0 2px'>✨ Rekomendasi untuknya</h3>"
                f"<div class='rb-meta' style='margin-bottom:6px'>Metode: <b>{metode}</b></div>", unsafe_allow_html=True)
    if seg == "inti":
        legend()
    shelf(rec, ranked=True, bars=(seg == "inti"), tag="dominant", why=(seg != "baru"))

# ==========================================
# HALAMAN 4: ESTETIKA DATA
# ==========================================
elif page == "📊 Estetika Data":
    title_block("Harmoni Ekosistem Buku", "data bersih hasil bagian A")
    R = E.ratings
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "📚", "Koleksi buku", f"{META['n_books']:,}", f"{META['n_works']:,} karya (edisi digabung)")
    kpi(k2, "👥", "Pembaca", f"{META['n_users']:,}", "punya ≥ 1 rating eksplisit")
    kpi(k3, "💌", "Rating eksplisit", f"{META['n_ratings']:,}", "skala 1–10")
    kpi(k4, "⭐", "Rata-rata rating", f"{R['rating'].mean():.2f}<small> / 10</small>", "cenderung positif")

    chart_head("Tren Tahun Terbit", "jumlah buku yang dirating, per tahun terbit (1950–2005)")
    yr = BOOKS["year"].dropna().astype(int)
    yr = yr[yr.between(1950, 2005)].value_counts().sort_index()
    fig = go.Figure(go.Scatter(x=yr.index, y=yr.values, mode="lines", line=dict(color=PURPLE, width=2.5),
                               fill="tozeroy", fillcolor="rgba(102,2,60,.18)",
                               hovertemplate="Tahun %{x}<br>%{y:,} buku<extra></extra>"))
    show(style_fig(fig, 320))

    a, b = st.columns(2)
    with a:
        chart_head("Sebaran Apresiasi", "jumlah rating per skor (1–10)")
        d = R["rating"].value_counts().sort_index()
        fig = go.Figure(go.Bar(x=d.index, y=d.values, marker_color=PURPLE, text=[f"{v/1000:.0f}rb" for v in d.values],
                               textposition="outside", textfont=dict(color=INK, size=14),
                               hovertemplate="Rating %{x}<br>%{y:,} rating<extra></extra>"))
        fig.update_xaxes(dtick=1)
        show(style_fig(fig, 340, yaxis_range=[0, d.max() * 1.15]))
    with b:
        chart_head("Rating per Buku", "ekor panjang: mengapa perlu hybrid")
        n = BOOKS["n_ratings"]
        grp = pd.Series({"1 rating": (n == 1).sum(), "2 rating": (n == 2).sum(), "3–9 rating": n.between(3, 9).sum(),
                         "≥ 10 rating": (n >= 10).sum()})
        fig = go.Figure(go.Bar(x=grp.index, y=grp.values, marker_color=[C_RND, "#B9A3AE", C_CB, PURPLE],
                               text=[f"{v:,}<br>({v/grp.sum():.0%})" for v in grp.values], textposition="outside",
                               textfont=dict(color=INK, size=14), hovertemplate="%{x}: %{y:,} buku<extra></extra>"))
        show(style_fig(fig, 340, yaxis_range=[0, grp.max() * 1.25]))
    st.markdown(f'<div class="info-box">Sebagian besar buku hanya dirating sekali, sehingga CF hanya bisa belajar dari '
                f'<b>{META["n_core_books"]:,} buku inti</b>. Content-Based menjangkau <b>{META["n_pool_books"]:,} buku</b> '
                f'(≥ {META["pool_min"]} rating) — itulah alasan model dibuat hybrid.</div>', unsafe_allow_html=True)

    a, b = st.columns(2)
    U = E.users
    with a:
        chart_head("Demografi Usia", f"{U['age'].notna().sum():,} pembaca yang mengisi usia")
        fig = go.Figure(go.Histogram(x=U["age"].dropna(), xbins=dict(start=5, end=100, size=5), marker_color=CITRON,
                                     marker_line=dict(color=PURPLE, width=1.2),
                                     hovertemplate="Usia %{x}<br>%{y:,} pembaca<extra></extra>"))
        show(style_fig(fig, 340))
    with b:
        chart_head("Asal Pembaca", "10 negara teratas")
        c = U["country"].value_counts().drop("Tidak diketahui", errors="ignore").head(10)[::-1]
        fig = go.Figure(go.Bar(x=c.values, y=c.index, orientation="h", marker_color=PURPLE,
                               text=[f"{v:,}" for v in c.values], textposition="outside", textfont=dict(color=INK, size=14),
                               hovertemplate="%{y}: %{x:,} pembaca<extra></extra>"))
        show(style_fig(fig, 340, xaxis_range=[0, c.max() * 1.18]))

    chart_head("Persona Pembaca", "k-means (jumlah rating, rata-rata rating, usia)")
    a, b = st.columns([1, 1.1], vertical_alignment="center")
    pc = U["persona"].dropna().value_counts()
    warna = {"Kutu Buku Aktif": PURPLE, "Pembaca Muda Kasual": CITRON, "Pembaca Senior Kasual": "#A34873"}
    teks = {"Kutu Buku Aktif": CREAM, "Pembaca Muda Kasual": PURPLE, "Pembaca Senior Kasual": CREAM}
    with a:
        fig = go.Figure(go.Pie(labels=pc.index, values=pc.values, hole=.5, sort=False,
                               marker=dict(colors=[warna[k] for k in pc.index], line=dict(color=CREAM, width=2)),
                               textinfo="percent", textfont=dict(size=16, color=[teks[k] for k in pc.index]),
                               hovertemplate="%{label}<br>%{value:,} pembaca (%{percent})<extra></extra>"))
        show(style_fig(fig, 320, showlegend=True, legend=dict(orientation="h", y=-0.08, font=dict(color=INK, size=15))))
    with b:
        prof = (U.dropna(subset=["persona"]).groupby("persona")
                .agg(**{"Jumlah pembaca": ("n_ratings", "size"), "Rata-rata buku dinilai": ("n_ratings", "mean"),
                        "Rata-rata rating": ("avg_rating", "mean"), "Rata-rata usia": ("age", "mean")}).round(1))
        prof.index.name = "Persona"
        table(prof, {"Jumlah pembaca": "{:,.0f}", "Rata-rata buku dinilai": "{:.1f}", "Rata-rata rating": "{:.2f}",
                     "Rata-rata usia": "{:.1f}"})
        st.markdown('<div class="info-box" style="margin-top:8px">Label ditentukan dari profil klaster: klaster dengan '
                    'rating terbanyak = <b>Kutu Buku Aktif</b>; dua klaster lainnya sama-sama jarang merating dan '
                    'hampir sama dalam rata-rata rating, sehingga dibedakan menurut <b>usia</b>.</div>',
                    unsafe_allow_html=True)

# ==========================================
# HALAMAN 5: EVALUASI MODEL
# ==========================================
elif page == "🧪 Evaluasi Model":
    title_block("Seberapa Tepat Rekomendasinya?", "hasil notebook bagian B · 5-fold cross-validation")
    ev, sweep, cold = load_eval()
    if ev is None:
        st.warning("File evaluasi belum ada di folder data/. Jalankan prepare_data.py dengan folder hasil_bagian_B.")
        st.stop()
    ev = ev.set_index("Model")
    hyb = [m for m in ev.index if m.startswith("Hybrid")][0]
    cf = "CF – co-rating (Amazon)"
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "🎯", "Precision@10", f"{ev.at[hyb, 'Precision@10_mean']:.4f}", f"CF saja {ev.at[cf, 'Precision@10_mean']:.4f}")
    kpi(k2, "🔎", "Recall@10", f"{ev.at[hyb, 'Recall@10_mean']:.3f}", f"CF saja {ev.at[cf, 'Recall@10_mean']:.3f}")
    kpi(k3, "✅", "HitRate@10", f"{ev.at[hyb, 'HitRate@10_mean']:.1%}", "pengguna dapat ≥ 1 tebakan tepat")
    kpi(k4, "📈", "Hybrid vs CF", f"+{ev.at[hyb, 'Precision@10_mean'] / ev.at[cf, 'Precision@10_mean'] - 1:.0%}",
        "kenaikan Precision@10")

    st.markdown('<div class="info-box" style="margin-top:22px">Rating tiap pengguna inti dibagi ke 5 lipatan; bergiliran '
                '4 lipatan untuk melatih dan 1 untuk menguji. Rekomendasi dianggap tepat bila buku itu diberi rating '
                '≥ 7 di lipatan uji. Angka absolut kecil karena data sangat jarang (densitas 0,48%) — yang bermakna '
                'adalah <b>perbandingan antar-model</b>.</div>', unsafe_allow_html=True)

    a, b = st.columns(2)
    with a:
        chart_head("Precision@10 per Model", "rata-rata ± simpangan baku 5 lipatan")
        e = ev.sort_values("Precision@10_mean")
        warna = {"Random": C_RND, "Popularitas": C_POP, "CF – adjusted cosine": "#9DB4CC", cf: C_CF,
                 "Content-Based": C_CB, hyb: C_HYB}
        fig = go.Figure(go.Bar(x=e["Precision@10_mean"], y=e.index, orientation="h",
                               marker_color=[warna.get(m, C_RND) for m in e.index],
                               error_x=dict(type="data", array=e["Precision@10_std"], color=INK, thickness=1.3),
                               text=[f"      {v:.4f}" for v in e["Precision@10_mean"]], textposition="outside",
                               textfont=dict(color=INK, size=14), cliponaxis=False,
                               hovertemplate="%{y}<br>Precision@10 = %{x:.4f}<extra></extra>"))
        show(style_fig(fig, 360, xaxis_range=[0, e["Precision@10_mean"].max() * 1.45], margin=dict(l=10, r=30, t=10, b=10)))
    with b:
        chart_head("Pemilihan α", "precision@10 untuk tiap campuran CF/CB")
        if sweep is not None:
            s = sweep
            best = s.loc[s["Precision@10_mean"].idxmax()]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.r_[s.alpha, s.alpha[::-1]],
                                     y=np.r_[s["Precision@10_mean"] + s["Precision@10_std"],
                                             (s["Precision@10_mean"] - s["Precision@10_std"])[::-1]],
                                     fill="toself", fillcolor="rgba(102,2,60,.13)", line=dict(width=0),
                                     hoverinfo="skip", name="± 1 simpangan baku"))
            fig.add_trace(go.Scatter(x=s.alpha, y=s["Precision@10_mean"], mode="lines+markers", name="Precision@10",
                                     line=dict(color=PURPLE, width=2.5), marker=dict(size=7, color=PURPLE),
                                     hovertemplate="α = %{x}<br>Precision@10 = %{y:.4f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[best.alpha], y=[best["Precision@10_mean"]], mode="markers+text",
                                     marker=dict(size=15, color=CITRON, line=dict(color=PURPLE, width=2)),
                                     text=[f"α* = {best.alpha:.1f}"], textposition="top center",
                                     textfont=dict(color=PURPLE, size=16), hoverinfo="skip", showlegend=False))
            fig.update_xaxes(title="α  (0 = murni Content-Based, 1 = murni CF)", dtick=0.1)
            show(style_fig(fig, 360, yaxis_range=[0, s["Precision@10_mean"].max() * 1.25]))

    a, b = st.columns(2)
    with a:
        chart_head("Uji Cold-Start", "pengguna di luar inti · leave-one-out · hitrate@10")
        if cold is not None:
            c = cold.set_index(cold.columns[0])["HitRate@10"]
            fig = go.Figure(go.Bar(x=c.index, y=c.values, marker_color=[C_POP if "Pop" in i else C_CB for i in c.index],
                                   text=[f"{v:.2%}" for v in c.values], textposition="outside",
                                   textfont=dict(color=INK, size=16), width=0.5,
                                   hovertemplate="%{x}<br>HitRate@10 = %{y:.2%}<extra></extra>"))
            fig.update_yaxes(tickformat=".0%")
            show(style_fig(fig, 320, yaxis_range=[0, c.max() * 1.3]))
            st.markdown(f'<div class="info-box">Content-Based menemukan buku tersembunyi pada {c.max():.1%} pengguna '
                        f'(≈ {c.max() / c.min():.1f}× Popularitas) → pengguna di luar inti dialihkan ke Content-Based.</div>',
                        unsafe_allow_html=True)
    with b:
        chart_head("Cakupan Tiap Komponen", "berapa buku & pengguna yang bisa dilayani")
        cak = pd.DataFrame({
            "Komponen": ["Item-Based CF", "Content-Based", "Hybrid (switching)"],
            "Buku kandidat": [f"{META['n_core_books']:,} ({META['n_core_books']/META['n_books']:.1%})",
                              f"{META['n_pool_books']:,} ({META['n_pool_books']/META['n_books']:.1%})",
                              f"{META['n_pool_books']:,} ({META['n_pool_books']/META['n_books']:.1%})"],
            "Pengguna terlayani": [f"{META['n_core_users']:,} ({META['n_core_users']/META['n_users']:.1%})",
                                   f"{META['n_users']:,} (100%)", "100% + pengguna baru"],
        }).set_index("Komponen")
        table(cak)
        tab = ev[["Precision@10_mean", "Recall@10_mean", "HitRate@10_mean", "Diversitas_mean"]].copy()
        tab.columns = ["Precision@10", "Recall@10", "HitRate@10", "Diversitas"]
        tab.index.name = "Model"
        table(tab.sort_values("Precision@10", ascending=False), {c: "{:.4f}" for c in tab.columns})
    st.markdown('<div class="info-box">Catatan jujur: α dipilih dan dilaporkan pada CV yang sama, sehingga angka Hybrid '
                'sedikit optimistis; α = 0,5 dan 0,6 praktis seri. Evaluasi akurasi hanya mewakili pengguna inti '
                f'({META["n_core_users"]/META["n_users"]:.0%} pembaca); untuk sisanya bukti yang ada adalah uji cold-start.</div>',
                unsafe_allow_html=True)
