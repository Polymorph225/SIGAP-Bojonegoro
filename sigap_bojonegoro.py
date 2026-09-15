import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import os
import re
import io
import warnings
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import concurrent.futures

# ── Pustaka berat sengaja TIDAK diimpor di tingkat modul ─────────────────────
# Prophet, XGBoost, statsmodels, dan klien Gemini menelan sekitar 1,6 detik CPU
# dan ratusan MB RAM pada setiap start container, padahal hanya dibutuhkan oleh
# dua dari sepuluh halaman. Di Streamlit Community Cloud yang sumber dayanya
# terbatas, beban itu ikut memicu throttling. Keempatnya kini dimuat saat
# pertama kali benar-benar dipakai (Python menyimpannya di sys.modules, jadi
# panggilan berikutnya tidak membayar lagi).

# ─── Machine Learning & Forecasting Libraries ───────────────────────────────

# XGBoost & scikit-learn

# SARIMA

warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="SIGAP-Bojonegoro – Ensemble AI Forecasting",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# TEMA / CSS
# ============================================================
def inject_custom_css():
    st.markdown("""
    <style>
    /* ── Palet & token kedalaman ──────────────────────────── */
    :root {
        --sigap-blue:      #1e5eff;
        --sigap-blue-deep: #1039a8;
        --sigap-cyan:      #06b6d4;
        --sigap-ink:       #0f172a;
        --sigap-muted:     #56637a;
        --sigap-line:      rgba(148,163,184,.30);
        --sigap-surface:   rgba(255,255,255,.74);

        /* bayangan berlapis: dekat + jauh, supaya terasa mengambang */
        --lift-1: 0 1px 2px rgba(15,23,42,.05), 0 2px 6px rgba(15,23,42,.06);
        --lift-2: 0 2px 4px rgba(15,23,42,.04), 0 10px 24px rgba(30,94,255,.12);
        --lift-3: 0 12px 28px rgba(30,94,255,.18), 0 28px 56px rgba(15,23,42,.12);
    }

    /* ── Latar: gradien lembut + noda cahaya ──────────────── */
    .stApp {
        background:
            radial-gradient(900px 520px at 12% -8%,  rgba(30,94,255,.13), transparent 60%),
            radial-gradient(760px 460px at 92% 4%,   rgba(6,182,212,.12), transparent 62%),
            linear-gradient(180deg, #f5f8ff 0%, #eef3fb 46%, #f7f9fc 100%);
        background-attachment: fixed;
    }
    .block-container { padding: 1.25rem 2rem 3.5rem 2rem; max-width: 1500px; }

    h1, h2, h3 { color: var(--sigap-ink); letter-spacing: -.015em; }
    h1 { font-weight: 800 !important; }
    h2, h3 { font-weight: 700 !important; }

    /* ── Kartu metrik: kaca + terangkat + miring saat disentuh ── */
    [data-testid="metric-container"],
    [data-testid="stMetric"] {
        position: relative;
        padding: 1rem 1.15rem;
        border-radius: 16px;
        background: var(--sigap-surface);
        backdrop-filter: blur(14px) saturate(150%);
        -webkit-backdrop-filter: blur(14px) saturate(150%);
        border: 1px solid var(--sigap-line);
        box-shadow: var(--lift-2);
        transform-style: preserve-3d;
        transition: transform .28s cubic-bezier(.22,1,.36,1), box-shadow .28s ease;
    }
    /* garis cahaya tipis di tepi atas — memberi kesan permukaan miring kena cahaya */
    [data-testid="metric-container"]::before,
    [data-testid="stMetric"]::before {
        content: "";
        position: absolute; inset: 0;
        border-radius: inherit;
        padding: 1px;
        background: linear-gradient(160deg, rgba(255,255,255,.95), rgba(255,255,255,0) 42%);
        -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
        -webkit-mask-composite: xor; mask-composite: exclude;
        pointer-events: none;
    }
    [data-testid="metric-container"]:hover,
    [data-testid="stMetric"]:hover {
        transform: perspective(900px) translateY(-4px) rotateX(4deg);
        box-shadow: var(--lift-3);
    }
    [data-testid="stMetricValue"] {
        font-weight: 800;
        background: linear-gradient(120deg, var(--sigap-blue-deep), var(--sigap-cyan));
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* ── Badge model: pil dengan sedikit tinggi ───────────── */
    .model-badge {
        display: inline-block;
        padding: .26rem .8rem;
        border-radius: 999px;
        font-size: .78rem;
        font-weight: 700;
        margin-right: .4rem;
        box-shadow: var(--lift-1);
        border: 1px solid rgba(255,255,255,.6);
    }
    .badge-prophet  { background:linear-gradient(145deg,#e6efff,#cfe0ff); color:#1746b8; }
    .badge-xgb      { background:linear-gradient(145deg,#e4fbec,#c9f3da); color:#12693c; }
    .badge-sarima   { background:linear-gradient(145deg,#fff5da,#ffe9b4); color:#8a5a06; }
    .badge-ensemble { background:linear-gradient(145deg,#f4e9ff,#e7d6ff); color:#6320a8; }
    .badge-winner   { background:linear-gradient(145deg,#ffe3e3,#ffc9c9); color:#a11a1a; }

    /* ── Kotak akurasi & info ─────────────────────────────── */
    .accuracy-box {
        padding: 1.05rem 1.2rem;
        border-radius: 14px;
        background: linear-gradient(150deg, rgba(30,94,255,.10), rgba(6,182,212,.07));
        border: 1px solid rgba(30,94,255,.22);
        box-shadow: var(--lift-1);
        margin-bottom: 1rem;
    }
    .highlight-estimasi {
        color: var(--sigap-blue-deep) !important;
        font-size: 1.06rem; font-weight: 800; line-height: 1.6;
    }

    /* ── Panel bawaan Streamlit: samakan bahasanya ────────── */
    div[data-testid="stExpander"],
    div[data-testid="stAlert"],
    div[data-testid="stDataFrame"],
    div[data-testid="stTable"] {
        border-radius: 14px !important;
        border: 1px solid var(--sigap-line) !important;
        box-shadow: var(--lift-1);
        overflow: hidden;
    }
    div[data-testid="stExpander"] { background: var(--sigap-surface); }

    /* ── Tab: seperti kartu kecil yang naik saat aktif ────── */
    .stTabs [data-baseweb="tab-list"] { gap: .4rem; border-bottom: none; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px 12px 0 0;
        padding: .55rem 1.05rem;
        background: rgba(255,255,255,.55);
        border: 1px solid var(--sigap-line);
        border-bottom: none;
        transition: transform .2s ease, box-shadow .2s ease, background .2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover { transform: translateY(-2px); }
    .stTabs [aria-selected="true"] {
        background: #fff;
        box-shadow: var(--lift-2);
        transform: translateY(-3px);
    }

    /* ── Tombol: timbul, menekan saat diklik ─────────────── */
    .stButton > button, .stDownloadButton > button, .stLinkButton > a {
        border-radius: 12px;
        font-weight: 650;
        border: 1px solid rgba(30,94,255,.28);
        box-shadow: var(--lift-1);
        transition: transform .16s ease, box-shadow .16s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover, .stLinkButton > a:hover {
        transform: translateY(-2px);
        box-shadow: var(--lift-2);
    }
    .stButton > button:active, .stDownloadButton > button:active {
        transform: translateY(0);
        box-shadow: inset 0 2px 5px rgba(15,23,42,.16);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--sigap-blue), var(--sigap-cyan));
        border: none; color: #fff;
    }

    /* ── Sidebar ─────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(185deg, #ffffff 0%, #f3f7ff 100%);
        border-right: 1px solid var(--sigap-line);
        box-shadow: 6px 0 26px rgba(15,23,42,.06);
    }

    /* ── Grafik Plotly ikut mengambang ───────────────────── */
    div[data-testid="stPlotlyChart"] {
        border-radius: 16px;
        background: rgba(255,255,255,.66);
        border: 1px solid var(--sigap-line);
        box-shadow: var(--lift-2);
        padding: .35rem;
    }

    hr { border-color: var(--sigap-line); }

    /* Hormati pengguna yang mematikan animasi di sistemnya */
    @media (prefers-reduced-motion: reduce) {
        * { transition: none !important; animation: none !important; }
    }
    /* Di layar sempit, efek angkat dimatikan agar tidak mengganggu */
    @media (max-width: 640px) {
        [data-testid="metric-container"]:hover,
        [data-testid="stMetric"]:hover { transform: none; }
        .block-container { padding: 1rem 1rem 2.5rem 1rem; }
    }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ============================================================
# HEADER
# ============================================================
# ============================================================
# HEADER — banner 3D (globe geospasial berputar)
# ============================================================
HERO_HEIGHT = 250

def render_hero_3d():
    """Banner header dengan globe 3D berputar (three.js) sebagai latar.

    Dirender di dalam iframe komponen Streamlit. Kalau three.js gagal dimuat
    (mis. jaringan puskesmas memblokir CDN), banner tetap tampil rapi dengan
    gradien saja — teksnya tidak pernah hilang.
    """
    # st.components.v1.html dijadwalkan dihapus Streamlit; pakai st.iframe bila
    # tersedia, dan tetap jatuh ke API lama pada versi Streamlit yang lebih tua.
    if hasattr(st, "iframe"):
        st.iframe(HERO_HTML, height=HERO_HEIGHT)
    else:
        components.html(HERO_HTML, height=HERO_HEIGHT, scrolling=False)

HERO_HTML = """
<div id="hero">
  <canvas id="globe"></canvas>
  <div id="copy">
    <div id="eyebrow">PEMERINTAH KABUPATEN BOJONEGORO &middot; UPT PUSKESMAS PURWOSARI</div>
    <h1>SIGAP&#8209;Bojonegoro</h1>
    <p>Sistem Informasi Geospasial &amp; Analitik Prediktif &mdash; dashboard analisis data kunjungan puskesmas</p>
  </div>
</div>

<style>
  html, body { margin:0; padding:0; background:transparent; overflow:hidden; }
  #hero {
    position: relative;
    height: 250px;
    border-radius: 20px;
    overflow: hidden;
    background:
      radial-gradient(680px 300px at 82% 48%, rgba(6,182,212,.30), transparent 66%),
      linear-gradient(126deg, #0b2f8f 0%, #1e5eff 46%, #2aa9d9 100%);
    box-shadow: 0 14px 32px rgba(16,57,168,.28), 0 34px 64px rgba(15,23,42,.16);
    font-family: "Source Sans Pro", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  /* kilau tipis di tepi atas, biar permukaannya terasa melengkung */
  #hero::after {
    content:""; position:absolute; inset:0; border-radius:inherit; pointer-events:none;
    background: linear-gradient(168deg, rgba(255,255,255,.30), rgba(255,255,255,0) 38%);
  }
  #globe { position:absolute; inset:0; width:100%; height:100%; display:block; }
  #copy {
    position: relative; z-index: 2;
    padding: 2.1rem 2.3rem;
    max-width: 58%;
    color: #fff;
  }
  #eyebrow {
    font-size: .68rem; font-weight: 700; letter-spacing: .14em;
    color: rgba(255,255,255,.80); margin-bottom: .55rem;
  }
  #copy h1 {
    margin: 0 0 .45rem 0;
    color: #fff;
    font-size: clamp(1.7rem, 3.6vw, 2.6rem);
    font-weight: 800; letter-spacing: -.02em; line-height: 1.08;
    text-shadow: 0 2px 18px rgba(4,22,74,.45);
  }
  #copy p {
    margin: 0; font-size: .95rem; line-height: 1.5;
    color: rgba(255,255,255,.90); max-width: 30rem;
    text-shadow: 0 1px 10px rgba(4,22,74,.35);
  }
  @media (max-width: 720px) {
    #copy { max-width: 100%; padding: 1.5rem 1.5rem; }
    #copy p { font-size: .86rem; }
  }
</style>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(function () {
  var cv = document.getElementById('globe');
  // three.js tidak termuat (CDN diblokir) -> banner tetap tampil, hanya tanpa globe
  if (typeof THREE === 'undefined' || !cv) { if (cv) cv.style.display = 'none'; return; }

  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  var scene  = new THREE.Scene();
  var camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
  camera.position.z = 5.2;

  var renderer = new THREE.WebGLRenderer({ canvas: cv, alpha: true, antialias: true });
  renderer.setClearColor(0x000000, 0);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  var world = new THREE.Group();
  // digeser ke kanan supaya tidak menabrak teks
  world.position.x = 1.55;
  scene.add(world);

  var R = 1.62;

  // rangka bola
  var wire = new THREE.Mesh(
    new THREE.IcosahedronGeometry(R, 2),
    new THREE.MeshBasicMaterial({ color: 0xbfe4ff, wireframe: true, transparent: true, opacity: 0.44 })
  );
  world.add(wire);

  // bola inti semu, memberi kesan padat
  var core = new THREE.Mesh(
    new THREE.SphereGeometry(R * 0.985, 32, 32),
    new THREE.MeshBasicMaterial({ color: 0x0a2a7a, transparent: true, opacity: 0.42 })
  );
  world.add(core);

  // titik-titik "desa" tersebar di permukaan (distribusi spiral Fibonacci)
  var N = 170, pos = new Float32Array(N * 3), gold = Math.PI * (3 - Math.sqrt(5));
  for (var i = 0; i < N; i++) {
    var y = 1 - (i / (N - 1)) * 2;
    var r = Math.sqrt(Math.max(0, 1 - y * y));
    var th = gold * i;
    pos[i*3]   = Math.cos(th) * r * R * 1.012;
    pos[i*3+1] = y * R * 1.012;
    pos[i*3+2] = Math.sin(th) * r * R * 1.012;
  }
  var pg = new THREE.BufferGeometry();
  pg.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  world.add(new THREE.Points(pg, new THREE.PointsMaterial({
    color: 0xaef6ff, size: 0.056, transparent: true, opacity: 1.0, sizeAttenuation: true
  })));

  // dua cincin orbit miring
  [[0.62, 2.18], [-0.45, 2.52]].forEach(function (o) {
    var ring = new THREE.Mesh(
      new THREE.TorusGeometry(o[1], 0.006, 8, 128),
      new THREE.MeshBasicMaterial({ color: 0x9fe9ff, transparent: true, opacity: 0.42 })
    );
    ring.rotation.x = Math.PI / 2 + o[0];
    ring.rotation.y = o[0] * 0.5;
    world.add(ring);
  });

  function resize() {
    var w = cv.clientWidth, h = cv.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    // di layar sempit globe dikecilkan & digeser supaya teks tetap terbaca
    var narrow = w < 720;
    world.position.x = narrow ? 0.85 : 1.55;
    world.scale.setScalar(narrow ? 0.72 : 1);
    camera.updateProjectionMatrix();
  }
  resize();
  window.addEventListener('resize', resize);

  // parallax halus mengikuti kursor
  var tx = 0, ty = 0;
  document.addEventListener('mousemove', function (e) {
    tx = (e.clientX / window.innerWidth  - 0.5) * 0.34;
    ty = (e.clientY / window.innerHeight - 0.5) * 0.22;
  });

  var visible = true;
  document.addEventListener('visibilitychange', function () { visible = !document.hidden; });

  var t = 0;
  function frame() {
    requestAnimationFrame(frame);
    if (!visible) return;                 // berhenti menggambar saat tab tidak aktif
    t += reduce ? 0 : 0.0024;             // hormati setelan "kurangi animasi"
    world.rotation.y = t * 2.6;
    world.rotation.x = -0.24 + Math.sin(t * 1.7) * 0.05;
    world.rotation.y += (tx - world.rotation.y % (Math.PI * 2)) * 0;
    camera.position.x = tx;
    camera.position.y = -ty;
    camera.lookAt(world.position.x * 0.45, 0, 0);
    renderer.render(scene, camera);
  }
  frame();
})();
</script>
"""

render_hero_3d()
col_header1, col_header2 = st.columns([2, 1])

with col_header1:
    st.markdown("""
    Dashboard ini membantu menganalisis **data kunjungan Puskesmas** dengan:
    - Filter interaktif: Poli, Diagnosa, Umur, Desa, Pembiayaan
    - **Ensemble Forecasting** & **Disease Dominance Detection**
    - **Evaluasi Akurasi** model (MAE, RMSE, MAPE)
    """)

with col_header2:
    st.info("💡 **Apabila Data Tidak Dapat Diunggah, Bersihkan Dulu Disini!**")
    st.link_button(
        "✨ Buka Data Cleaning App", 
        "https://data-cleaning-app-for-puskesmasapp.streamlit.app/",
        use_container_width=True,
        help="Klik untuk membersihkan format data sebelum diupload ke dashboard ini."
    )

st.markdown("---")
st.markdown("⬅️ **Mulai dengan meng-upload file data di sidebar.**")

# ============================================================
# GEMINI CLIENT
# ============================================================
@st.cache_resource
def get_gemini_client():
    import google.generativeai as genai
    api_key = None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        st.warning("⚠️ API key Gemini belum diset. Fitur AI Agent tidak aktif.")
        return False
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Gagal inisialisasi Gemini: {e}")
        return False

# ============================================================
# HELPER UMUM
# ============================================================
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
    return output.getvalue()


TARGET_COLS = ["tanggal_kunjungan","no_rm","umur","jenis_kelamin","poli","diagnosa","pembiayaan","desa"]

def _normalize_col_name(col):
    col = str(col).strip().lower()
    return re.sub(r"[^a-z0-9]", "", col)

def _build_column_mapping(df_raw):
    mapping = {}
    alias_dict = {
        "tanggal_kunjungan": ["tanggalkunjungan","tglkunjungan","tanggal","tgl","visitdate"],
        "no_rm":    ["norm","norekammedis","norekam_medis","no_rekam_medis","rekammedis"],
        "umur":     ["umur","usia","age","umurth"],
        "jenis_kelamin": ["jeniskelamin","jk","sex","kelamin","llp"],
        "poli":     ["poli","politujuan","unit","unitlayanan","poliklinik"],
        "diagnosa": ["diagnosa","diagnosautama","dxutama","diag","icd10","diagnosis"],
        "pembiayaan":["pembiayaan","carabayar","penjamin","jaminan","pembayar"],
        "desa":     ["desa","alamatdesa","kelurahan","desakelurahan","namadesa"],
    }
    norm_cols = {col: _normalize_col_name(col) for col in df_raw.columns}
    for std_name, aliases in alias_dict.items():
        for raw_col, norm_key in norm_cols.items():
            if norm_key in aliases:
                mapping[raw_col] = std_name
                break
    return mapping

def _clean_jk(val):
    if pd.isna(val): return np.nan
    v = str(val).strip().lower()
    if v in ["l","lk","laki","laki-laki","male","m","1"]: return "Laki-Laki"
    if v in ["p","pr","perempuan","wanita","female","f","2"]: return "Perempuan"
    return str(val).title()

def _parse_umur(val):
    if pd.isna(val): return np.nan
    m = re.search(r"(\d+)", str(val))
    if m:
        try: return int(m.group(1))
        except: return np.nan
    try: return int(float(str(val)))
    except: return np.nan

def clean_raw_data(df_raw):
    if df_raw is None or df_raw.empty: return df_raw
    col_map = _build_column_mapping(df_raw)
    df = df_raw.rename(columns=col_map).copy()
    keep = [c for c in TARGET_COLS if c in df.columns]
    df = df[keep].copy()
    if "tanggal_kunjungan" in df.columns:
        df["tanggal_kunjungan"] = pd.to_datetime(df["tanggal_kunjungan"], errors="coerce")
    if "umur" in df.columns:
        df["umur"] = df["umur"].apply(_parse_umur)
    if "jenis_kelamin" in df.columns:
        df["jenis_kelamin"] = df["jenis_kelamin"].apply(_clean_jk)
    for col in ["poli","diagnosa","pembiayaan","desa","no_rm"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan})
    return df

@st.cache_data
def load_data(file):
    ext = file.name.lower().split(".")[-1]
    if ext == "csv":
        df_raw = pd.read_csv(file)
    else:
        try:    df_raw = pd.read_excel(file, engine="openpyxl")
        except: df_raw = pd.read_excel(file, engine="xlrd")
    return clean_raw_data(df_raw), df_raw

MUSIM_URUT = ["🌧️ Hujan (Des–Feb)", "🌤️ Pancaroba I (Mar–Mei)",
              "☀️ Kemarau (Jun–Agu)", "🌦️ Pancaroba II (Sep–Nov)"]

def label_musim(bulan):
    """Golongkan bulan ke empat musim, termasuk dua masa pancaroba.

    Pancaroba II (Sep-Nov) adalah peralihan kemarau ke hujan — masa yang pada data
    kunjungan Puskesmas Purwosari justru menunjukkan puncak ISPA. Dengan pembagian
    dua musim saja, puncak itu terbelah di perbatasan dan polanya hilang.
    """
    try:
        b = int(bulan)
    except (TypeError, ValueError):
        return None
    if b in (12, 1, 2):  return MUSIM_URUT[0]
    if b in (3, 4, 5):   return MUSIM_URUT[1]
    if b in (6, 7, 8):   return MUSIM_URUT[2]
    if b in (9, 10, 11): return MUSIM_URUT[3]
    return None


@st.cache_data(show_spinner=False)
def load_and_prepare(file):
    """Baca berkas lalu siapkan datanya, sekali saja per berkas.

    PERF: preprocess_data sebelumnya dipanggil di luar cache, sehingga seluruh
    penyiapan data diulang pada SETIAP rerun Streamlit — dan Streamlit rerun
    tiap kali filter disentuh atau halaman diganti. Pada 23.616 baris data nyata
    biayanya terukur ~230 ms per interaksi, yang pada CPU terbatas Streamlit
    Community Cloud menumpuk menjadi pemakaian berkelanjutan dan memicu throttle.
    """
    df_clean, _ = load_data(file)
    return preprocess_data(df_clean)


def preprocess_data(df):
    if df is None or df.empty: return df
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "umur" in df.columns:
        df["umur"] = pd.to_numeric(df["umur"], errors="coerce")
        bins   = [0,1,5,15,25,45,60,200]
        labels = ["<1 th","1-4 th","5-14 th","15-24 th","25-44 th","45-59 th","60+ th"]
        df["kelompok_umur"] = pd.cut(df["umur"], bins=bins, labels=labels, right=False)
    if "tanggal_kunjungan" in df.columns:
        # ── FIX: pastikan tipe datetime dan buang baris dengan tanggal tidak valid ──
        df["tanggal_kunjungan"] = pd.to_datetime(df["tanggal_kunjungan"], errors="coerce")
        df = df.dropna(subset=["tanggal_kunjungan"]).reset_index(drop=True)

        df["tahun"]      = df["tanggal_kunjungan"].dt.year
        df["bulan"]      = df["tanggal_kunjungan"].dt.month
        df["nama_bulan"] = df["tanggal_kunjungan"].dt.strftime("%b")

        # ── FIX: isocalendar().week mengembalikan UInt32 nullable → gunakan Int64
        #         agar tidak crash saat ada nilai NA (seharusnya sudah tidak ada
        #         setelah dropna di atas, tapi ini sebagai lapisan keamanan ekstra)
        df["minggu"]  = (
            df["tanggal_kunjungan"]
            .dt.isocalendar()
            .week
            .astype("Int64")   # nullable integer, aman terhadap NA
            .astype(int)       # konversi akhir ke int biasa (aman karena NA sudah hilang)
        )

        df["hari_ke"]    = df["tanggal_kunjungan"].dt.dayofyear
        # Musim hujan Indonesia: Nov-Apr → 1, kemarau: Mei-Okt → 0
        df["musim_hujan"] = df["bulan"].apply(lambda m: 1 if m in [11,12,1,2,3,4] else 0)
        # Penggolongan empat musim. Pembagian dua-musim saja menyesatkan untuk
        # penyakit yang memuncak di masa peralihan: pada data nyata Puskesmas
        # Purwosari, ISPA memuncak Agustus-Oktober (tepat di perbatasan), sehingga
        # agregat dua-musim justru menyimpulkan sebaliknya.
        df["musim"] = df["bulan"].apply(label_musim)
    for col in ["poli","jenis_kelamin","pembiayaan","diagnosa","desa"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
    return df

# ============================================================
# SIDEBAR + FILTER
# ============================================================
def apply_filters(_):
    with st.sidebar:
        st.markdown("## 🏥 Data Kunjungan")
        uploaded_file = st.file_uploader("Upload CSV / Excel", type=["csv","xlsx","xls"])
        if uploaded_file is None:
            st.info("Silakan upload file CSV/Excel.")
            return None, None
        df = load_and_prepare(uploaded_file)
        st.success("Data berhasil dimuat ✅")
        st.caption(f"📊 {len(df):,} baris · {len(df.columns)} kolom".replace(",","."))

        st.markdown("### 🔍 Filter Data")
        date_range = poli_pilihan = jk_pilihan = bayar_pilihan = None
        kelompok_umur_pilihan = desa_pilihan = kecuali_penyakit = None

        with st.expander("Waktu & Poli", expanded=True):
            if "tanggal_kunjungan" in df.columns:
                mn, mx = df["tanggal_kunjungan"].min(), df["tanggal_kunjungan"].max()
                if pd.notna(mn) and pd.notna(mx):
                    date_range = st.date_input("Rentang Tanggal", value=[mn.date(), mx.date()])
            if "poli" in df.columns:
                poli_pilihan = st.multiselect("Poli", sorted(df["poli"].dropna().unique()))

        with st.expander("Filter Lainnya", expanded=False):
            if "jenis_kelamin" in df.columns:
                jk_pilihan = st.multiselect("Jenis Kelamin", sorted(df["jenis_kelamin"].dropna().unique()))
            if "kelompok_umur" in df.columns:
                kelompok_umur_pilihan = st.multiselect("Kelompok Umur", df["kelompok_umur"].dropna().unique())
            if "desa" in df.columns:
                desa_pilihan = st.multiselect("Desa", sorted(df["desa"].dropna().unique()))
            if "pembiayaan" in df.columns:
                bayar_pilihan = st.multiselect("Pembiayaan", sorted(df["pembiayaan"].dropna().unique()))
            if "diagnosa" in df.columns:
                kecuali_penyakit = st.multiselect(
                    "❌ Kecualikan Penyakit",
                    sorted(df["diagnosa"].dropna().unique()),
                    help="Penyakit ini tidak diikutkan dalam analisis."
                )

        df_f = df.copy()
        if date_range and len(date_range)==2:
            df_f = df_f[(df_f["tanggal_kunjungan"].dt.date >= date_range[0]) &
                        (df_f["tanggal_kunjungan"].dt.date <= date_range[1])]
        if poli_pilihan:           df_f = df_f[df_f["poli"].isin(poli_pilihan)]
        if jk_pilihan:             df_f = df_f[df_f["jenis_kelamin"].isin(jk_pilihan)]
        if bayar_pilihan:          df_f = df_f[df_f["pembiayaan"].isin(bayar_pilihan)]
        if kelompok_umur_pilihan:  df_f = df_f[df_f["kelompok_umur"].isin(kelompok_umur_pilihan)]
        if desa_pilihan:           df_f = df_f[df_f["desa"].isin(desa_pilihan)]
        if kecuali_penyakit:       df_f = df_f[~df_f["diagnosa"].isin(kecuali_penyakit)]

        return df_f, {
            "poli": poli_pilihan, "jenis_kelamin": jk_pilihan,
            "pembiayaan": bayar_pilihan, "kelompok_umur": kelompok_umur_pilihan,
            "desa": desa_pilihan, "penyakit_dikecualikan": kecuali_penyakit,
        }

def show_active_filters(fi):
    if not fi: return
    chips = [f"{k.replace('_',' ').title()}: {', '.join(map(str,v))}" for k,v in fi.items() if v]
    if chips: st.caption("🎯 **Filter aktif:** " + " | ".join(chips))

# ============================================================
# ═══════════  ENSEMBLE FORECASTING ENGINE  ═══════════════════
# ============================================================

# ── Feature Engineering ─────────────────────────────────────
def build_features(weekly: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan fitur temporal & lag ke data mingguan."""
    df = weekly.copy().sort_values("ds").reset_index(drop=True)
    # ── FIX: isocalendar().week → Int64 dulu sebelum astype(int) ──
    df["week_of_year"] = (
        df["ds"].dt.isocalendar().week.astype("Int64").astype(int)
    )
    df["month"]         = df["ds"].dt.month
    df["quarter"]       = df["ds"].dt.quarter
    df["year"]          = df["ds"].dt.year
    df["musim_hujan"]   = df["month"].apply(lambda m: 1 if m in [11,12,1,2,3,4] else 0)
    df["is_ramadan"]    = 0  # placeholder – bisa diisi manual
    # Lag features
    for lag in [1, 2, 3, 4]:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    # Rolling mean
    df["roll_mean_4"]  = df["y"].shift(1).rolling(4).mean()
    df["roll_mean_8"]  = df["y"].shift(1).rolling(8).mean()
    df["roll_std_4"]   = df["y"].shift(1).rolling(4).std()
    return df.dropna().reset_index(drop=True)


# ── Metrik Evaluasi ──────────────────────────────────────────
def eval_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask   = y_true != 0
    # Dihitung langsung dengan numpy (hasil identik dengan sklearn) agar
    # scikit-learn tidak perlu ikut dimuat saat aplikasi start.
    mae    = float(np.mean(np.abs(y_true - y_pred)))
    rmse   = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape   = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE": round(mape, 2)}


# ── Prophet ──────────────────────────────────────────────────
def run_prophet(train_df, periods, freq="W-MON"):
    """Fit Prophet dan kembalikan forecast + metrics."""
    from prophet import Prophet
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.80,
        changepoint_prior_scale=0.05,
    )
    # Tambah regressor musim hujan
    train_ext = train_df.copy()
    train_ext["musim_hujan"] = train_ext["ds"].dt.month.apply(lambda mo: 1 if mo in [11,12,1,2,3,4] else 0)
    m.add_regressor("musim_hujan")
    m.fit(train_ext)

    future = m.make_future_dataframe(periods=periods, freq=freq)
    future["musim_hujan"] = future["ds"].dt.month.apply(lambda mo: 1 if mo in [11,12,1,2,3,4] else 0)
    fc = m.predict(future)

    # Hitung metrik pada data training
    merged = train_ext.merge(fc[["ds","yhat"]], on="ds")
    metrics = eval_metrics(merged["y"], merged["yhat"])
    return fc, metrics, m


# ── XGBoost ──────────────────────────────────────────────────
def run_xgboost(train_df, periods, freq="W-MON"):
    """Fit XGBoost dengan feature engineering dan prediksi iteratif."""
    import xgboost as xgb
    feat_df = build_features(train_df)
    if len(feat_df) < 10:
        return None, None

    feat_cols = ["week_of_year","month","quarter","year","musim_hujan",
                 "lag_1","lag_2","lag_3","lag_4",
                 "roll_mean_4","roll_mean_8","roll_std_4"]
    X_train = feat_df[feat_cols]
    y_train = feat_df["y"]

    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    # In-sample metrics
    y_pred_train = model.predict(X_train)
    metrics = eval_metrics(y_train, y_pred_train)

    # Iterative future prediction — mengikuti frekuensi (mingguan/bulanan) yang dipilih user
    # PERF: bangun baris fitur sebagai array numpy (bukan pd.DataFrame per-iterasi) —
    # menghindari overhead konstruksi DataFrame di setiap langkah forecasting iteratif.
    last_date    = train_df["ds"].max()
    future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
    history_y    = list(train_df["y"].values)
    future_rows  = []
    for next_date in future_dates:
        row = {
            # ── FIX: isocalendar() mengembalikan named tuple; ambil elemen ke-2 (week) ──
            "week_of_year": int(next_date.isocalendar()[1]),
            "month":        next_date.month,
            "quarter":      (next_date.month - 1) // 3 + 1,
            "year":         next_date.year,
            "musim_hujan":  1 if next_date.month in [11,12,1,2,3,4] else 0,
            "lag_1": history_y[-1],
            "lag_2": history_y[-2] if len(history_y) >= 2 else history_y[-1],
            "lag_3": history_y[-3] if len(history_y) >= 3 else history_y[-1],
            "lag_4": history_y[-4] if len(history_y) >= 4 else history_y[-1],
            "roll_mean_4": np.mean(history_y[-4:]),
            "roll_mean_8": np.mean(history_y[-8:]),
            "roll_std_4":  np.std(history_y[-4:]),
        }
        X_row  = np.array([[row[c] for c in feat_cols]], dtype=float)
        y_next = float(model.predict(X_row)[0])
        history_y.append(max(0, y_next))
        future_rows.append({"ds": next_date, "yhat": max(0, y_next)})

    # Gabung dengan in-sample predictions
    train_preds = pd.DataFrame({"ds": feat_df["ds"], "yhat": y_pred_train.clip(min=0)})
    future_df   = pd.DataFrame(future_rows)
    all_preds   = pd.concat([train_preds, future_df], ignore_index=True)
    return all_preds, metrics


# ── SARIMA ───────────────────────────────────────────────────
def run_sarima(train_df, periods, freq="W-MON"):
    """Fit SARIMA dan prediksi, mengikuti frekuensi (mingguan/bulanan) yang dipilih user.
    Periode musiman: 52 untuk mingguan, 12 untuk bulanan.

    PERF: seasonal_order dengan periode 52 (mingguan) sangat berat untuk statsmodels
    jika memakai inisialisasi exact/diffuse default — waktu fit naik jauh lebih cepat
    dari linear seiring bertambahnya histori data (terukur ~5-6 detik pada 200-260
    titik mingguan). `simple_differencing=True` + `concentrate_scale=True` memangkas
    ini sampai ~15-20x lebih cepat (jadi <0.5 detik pada rentang data yang sama) tanpa
    mengubah hasil peramalan secara signifikan, sehingga maxiter juga bisa diturunkan
    karena optimizer konvergen lebih cepat."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    seasonal_period = 52 if freq == "W-MON" else 12
    ts = train_df.set_index("ds")["y"].asfreq(freq).ffill()
    try:
        model = SARIMAX(
            ts, order=(1,1,1), seasonal_order=(1,1,0,seasonal_period),
            enforce_stationarity=False, enforce_invertibility=False,
            simple_differencing=True, concentrate_scale=True,
        )
        fit   = model.fit(disp=False, maxiter=100)

        # In-sample
        y_pred_train = fit.fittedvalues.clip(lower=0)
        # FIX: simple_differencing=True membuat statsmodels membuang d + D*s observasi
        # pertama (53 titik pada mingguan musiman-52, 13 pada bulanan musiman-12),
        # sehingga fittedvalues lebih pendek daripada ts. Sebelumnya eval_metrics
        # dipanggil dengan dua larik berbeda panjang -> exception -> SARIMA gugur
        # diam-diam dan tidak pernah masuk ensemble. Samakan dulu panjangnya.
        ts_eval = ts.iloc[len(ts) - len(y_pred_train):]
        metrics = eval_metrics(ts_eval.values, y_pred_train.values)

        # Future
        fc_obj    = fit.get_forecast(steps=periods)
        fc_mean   = fc_obj.predicted_mean.clip(lower=0)
        fc_ci     = fc_obj.conf_int(alpha=0.2)

        # Build output dataframe
        train_preds = pd.DataFrame({"ds": ts_eval.index, "yhat": y_pred_train.values})
        future_df   = pd.DataFrame({
            "ds":         fc_mean.index,
            "yhat":       fc_mean.values,
            "yhat_lower": fc_ci.iloc[:,0].clip(lower=0).values,
            "yhat_upper": fc_ci.iloc[:,1].values,
        })
        all_preds = pd.concat([train_preds, future_df], ignore_index=True)
        return all_preds, metrics
    except Exception as e:
        return None, None


# ── Ensemble Auto-Selection ───────────────────────────────────
def _safe_run(fn, *args, **kwargs):
    """Jalankan fungsi model di worker thread; tangkap error di sini (bukan lewat
    panggilan st.* di dalam thread, yang tidak aman dipanggil dari luar main thread)."""
    try:
        return fn(*args, **kwargs), None
    except Exception as e:
        return None, str(e)


def _holdout_split(train_df, min_latih=24):
    """Bagi deret jadi bagian latih dan bagian uji yang disisihkan.

    Bagian uji diambil dari ekor deret (data terbaru) sebanyak 20% panjang deret,
    dibatasi 4 sampai 26 periode. Mengembalikan (latih, uji, h) atau (None, None, 0)
    bila deret terlalu pendek untuk disisihkan.
    """
    n = len(train_df)
    h = int(round(n * 0.2))
    h = max(4, min(26, h))
    if n - h < min_latih:
        return None, None, 0
    return train_df.iloc[:n - h].copy(), train_df.iloc[n - h:].copy(), h


def _eval_holdout(fn, train_df, freq):
    """Nilai sebuah model pada data yang BELUM pernah dilihatnya.

    PENTING: sebelumnya metrik MAE/RMSE/MAPE dihitung in-sample — model dinilai
    memakai data yang dipakai melatihnya sendiri. Untuk model berkapasitas tinggi
    seperti XGBoost hasilnya sangat optimistis (terukur MAPE 2,3% in-sample lawan
    48,8% out-of-sample pada deret yang sama), sehingga lencana "Terbaik" cenderung
    selalu jatuh ke model yang paling pandai menghafal, bukan yang paling tepat
    meramal. Fungsi ini melatih model hanya pada bagian awal deret lalu menilainya
    pada ekor deret yang disisihkan.

    Mengembalikan (metrics, catatan) — catatan berisi keterangan bila deret terlalu
    pendek sehingga evaluasi terpaksa jatuh kembali ke in-sample.
    """
    latih, uji, h = _holdout_split(train_df)
    if latih is None:
        return None, "deret terlalu pendek untuk disisihkan"
    try:
        out = fn(latih, h, freq)
    except Exception:
        return None, "gagal dilatih pada bagian uji"
    fc = out[0] if isinstance(out, tuple) else out
    if fc is None or "yhat" not in getattr(fc, "columns", []):
        return None, "tidak menghasilkan prediksi"
    batas = latih["ds"].max()
    masa_depan = fc[fc["ds"] > batas][["ds", "yhat"]]
    gabung = uji.merge(masa_depan, on="ds", how="inner")
    if len(gabung) == 0:
        return None, "tanggal prediksi tidak bertemu data uji"
    return eval_metrics(gabung["y"].values, gabung["yhat"].values), ""


# Cache: menghindari re-training 3 model dari nol setiap kali skrip Streamlit
# rerun (mis. widget lain disentuh, tombol download diklik) selama data,
# periods, dan freq-nya sama persis dengan run sebelumnya.
@st.cache_data(show_spinner=False, ttl=3600)
def ensemble_forecast(train_df: pd.DataFrame, periods: int, freq: str = "W-MON"):
    """
    Jalankan ketiga model, evaluasi MAPE, buat weighted ensemble,
    dan pilih model terbaik otomatis.
    `freq` diteruskan ke semua model agar tanggal prediksi konsisten
    dengan frekuensi (mingguan/bulanan) yang dipilih user, sehingga
    prediksi benar-benar mencapai tanggal target yang diminta.

    PERF: ketiga model dilatih PARALEL (ThreadPoolExecutor) — sebelumnya berjalan
    berurutan sehingga total waktu = waktu Prophet + XGBoost + SARIMA. Dengan paralel,
    total waktu ≈ waktu model paling lambat saja. Hasil juga di-cache (`st.cache_data`)
    berdasarkan data+periods+freq, jadi klik ulang dengan parameter yang sama tidak
    melatih ulang dari nol.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            "Prophet": executor.submit(_safe_run, run_prophet, train_df, periods, freq),
            "XGBoost": executor.submit(_safe_run, run_xgboost, train_df, periods, freq),
            "SARIMA":  executor.submit(_safe_run, run_sarima, train_df, periods, freq),
        }
        raw = {name: fut.result() for name, fut in futures.items()}

    results = {}
    errors  = {}

    p_out, p_err = raw["Prophet"]
    if p_err:
        errors["Prophet"] = p_err
    elif p_out is not None:
        fc_p, met_p, _ = p_out
        results["Prophet"] = {"fc": fc_p, "metrics": met_p}

    x_out, x_err = raw["XGBoost"]
    if x_err:
        errors["XGBoost"] = x_err
    elif x_out is not None and x_out[0] is not None:
        fc_x, met_x = x_out
        results["XGBoost"] = {"fc": fc_x, "metrics": met_x}

    s_out, s_err = raw["SARIMA"]
    if s_err:
        errors["SARIMA"] = s_err
    elif s_out is not None and s_out[0] is not None:
        fc_s, met_s = s_out
        results["SARIMA"] = {"fc": fc_s, "metrics": met_s}

    # ── Nilai ulang setiap model secara out-of-sample ────────────────────────
    # Metrik dari run_* di atas bersifat in-sample. Yang dilaporkan ke pengguna
    # harus metrik pada data yang disisihkan, agar perbandingan antar model adil.
    if results:
        _fns = {"Prophet": run_prophet, "XGBoost": run_xgboost, "SARIMA": run_sarima}
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            _fut = {nm: ex.submit(_eval_holdout, _fns[nm], train_df, freq)
                    for nm in results if nm in _fns}
            for nm, ft in _fut.items():
                try:
                    met_oos, catatan = ft.result()
                except Exception:
                    met_oos, catatan = None, "gagal dievaluasi"
                if met_oos is not None:
                    results[nm]["metrics"] = met_oos
                    results[nm]["mode"] = "uji"          # diuji pada data disisihkan
                else:
                    results[nm]["mode"] = "insample"     # terpaksa jatuh ke in-sample
                    results[nm]["catatan"] = catatan

    if not results:
        return None, None, None, None, errors

    # Tentukan model terbaik berdasarkan MAPE (atau RMSE jika MAPE nan)
    def score(m):
        mape = m["metrics"].get("MAPE", np.nan) if m["metrics"] else np.nan
        rmse = m["metrics"].get("RMSE", np.nan) if m["metrics"] else np.nan
        return mape if not np.isnan(mape) else rmse

    # FIX: skor di-filter dulu (bukan langsung min() atas seluruh results) —
    # min() dengan nilai NaN di dalamnya berperilaku tidak terdefinisi/salah pilih.
    # FIX: cek "sc is not None" (bukan "if sc") — MAPE=0 (model sempurna) sebelumnya
    # ikut dianggap falsy oleh Python dan malah dibuang dari kandidat/ensemble.
    valid_scores = {
        name: score(r) for name, r in results.items()
    }
    valid_scores = {
        name: sc for name, sc in valid_scores.items()
        if sc is not None and not np.isnan(sc)
    }

    if not valid_scores:
        # Semua skor NaN (kasus langka) — pakai model pertama yang berhasil sbg fallback.
        best_name = next(iter(results))
        return results, best_name, results[best_name]["fc"], None, errors

    best_name = min(valid_scores, key=valid_scores.get)
    best_fc   = results[best_name]["fc"]

    # Weighted ensemble untuk future predictions.
    # FIX: dipasangkan berdasarkan NAMA model (dict), bukan urutan index dua list
    # terpisah (`weights` & `future_dfs`) seperti sebelumnya — versi lama bisa membuat
    # bobot "tertukar" ke model yang salah kalau salah satu model tidak punya kolom
    # "yhat" padahal skornya valid.
    weight_map = {name: 1.0 / (sc + 1e-6) for name, sc in valid_scores.items()}
    fc_map = {}
    for name in valid_scores:
        fc = results[name]["fc"]
        if "yhat" in fc.columns:
            fc_map[name] = fc[["ds", "yhat"]].rename(columns={"yhat": f"yhat_{name}"})

    ensemble_fc = None
    if len(fc_map) > 1:
        names  = list(fc_map.keys())
        merged = fc_map[names[0]]
        for nm in names[1:]:
            merged = merged.merge(fc_map[nm], on="ds", how="inner")
        total_w = sum(weight_map[nm] for nm in names)
        merged["yhat_ensemble"] = sum(
            merged[f"yhat_{nm}"] * weight_map[nm] for nm in names
        ) / total_w
        ensemble_fc = merged

    return results, best_name, best_fc, ensemble_fc, errors


# ══════════════════════════════════════════════════════════════
# HALAMAN UTAMA: ENSEMBLE FORECASTING (page_ml_upgraded)
# ══════════════════════════════════════════════════════════════

def page_ml_upgraded(df_filtered, filter_info):
    st.subheader("🤖 Ensemble AI Forecasting")
    show_active_filters(filter_info)

    if df_filtered is None or len(df_filtered) == 0:
        st.warning("⚠️ Data kosong.")
        return
    if "tanggal_kunjungan" not in df_filtered.columns:
        st.error("❌ Kolom 'tanggal_kunjungan' tidak ditemukan.")
        return

    # ── Penjelasan arsitektur ──────────────────────────────────
    with st.expander("ℹ️ Tentang Arsitektur Ensemble Model", expanded=False):
        st.markdown("""
        | Model | Keunggulan | Kelemahan |
        |---|---|---|
        | **Prophet** | Tren + seasonality otomatis, robust terhadap missing data | Tidak memodelkan fitur eksogen secara eksplisit |
        | **XGBoost** | Feature engineering fleksibel (lag, musim, rolling) | Butuh data cukup untuk konvergensi |
        | **SARIMA** | Kuat untuk pola autoregresif musiman | Sensitif terhadap stasioneritas, lambat untuk data panjang |
        | **Ensemble** | Gabungan bobot berdasarkan MAPE terbaik → akurasi tertinggi | Lebih kompleks, waktu training lebih lama |

        **Auto-Selection:** Model dengan MAPE terendah dipilih sebagai *best single model*.
        **Weighted Ensemble:** Bobot proporsional terhadap 1/MAPE masing-masing model.
        """)

    df_ml = df_filtered.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_ml["tanggal_kunjungan"]):
        df_ml["tanggal_kunjungan"] = pd.to_datetime(df_ml["tanggal_kunjungan"], errors="coerce")
    df_ml = df_ml.dropna(subset=["tanggal_kunjungan"])
    if len(df_ml) == 0:
        st.warning("⚠️ Data tanggal tidak valid.")
        return

    max_date = df_ml["tanggal_kunjungan"].max().date()

    st.markdown("---")
    col1, col2, col3, col4 = st.columns([1.5, 1.5, 1.5, 1])
    with col1:
        fokus = st.radio("Analisis:", ["Diagnosa Penyakit", "Poli / Unit"], horizontal=True)
    kolom_fokus = "diagnosa" if fokus == "Diagnosa Penyakit" else "poli"
    with col2:
        if kolom_fokus not in df_ml.columns:
            st.error(f"Kolom '{kolom_fokus}' tidak ada.")
            return
        top_items   = df_ml[kolom_fokus].value_counts().head(30)
        pilihan_item = st.selectbox(f"Pilih {fokus}:", top_items.index.tolist())
    with col3:
        target_date = st.date_input(
            "Prediksi Sampai:",
            value=max_date + pd.Timedelta(days=60),
            min_value=max_date + pd.Timedelta(days=7),
            max_value=max_date + pd.Timedelta(days=365),
        )
    with col4:
        freq_label  = st.selectbox("Frekuensi:", ["Mingguan","Bulanan"])
        freq_map    = {"Mingguan": "W-MON", "Bulanan": "MS"}
        freq        = freq_map[freq_label]

    # ── Siapkan data time-series ──────────────────────────────
    df_item = df_ml[df_ml[kolom_fokus] == pilihan_item].copy()
    if len(df_item) < 15:
        st.error("❌ Data terlalu sedikit (< 15 baris) untuk ensemble forecasting.")
        return

    grouper   = pd.Grouper(key="tanggal_kunjungan", freq=freq)
    weekly    = df_item.groupby(grouper).size().reset_index(name="y")
    weekly    = weekly.rename(columns={"tanggal_kunjungan": "ds"})
    weekly    = weekly[weekly["y"] > 0]

    if len(weekly) < 8:
        st.error("❌ Data agregat terlalu sedikit setelah resampling.")
        return

    # Hitung periods
    last_date_ts = weekly["ds"].max()
    target_dt    = pd.to_datetime(target_date)
    if freq == "W-MON":
        periods = max(1, int((target_dt - last_date_ts).days // 7))
    else:
        periods = max(1, int((target_dt.year - last_date_ts.year) * 12 +
                             (target_dt.month - last_date_ts.month)))

    st.info(f"📊 Data: **{len(weekly)}** titik · Prediksi: **{periods}** {'minggu' if freq=='W-MON' else 'bulan'} ke depan")

    # ── Peringatan mutu deret sebelum model dijalankan ───────────────────────
    # 1) Patahan struktural: bila awal deret jauh lebih sepi daripada akhirnya,
    #    kemungkinan besar itu masa awal pencatatan digital, bukan kenaikan kasus.
    #    Model akan membacanya sebagai tren naik lalu melanjutkannya.
    _y = weekly["y"].astype(float).values
    if len(_y) >= 12:
        _k = max(3, len(_y) // 4)
        _awal, _akhir = _y[:_k].mean(), _y[-_k:].mean()
        if _akhir > 0 and _awal < 0.35 * _akhir:
            st.warning(
                f"⚠️ **Awal deret jauh lebih sepi daripada akhirnya** "
                f"(rata-rata {_awal:.1f} lawan {_akhir:.1f} per periode). Ini lazim terjadi "
                "pada masa awal pencatatan rekam medis elektronik, dan bukan berarti kasus "
                "benar-benar meningkat sebesar itu. Model akan membacanya sebagai tren naik. "
                "Persempit **Rentang Tanggal** di sidebar ke periode yang pencatatannya sudah "
                "mapan agar prediksinya lebih dapat dipercaya."
            )
    # 2) Deret terlalu pendek untuk musiman tahunan.
    _min_musim = 156 if freq == "W-MON" else 36      # ±3 tahun
    if len(weekly) < _min_musim:
        _thn = len(weekly) / (52 if freq == "W-MON" else 12)
        st.warning(
            f"⚠️ **Deret hanya mencakup sekitar {_thn:.1f} tahun.** Prophet dan SARIMA "
            "membutuhkan kira-kira tiga tahun data untuk mengenali pola musiman tahunan "
            "dengan mantap; di bawah itu komponen musimannya kurang teridentifikasi dan "
            "hasilnya bisa tidak stabil. "
            + ("Coba frekuensi **Bulanan** agar deretnya lebih ringkas dan stabil."
               if freq == "W-MON" else "Perpanjang rentang tanggal bila memungkinkan.")
        )

    # ── Jalankan Ensemble ─────────────────────────────────────
    # Hasil disimpan di st.session_state (bukan langsung ditampilkan di dalam
    # blok `if st.button(...)`) supaya TIDAK hilang saat skrip Streamlit rerun
    # karena interaksi lain — mis. klik tombol "Download Prediksi (Excel)" di
    # bawah, yang juga memicu rerun, sebelumnya membuat seluruh hasil di atasnya
    # langsung menghilang karena status tombol "Jalankan" kembali False.
    run_key = f"{kolom_fokus}|{pilihan_item}|{freq}|{periods}|{len(weekly)}"

    if st.button("🚀 Jalankan Ensemble Forecasting", type="primary"):
        with st.spinner("🔄 Melatih Prophet, XGBoost & SARIMA (paralel)..."):
            results, best_name, best_fc, ensemble_fc, model_errors = ensemble_forecast(
                weekly, periods, freq=freq
            )

        for name, err in (model_errors or {}).items():
            st.warning(f"{name} gagal: {err}")

        st.session_state["ensemble_run"] = {
            "key": run_key, "results": results, "best_name": best_name,
            "best_fc": best_fc, "ensemble_fc": ensemble_fc,
        }

    saved = st.session_state.get("ensemble_run")
    if not saved or saved["key"] != run_key:
        if saved:
            st.caption("ℹ️ Pilihan berubah — klik \"Jalankan Ensemble Forecasting\" lagi untuk hasil terbaru.")
        return

    results     = saved["results"]
    best_name   = saved["best_name"]
    best_fc     = saved["best_fc"]
    ensemble_fc = saved["ensemble_fc"]

    if not results:
        st.error("Semua model gagal. Coba dengan data lebih panjang.")
        return

    st.success(f"✅ Selesai! Model terbaik: **{best_name}**")

    # ── Tabel Evaluasi Akurasi ────────────────────────────
    st.markdown("### 📊 Evaluasi Akurasi Model")
    st.markdown("""
    > **MAE** = rata-rata error absolut · **RMSE** = root mean square error · **MAPE** = error persentase rata-rata
    > MAPE < 10% = Sangat Baik · 10–20% = Baik · > 20% = Perlu Perbaikan
    """)
    eval_rows = []
    for name, r in results.items():
        if r["metrics"]:
            row = {"Model": name, **r["metrics"]}
            row["Dinilai pada"] = ("Data uji disisihkan" if r.get("mode") == "uji"
                                   else "Data latih sendiri")
            row["Status"] = "🏆 Terbaik" if name == best_name else ""
            eval_rows.append(row)

    _mode = {r.get("mode") for r in results.values() if r.get("metrics")}
    if _mode == {"uji"}:
        st.caption(
            "Angka di bawah dihitung pada 20% data terakhir yang **disisihkan** dan tidak "
            "dipakai melatih model, sehingga mencerminkan ketepatan meramal ke depan. "
            "Prediksi yang ditampilkan di grafik tetap memakai seluruh data."
        )
    elif "insample" in _mode:
        st.warning(
            "⚠️ Sebagian model dinilai memakai data latihnya sendiri karena deret terlalu "
            "pendek untuk disisihkan. Angka akurasinya cenderung terlalu bagus — "
            "perpanjang rentang tanggal agar penilaian lebih jujur."
        )

    eval_df = pd.DataFrame(eval_rows)
    st.dataframe(
        eval_df.style.highlight_min(subset=["MAE","RMSE","MAPE"], color="#d1fae5"),
        use_container_width=True,
        hide_index=True,
    )

    # ── Grafik Perbandingan Model ─────────────────────────
    st.markdown(f"### 📈 Perbandingan Prediksi: **{pilihan_item}**")
    fig = go.Figure()

    # Aktual
    fig.add_trace(go.Scatter(
        x=weekly["ds"], y=weekly["y"],
        mode="lines+markers", name="Aktual",
        line=dict(color="#1d4ed8", width=2),
    ))

    colors = {"Prophet": "#ef4444", "XGBoost": "#16a34a", "SARIMA": "#d97706"}
    for name, r in results.items():
        fc = r["fc"]
        fc_future = fc[fc["ds"] > last_date_ts]
        if "yhat" in fc.columns:
            fig.add_trace(go.Scatter(
                x=fc_future["ds"], y=fc_future["yhat"].clip(lower=0),
                mode="lines", name=f"{name}",
                line=dict(color=colors.get(name, "#6b7280"), width=2, dash="dot"),
            ))

    # Ensemble
    if ensemble_fc is not None:
        ens_future = ensemble_fc[ensemble_fc["ds"] > last_date_ts]
        if not ens_future.empty:
            fig.add_trace(go.Scatter(
                x=ens_future["ds"], y=ens_future["yhat_ensemble"].clip(lower=0),
                mode="lines+markers", name="⭐ Ensemble",
                line=dict(color="#7c3aed", width=3),
            ))

    # Best model confidence interval (Prophet saja yang punya CI lengkap)
    if "Prophet" in results and best_name == "Prophet":
        fc_p = results["Prophet"]["fc"]
        fc_p_future = fc_p[fc_p["ds"] > last_date_ts]
        if {"yhat_upper","yhat_lower"}.issubset(fc_p_future.columns):
            fig.add_trace(go.Scatter(
                x=fc_p_future["ds"].tolist() + fc_p_future["ds"].tolist()[::-1],
                y=fc_p_future["yhat_upper"].tolist() + fc_p_future["yhat_lower"].tolist()[::-1],
                fill="toself", fillcolor="rgba(239,68,68,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="CI Prophet 80%", hoverinfo="skip",
            ))

    fig.update_layout(
        xaxis_title="Periode", yaxis_title="Jumlah Kunjungan",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Ringkasan Estimasi ────────────────────────────────
    st.markdown(f"### 📢 Kesimpulan Estimasi Hingga {target_dt.strftime('%d %B %Y')}")

    # Gunakan ensemble atau best model
    use_fc = None
    use_label = ""
    if ensemble_fc is not None and "yhat_ensemble" in ensemble_fc.columns:
        use_fc    = ensemble_fc[ensemble_fc["ds"] > last_date_ts][["ds","yhat_ensemble"]].rename(columns={"yhat_ensemble":"yhat"})
        use_label = "Ensemble (berbobot)"
    elif best_fc is not None and "yhat" in best_fc.columns:
        use_fc    = best_fc[best_fc["ds"] > last_date_ts][["ds","yhat"]]
        use_label = best_name

    if use_fc is not None and not use_fc.empty:
        use_fc["yhat"] = use_fc["yhat"].clip(lower=0)
        total_est      = int(round(use_fc["yhat"].sum()))
        akhir_row      = use_fc.iloc[-1]
        tgl_akhir      = akhir_row["ds"].strftime("%d %B %Y")
        est_akhir      = int(round(akhir_row["yhat"]))

        col_a, col_b, col_c = st.columns([2, 1, 1])
        with col_a:
            st.markdown(f"""
            <div class="accuracy-box">
            <div class="highlight-estimasi">
            📆 Hingga <b>{tgl_akhir}</b>, model <b>{use_label}</b> memperkirakan 
            total akumulasi <b>{total_est:,} kunjungan/kasus</b> untuk <b>{pilihan_item}</b>.<br><br>
            Pada periode terakhir, diperkirakan <b>{est_akhir} kunjungan</b> baru.
            </div></div>
            """.replace(",","."), unsafe_allow_html=True)
        with col_b:
            st.metric("Total Estimasi Akumulasi", f"{total_est:,}".replace(",","."))
        with col_c:
            st.metric("Estimasi Periode Terakhir", f"{est_akhir}")

        # Download
        dl_df = use_fc.copy()
        dl_df.columns = ["Periode","Prediksi_Jumlah"]
        st.download_button(
            "📥 Download Prediksi (Excel)",
            convert_df_to_excel(dl_df),
            f"prediksi_{pilihan_item.lower().replace(' ','_')}.xlsx",
        )


# ══════════════════════════════════════════════════════════════
# HALAMAN: DETEKSI PENYAKIT DOMINAN PER MUSIM
# ══════════════════════════════════════════════════════════════

def page_disease_seasonality(df_filtered, filter_info):
    st.subheader("🌧️ Deteksi Penyakit Dominan per Musim / Bulan")
    show_active_filters(filter_info)

    if df_filtered is None or len(df_filtered) == 0:
        st.warning("⚠️ Data kosong.")
        return
    if not {"tanggal_kunjungan","diagnosa"}.issubset(df_filtered.columns):
        st.error("❌ Butuh kolom 'tanggal_kunjungan' dan 'diagnosa'.")
        return

    df = df_filtered.copy()
    df["bulan"]       = df["tanggal_kunjungan"].dt.month
    df["nama_bulan"]  = df["tanggal_kunjungan"].dt.strftime("%b")
    if "musim" not in df.columns:
        df["musim"] = df["bulan"].apply(label_musim)

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Heatmap Bulanan", "Penyakit per Musim", "Tren Penyakit Spesifik"])

    with tab1:
        st.markdown("#### 🗓️ Heatmap Frekuensi Penyakit per Bulan")
        top_n = st.slider("Jumlah penyakit ditampilkan", 5, 20, 10, key="heat_n")
        top_diag = df["diagnosa"].value_counts().head(top_n).index.tolist()
        df_heat = df[df["diagnosa"].isin(top_diag)].copy()
        pivot = df_heat.groupby(["diagnosa","bulan"]).size().unstack(fill_value=0)
        pivot.columns = ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"][:len(pivot.columns)]

        fig_heat = px.imshow(
            pivot,
            color_continuous_scale="YlOrRd",
            aspect="auto",
            title="Jumlah Kasus per Penyakit per Bulan",
            labels={"color":"Jumlah Kasus"},
        )
        fig_heat.update_layout(margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_heat, use_container_width=True)
        st.download_button("📥 Download Heatmap Data", convert_df_to_excel(pivot.reset_index()), "heatmap_penyakit.xlsx")

    with tab2:
        st.markdown("#### 🌦️ Top Penyakit per Musim")
        _kolom = st.columns(2) + st.columns(2)     # 4 musim, disusun 2x2
        for musim_label, col_ui in zip(MUSIM_URUT, _kolom):
            df_musim = df[df["musim"] == musim_label]
            if df_musim.empty:
                col_ui.info(f"Tidak ada data untuk {musim_label}")
                continue
            top_diag_musim = df_musim["diagnosa"].value_counts().head(10).reset_index()
            top_diag_musim.columns = ["Diagnosa","Kasus"]
            fig = px.bar(
                top_diag_musim, x="Kasus", y="Diagnosa",
                orientation="h", title=musim_label,
                color="Kasus", color_continuous_scale="Blues",
                text="Kasus",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(coloraxis_showscale=False, height=350, yaxis=dict(categoryorder="total ascending"))
            col_ui.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 📊 Perbandingan Intensitas Penyakit antar Musim")
        st.caption(
            "Dibaca per musim, bukan sekadar hujan lawan kemarau. Penyakit saluran "
            "napas kerap memuncak pada masa **pancaroba**, dan puncak itu tidak terlihat "
            "bila setahun hanya dibelah menjadi dua musim."
        )
        df_comp = df.groupby(["musim","diagnosa"]).size().unstack(fill_value=0).T
        _ada = [m for m in MUSIM_URUT if m in df_comp.columns]
        if len(_ada) >= 2:
            df_comp = df_comp[_ada]
            df_comp["Musim Dominan"] = df_comp[_ada].idxmax(axis=1)
            # urutkan berdasarkan seberapa timpang sebaran kasusnya antar musim
            _tot = df_comp[_ada].sum(axis=1).replace(0, 1)
            df_comp["Ketimpangan"] = (df_comp[_ada].max(axis=1) / _tot * 100).round(1)
            df_comp_top = df_comp.sort_values("Ketimpangan", ascending=False).head(15)
            st.dataframe(df_comp_top.reset_index().rename(columns={"diagnosa":"Diagnosa"}), use_container_width=True, hide_index=True)
            
        st.markdown("#### 📋 Rincian Lengkap Kasus per Musim")
        df_rincian = df.groupby(["diagnosa", "musim"]).size().reset_index(name="Jumlah Kasus")
        df_pivot = df_rincian.pivot(index="diagnosa", columns="musim", values="Jumlah Kasus").fillna(0).astype(int)
        df_pivot["Total Kasus"] = df_pivot.sum(axis=1)
        df_pivot = df_pivot.sort_values("Total Kasus", ascending=False).reset_index()
        st.dataframe(df_pivot, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("#### 📈 Tren Bulanan Penyakit Tertentu")
        top_all = df["diagnosa"].value_counts().head(20).index.tolist()
        penyakit_pilih = st.multiselect("Pilih penyakit untuk dibandingkan:", top_all, default=top_all[:3])
        if penyakit_pilih:
            df_trend = df[df["diagnosa"].isin(penyakit_pilih)].groupby(["bulan","diagnosa"]).size().reset_index(name="kasus")
            bulan_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"Mei",6:"Jun",
                         7:"Jul",8:"Agu",9:"Sep",10:"Okt",11:"Nov",12:"Des"}
            df_trend["nama_bulan"] = df_trend["bulan"].map(bulan_map)
            fig_trend = px.line(
                df_trend, x="nama_bulan", y="kasus",
                color="diagnosa", markers=True,
                title="Tren Kasus Bulanan per Penyakit",
                category_orders={"nama_bulan": list(bulan_map.values())},
            )
            st.plotly_chart(fig_trend, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# HALAMAN LAINNYA
# ══════════════════════════════════════════════════════════════

def page_overview(df_filtered, filter_info):
    st.subheader("📌 Ringkasan Umum")
    show_active_filters(filter_info)
    if df_filtered is None or len(df_filtered) == 0:
        st.warning("Tidak ada data.")
        return
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Kunjungan", len(df_filtered))
    if "no_rm"    in df_filtered.columns: col2.metric("Pasien Unik", df_filtered["no_rm"].nunique())
    if "poli"     in df_filtered.columns: col3.metric("Poli Aktif",  df_filtered["poli"].nunique())
    if "diagnosa" in df_filtered.columns: col4.metric("Diagnosa",    df_filtered["diagnosa"].nunique())
    st.markdown("---")
    if "tanggal_kunjungan" in df_filtered.columns:
        st.markdown("### 📈 Tren Kunjungan")
        trend = df_filtered.groupby(["tahun","bulan","nama_bulan"]).size().reset_index(name="count") \
                           .sort_values(["tahun","bulan"])
        trend["label"] = trend["nama_bulan"].astype(str) + "-" + trend["tahun"].astype(str)
        st.line_chart(trend.set_index("label")["count"])
        st.download_button("📥 Download Tren", convert_df_to_excel(trend), "tren_kunjungan.xlsx")
    if "poli" in df_filtered.columns:
        st.markdown("### 🏥 Distribusi Poli")
        df_poli = df_filtered["poli"].value_counts().reset_index()
        df_poli.columns = ["Poli","Jumlah"]
        st.bar_chart(df_poli.set_index("Poli"))


def page_kunjungan(df_filtered, filter_info):
    st.subheader("👥 Analisis Kunjungan")
    show_active_filters(filter_info)
    if df_filtered is None or len(df_filtered) == 0: return
    col1, col2 = st.columns(2)
    if "jenis_kelamin" in df_filtered.columns:
        df_jk = df_filtered["jenis_kelamin"].value_counts().reset_index()
        df_jk.columns = ["Jenis Kelamin","Jumlah"]
        col1.markdown("#### Jenis Kelamin")
        col1.bar_chart(df_jk.set_index("Jenis Kelamin"))
    if "kelompok_umur" in df_filtered.columns:
        df_umur = df_filtered["kelompok_umur"].value_counts().sort_index().reset_index()
        df_umur.columns = ["Kelompok Umur","Jumlah"]
        col2.markdown("#### Kelompok Umur")
        col2.bar_chart(df_umur.set_index("Kelompok Umur"))


def page_penyakit(df_filtered, filter_info):
    st.subheader("🦠 Analisis Penyakit")
    show_active_filters(filter_info)
    if df_filtered is None or len(df_filtered) == 0: return
    if "diagnosa" not in df_filtered.columns:
        st.error("❌ Kolom 'diagnosa' tidak ditemukan.")
        return
        
    col_s, col_u, col_t = st.columns([2,2,1])
    with col_s: top_n = st.slider("Jumlah diagnosa", 5, 30, 10)
    with col_u:
        urutan = st.radio("Urut:", ["⬇️ Terbanyak","⬆️ Tersedikit"], horizontal=True)
    with col_t:
        orientasi = st.selectbox("Orientasi", ["Horizontal","Vertikal"])
        
    ascending = urutan.startswith("⬆️")
    df_diag = df_filtered["diagnosa"].value_counts().head(top_n).reset_index()
    df_diag.columns = ["Diagnosa","Jumlah Kasus"]
    df_diag = df_diag.sort_values("Jumlah Kasus", ascending=ascending).reset_index(drop=True)
    
    chart_h = max(350, top_n * 38)
    if orientasi == "Horizontal":
        fig = px.bar(df_diag, x="Jumlah Kasus", y="Diagnosa", orientation="h",
                     text="Jumlah Kasus", color="Jumlah Kasus", color_continuous_scale="Blues")
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis=dict(categoryorder="total ascending" if ascending else "total descending"),
                          coloraxis_showscale=False, height=chart_h)
    else:
        fig = px.bar(df_diag, x="Diagnosa", y="Jumlah Kasus",
                     text="Jumlah Kasus", color="Jumlah Kasus", color_continuous_scale="Blues")
        fig.update_layout(xaxis=dict(tickangle=-35), coloraxis_showscale=False, height=chart_h)
        
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📋 Rincian Data Penyakit")
    st.dataframe(df_diag, use_container_width=True, hide_index=True)
    
    st.download_button("📥 Download Data Penyakit", convert_df_to_excel(df_diag), "top_penyakit.xlsx")


def page_peta_persebaran(df_filtered, filter_info):
    st.subheader("🗺️ Peta Persebaran Penyakit")
    show_active_filters(filter_info)
    if df_filtered is None or len(df_filtered) == 0: return
    if not {"desa","diagnosa"}.issubset(df_filtered.columns):
        st.error("❌ Butuh kolom 'desa' dan 'diagnosa'.")
        return
        
    top_penyakit = df_filtered["diagnosa"].value_counts().head(20).index.tolist()
    pilihan = st.selectbox("Filter Peta (Berdasarkan Diagnosa):", ["-- Semua Top 10 --"] + top_penyakit)
    
    if pilihan != "-- Semua Top 10 --":
        df_map = df_filtered[df_filtered["diagnosa"] == pilihan].copy()
    else:
        df_map = df_filtered[df_filtered["diagnosa"].isin(df_filtered["diagnosa"].value_counts().head(10).index)]
        
    df_grouped = df_map.groupby(["desa","diagnosa"]).size().reset_index(name="jumlah_kasus")
    
    koordinat_desa = {
        "Donan":(-7.2131,111.6364),"Gapluk":(-7.2017,111.6617),
        "Kaliombo":(-7.235,111.6817),"Kuniran":(-7.2346,111.6510),
        "Ngrejeng":(-7.2260,111.7071),"Pelem":(-7.2394,111.7011),
        "Pojok":(-7.1892,111.6728),"Punggur":(-7.2048,111.6808),
        "Purwosari":(-7.1798,111.6608),"Sedahkidul":(-7.1973,111.6792),
        "Tinumpuk":(-7.2117,111.68),"Tlatah":(-7.2172,111.6975),
    }
    # ── FIX: desa di luar 12 desa wilayah kerja TIDAK lagi dilempar ke satu
    # koordinat cadangan. Sebelumnya setiap nama desa asing menumpuk di titik yang
    # sama (-7.1509, 111.8817) — bukan desa mana pun — sehingga pada data nyata
    # 334 nama desa dengan 2.972 kunjungan membentuk "gelembung hantu" yang terbaca
    # sebagai kantong penyakit. Kini pasien luar wilayah dipisahkan dari peta.
    def _norm(d): return str(d).strip().title()
    df_grouped["_dalam"] = df_grouped["desa"].apply(lambda x: _norm(x) in koordinat_desa)
    df_luar    = df_grouped[~df_grouped["_dalam"]].drop(columns=["_dalam"])
    df_grouped = df_grouped[df_grouped["_dalam"]].drop(columns=["_dalam"]).copy()

    if len(df_luar):
        n_kasus = int(df_luar["jumlah_kasus"].sum())
        st.info(
            f"🧭 {n_kasus:,} kunjungan berasal dari {df_luar['desa'].nunique()} desa "
            "di luar wilayah kerja Puskesmas Purwosari, sehingga tidak ditampilkan "
            "di peta. Rinciannya tetap ada pada tabel di bawah.".replace(",", ".")
        )
    if df_grouped.empty:
        st.warning("Tidak ada kunjungan dari 12 desa wilayah kerja pada filter ini.")
        return

    df_grouped["latitude"]  = df_grouped["desa"].apply(lambda x: koordinat_desa[_norm(x)][0])
    df_grouped["longitude"] = df_grouped["desa"].apply(lambda x: koordinat_desa[_norm(x)][1])
    
    # Catatan: px.scatter_mapbox() dihapus total di Plotly 7.0 (diganti px.scatter_map(),
    # berbasis MapLibre — tidak butuh Mapbox token). "mapbox_style" juga berganti nama
    # menjadi "map_style" di update_layout.
    fig = px.scatter_map(
        df_grouped, lat="latitude", lon="longitude",
        color="diagnosa", size="jumlah_kasus",
        hover_name="desa", hover_data={"diagnosa":True,"jumlah_kasus":True},
        custom_data=["desa"],
        zoom=11.5, center={"lat":-7.218,"lon":111.675}, height=550,
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    fig.update_layout(map_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
    
    st.markdown("*(💡 Klik salah satu titik/lingkaran pada peta untuk melihat detail data dari desa tersebut)*")
    
    map_event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points")

    selected_desa = None
    if map_event and map_event.get("selection") and map_event["selection"].get("points"):
        selected_desa = map_event["selection"]["points"][0]["customdata"][0]

    if selected_desa:
        st.markdown("---")
        st.markdown(f"### 📍 Rincian Khusus Desa: **{selected_desa}**")
        
        df_desa_detail = df_filtered[df_filtered["desa"].str.title() == selected_desa]
        
        st.markdown("#### 📊 Top Diagnosa")
        df_desa_diag = df_desa_detail["diagnosa"].value_counts().head(10).reset_index()
        df_desa_diag.columns = ["Diagnosa", "Jumlah Kasus"]
        fig_desa = px.bar(df_desa_diag, x="Jumlah Kasus", y="Diagnosa", orientation="h", color="Jumlah Kasus", color_continuous_scale="Viridis")
        fig_desa.update_layout(yaxis=dict(categoryorder="total ascending"), height=400, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_desa, use_container_width=True)
        
        st.markdown("#### 📋 Data Histori Kunjungan")
        kolom_tabel = [c for c in ["tanggal_kunjungan", "no_rm", "umur", "jenis_kelamin", "diagnosa", "poli"] if c in df_desa_detail.columns]
        st.dataframe(df_desa_detail[kolom_tabel], use_container_width=True, hide_index=True)
            
    else:
        st.markdown("### 📋 Tabel Akumulasi")
        _tab = df_grouped.drop(columns=["latitude", "longitude"]).copy()
        _tab["wilayah"] = "Dalam wilayah"
        if len(df_luar):
            _l = df_luar.copy(); _l["wilayah"] = "Luar wilayah"
            _tab = pd.concat([_tab, _l], ignore_index=True)
        st.dataframe(_tab.sort_values("jumlah_kasus", ascending=False),
                     use_container_width=True, hide_index=True)


def page_pembiayaan(df_filtered, filter_info):
    st.subheader("💳 Analisis Pembiayaan")
    if df_filtered is None or "pembiayaan" not in df_filtered.columns: return
    df_b = df_filtered["pembiayaan"].value_counts().reset_index()
    df_b.columns = ["Pembiayaan","Jumlah"]
    st.bar_chart(df_b.set_index("Pembiayaan"))
    st.download_button("📥 Download Pembiayaan", convert_df_to_excel(df_b), "pembiayaan.xlsx")


def page_data(df_filtered, filter_info):
    st.subheader("📄 Data & Unduhan")
    if df_filtered is None: return
    st.dataframe(df_filtered)
    st.download_button("💾 Download CSV", df_filtered.to_csv(index=False).encode("utf-8"), "data_puskesmas.csv", "text/csv")


def page_quality(df):
    st.subheader("🧹 Kualitas Data")
    if df is None: return
    st.write(f"Duplikasi: **{df.duplicated().sum()}** baris")
    st.markdown("**Missing Values:**")
    st.dataframe(df.isna().sum().to_frame("Missing Count"), use_container_width=True)


def page_ai_assistant(df_filtered, filter_info, is_genai):
    st.subheader("🤖 Asisten AI")
    if df_filtered is None or df_filtered.empty:
        st.warning("Upload dan filter data terlebih dahulu.")
        return
    if not is_genai:
        st.error("❌ API Key Gemini belum diset.")
        return
    total = len(df_filtered)
    top_dx = ", ".join([f"{k}({v})" for k,v in df_filtered["diagnosa"].value_counts().head(5).items()]) \
             if "diagnosa" in df_filtered.columns else "-"
    top_pl = ", ".join([f"{k}({v})" for k,v in df_filtered["poli"].value_counts().head(3).items()]) \
             if "poli" in df_filtered.columns else "-"
    ctx = f"""[DATA REAL-TIME]
- Total kunjungan: {total}
- 5 penyakit terbanyak: {top_dx}
- 3 poli terpadat: {top_pl}"""
    user_q = st.text_area("Tanyakan strategi/analisis:", placeholder="Contoh: Program promkes apa yang paling mendesak?")
    if st.button("Kirim"):
        if not user_q.strip(): st.warning("Pertanyaan kosong."); return
        prompt = f"""Anda adalah Analis Kesehatan Masyarakat di UPT Puskesmas Purwosari (Bojonegoro).
{ctx}
Jawab pertanyaan berikut secara spesifik, berbasis data, dan terstruktur:
{user_q}"""
        with st.spinner("AI menganalisis..."):
            try:
                import google.generativeai as genai
                model = genai.GenerativeModel("gemini-3.6-flash")
                resp  = model.generate_content(prompt)
                st.markdown("### 📊 Analisis AI:")
                st.markdown(resp.text)
            except Exception as e:
                st.error(f"Gagal: {e}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    is_genai = get_gemini_client()
    df_filtered, filter_info = apply_filters(None)

    pages = {
        "📌 Ringkasan":                page_overview,
        "👥 Analisis Kunjungan":       page_kunjungan,
        "🦠 Analisis Penyakit":        page_penyakit,
        "🌧️ Penyakit per Musim":      page_disease_seasonality,
        "🗺️ Peta Persebaran":         page_peta_persebaran,
        "🤖 Ensemble Forecasting":     page_ml_upgraded,
        "💳 Pembiayaan":               page_pembiayaan,
        "🧹 Kualitas Data":            page_quality,
        "📄 Data & Unduhan":           page_data,
        "💬 Asisten AI":               None,
    }

    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🗂️ Navigasi")
        page_sel = st.radio("Pilih halaman:", list(pages.keys()), label_visibility="collapsed")

    if df_filtered is None:
        st.info("👈 Upload file data di sidebar untuk memulai.")
        return

    if page_sel == "💬 Asisten AI":
        page_ai_assistant(df_filtered, filter_info, is_genai)
    elif page_sel == "🧹 Kualitas Data":
        page_quality(df_filtered)
    else:
        fn = pages[page_sel]
        if fn: fn(df_filtered, filter_info)

if __name__ == "__main__":
    main()
