"""
IPO Insight AI — tanya jawab prospektus IPO, diperkaya data e-IPO (Kaggle).

Alur jawaban chat:
1. Planner (LLM, JSON) menentukan sumber: prospektus / dataset e-IPO / keduanya,
   dan bila perlu menyusun query terstruktur (filter, group by, agregasi).
2. Query dataset dieksekusi dengan pandas secara aman (whitelist operasi, tanpa eval kode).
3. Kutipan prospektus (Supabase pgvector) + tabel hasil dataset dikirim ke LLM untuk dijawab.
"""
import glob
import html
import json
import numbers
import os
import re

# Pakai PyTorch saja; cegah transformers memuat TensorFlow/Flax (sumber error protobuf 'GetPrototype').
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def get_config(key, default=None):
    """Ambil konfigurasi dari Streamlit Secrets (deploy) atau environment / .env (lokal)."""
    try:
        value = st.secrets.get(key)
        if value not in (None, ""):
            return str(value)
    except Exception:
        pass
    value = os.getenv(key)
    return value if value not in (None, "") else default


# ══════════════════════════════════════════════════════════════════════════════
# Konfigurasi (tidak ditampilkan ke pengguna)
# ══════════════════════════════════════════════════════════════════════════════
APP_NAME = "IPO Insight AI"
DOC_NAME = get_config("DOC_NAME", "PT Niramas Utama Tbk")
DOC_KEYWORD = get_config("DOC_KEYWORD", "Niramas")     # untuk menemukan baris perusahaan di dataset
DOC_TICKER = get_config("DOC_TICKER", "")              # opsional, mis. kode saham perusahaan

EMBED_MODEL = "intfloat/multilingual-e5-base"
DEEPSEEK_MODEL = get_config("DEEPSEEK_MODEL", "deepseek-v4-flash")
TOP_K = int(get_config("RAG_TOP_K", "10"))
TABLE_NAME = get_config("SUPABASE_TABLE", "documents")
QUERY_NAME = get_config("SUPABASE_QUERY_NAME", "match_documents")
VECTORSTORE_FILTER = {"vectorstore_name": "structure_aware"}

EIPO_DATASET = get_config("EIPO_DATASET", "fahmirk/e-ipo-realtime-data")
EIPO_CSV = get_config("EIPO_CSV", "")      # path CSV lokal (opsional; diprioritaskan bila diisi)
EIPO_FILE = get_config("EIPO_FILE", "")    # nama file tertentu di dataset Kaggle (opsional)

HISTORY_TURNS = 3
MAX_SOURCES_SHOWN = 5
MAX_TABLE_ROWS_CONTEXT = 40
MAX_QUERIES = 3
NOT_FOUND = "Informasi tidak tersedia dalam dokumen yang berhasil ditemukan."

QUICK_QUESTIONS = [
    "Berapa harga penawaran saham IPO dan nilai nominal per saham?",
    "Kapan masa penawaran umum dan tanggal pencatatan saham?",
    "Apa saja faktor risiko utama perusahaan?",
    "Bagaimana rencana penggunaan dana hasil IPO?",
    "Penjamin emisi mana yang paling sering menangani IPO dalam data e-IPO?",
    f"Siapa penjamin emisi {DOC_NAME} dan berapa IPO lain yang pernah mereka tangani?",
]

st.set_page_config(page_title=APP_NAME, page_icon="📈", layout="centered")

# ══════════════════════════════════════════════════════════════════════════════
# Kamus kode penjamin emisi (kode Anggota Bursa BEI -> nama lengkap)
# Sumber: daftar kode broker BEI 2025 (investasiku.id, merujuk IDX) & Bisnis.com (Sep 2023)
# ══════════════════════════════════════════════════════════════════════════════
UNDERWRITER_NAMES = {
    "AD": "PT OSO Sekuritas Indonesia", "AF": "PT Harita Kencana Sekuritas",
    "AG": "PT Kiwoom Sekuritas Indonesia", "AH": "PT Shinhan Sekuritas Indonesia",
    "AI": "PT UOB Kay Hian Sekuritas", "AK": "PT UBS Sekuritas Indonesia",
    "AO": "PT Erdikha Elit Sekuritas", "AP": "PT Pacific Sekuritas Indonesia",
    "AR": "PT Binaartha Sekuritas", "AT": "PT Phintraco Sekuritas",
    "AZ": "PT Sucor Sekuritas", "BB": "PT Verdhana Sekuritas Indonesia",
    "BF": "PT Inti Fikasa Sekuritas", "BK": "PT J.P. Morgan Sekuritas Indonesia",
    "BQ": "PT Korea Investment and Sekuritas Indonesia", "BR": "PT Trust Sekuritas",
    "BS": "PT Equity Sekuritas Indonesia", "CC": "PT Mandiri Sekuritas",
    "CD": "PT Mega Capital Sekuritas", "CP": "PT KB Valbury Sekuritas",
    "DD": "PT Makindo Sekuritas", "DH": "PT Sinarmas Sekuritas",
    "DP": "PT DBS Vickers Sekuritas Indonesia", "DR": "PT RHB Sekuritas Indonesia",
    "DU": "PT KAF Sekuritas Indonesia", "DX": "PT Bahana Sekuritas",
    "EL": "PT Evergreen Sekuritas Indonesia", "EP": "PT MNC Sekuritas",
    "ES": "PT Ekokapital Sekuritas", "FO": "PT Forte Global Sekuritas",
    "FS": "PT Yuanta Sekuritas Indonesia", "FZ": "PT Waterfront Sekuritas Indonesia",
    "GA": "PT BNC Sekuritas Indonesia", "GI": "PT Webull Sekuritas Indonesia",
    "GR": "PT Panin Sekuritas Tbk", "HD": "PT KGI Sekuritas Indonesia",
    "HP": "PT Henan Putihrai Sekuritas", "IC": "PT Integrity Capital Sekuritas",
    "ID": "PT Anugerah Sekuritas Indonesia", "IF": "PT Samuel Sekuritas Indonesia",
    "IH": "PT Pacific 2000 Sekuritas", "II": "PT Danatama Makmur Sekuritas",
    "IN": "PT Investindo Nusantara Sekuritas", "IT": "PT Inti Teladan Sekuritas",
    "IU": "PT Indo Capital Sekuritas", "KI": "PT Ciptadana Sekuritas Asia",
    "KK": "PT Phillip Sekuritas Indonesia", "KZ": "PT CLSA Sekuritas Indonesia",
    "LG": "PT Trimegah Sekuritas Indonesia Tbk", "LS": "PT Reliance Sekuritas Indonesia Tbk",
    "MG": "PT Semesta Indovest Sekuritas", "MI": "PT Victoria Sekuritas Indonesia",
    "MU": "PT Minna Padi Investama Sekuritas Tbk", "NI": "PT BNI Sekuritas",
    "OD": "PT BRI Danareksa Sekuritas", "OK": "PT NET Sekuritas",
    "PC": "PT FAC Sekuritas Indonesia", "PD": "PT Indo Premier Sekuritas",
    "PF": "PT Danasakti Sekuritas Indonesia", "PG": "PT Panca Global Sekuritas",
    "PI": "PT Magenta Kapital Sekuritas Indonesia", "PO": "PT Pilarmas Investindo Sekuritas",
    "PP": "PT Aldiracita Sekuritas Indonesia", "QA": "PT Tuntun Sekuritas Indonesia",
    "RB": "PT Ina Sekuritas Indonesia", "RF": "PT Buana Capital Sekuritas",
    "RG": "PT Profindo Sekuritas Indonesia",
    "RO": "PT Pluang Maju Sekuritas",                  # dahulu PT Nilai Inti Sekuritas
    "RS": "PT Yulie Sekuritas Indonesia Tbk", "RX": "PT Macquarie Sekuritas Indonesia",
    "SA": "PT Elit Sukses Sekuritas", "SF": "PT Surya Fajar Sekuritas",
    "SH": "PT Artha Sekuritas Indonesia", "SQ": "PT BCA Sekuritas",
    "SS": "PT Supra Sekuritas Indonesia", "TF": "PT Universal Broker Indonesia Sekuritas",
    "TP": "PT OCBC Sekuritas Indonesia", "TS": "PT Dwidana Sakti Sekuritas",
    "XA": "PT NH Korindo Sekuritas Indonesia", "XC": "PT Ajaib Sekuritas Asia",
    "XL": "PT Stockbit Sekuritas Digital",
    "YB": "PT Yakin Bertumbuh Sekuritas",              # dahulu PT Jasa Utama Capital Sekuritas
    "YJ": "PT Lotus Andalan Sekuritas", "YO": "PT Amantara Sekuritas Indonesia",
    "YP": "PT Mirae Asset Sekuritas Indonesia",
    "YU": "PT CGS International Sekuritas Indonesia",  # dahulu PT CGS-CIMB Sekuritas Indonesia
    "ZP": "PT Maybank Sekuritas Indonesia", "ZR": "PT Bumiputera Sekuritas",
}
# Ada di daftar 2023, tidak ada di daftar 2025 (kemungkinan tidak aktif) — tetap dipakai untuk IPO lama
INACTIVE_UNDERWRITER_NAMES = {
    "AN": "PT Wanteg Sekuritas", "CS": "PT Credit Suisse Sekuritas Indonesia",
    "DM": "PT Masindo Artha Sekuritas", "GW": "PT HSBC Sekuritas Indonesia",
    "IP": "PT Yugen Bertumbuh Sekuritas", "MK": "PT Ekuator Swarna Sekuritas",
    "PS": "PT Paramitra Alfa Sekuritas", "SC": "PT IMG Sekuritas",
    "TX": "PT Dhanawibawa Sekuritas Indonesia",
}
ALL_UNDERWRITER_NAMES = {**INACTIVE_UNDERWRITER_NAMES, **UNDERWRITER_NAMES}
UNKNOWN_SUFFIX = " (kode tidak dikenal)"

# ══════════════════════════════════════════════════════════════════════════════
# Styling — warna semi-transparan / inherit agar aman di mode terang & gelap
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
:root {
  --accent-soft: rgba(31,157,104,.14);
  --surface: rgba(128,128,128,.07);
  --border: rgba(128,128,128,.22);
}
.block-container { padding-top: 2rem; padding-bottom: 6rem; max-width: 900px; }
.hero {
  background: linear-gradient(120deg,#123829 0%,#176b4d 60%,#1f8a5c 100%);
  color:#fff; padding:22px 26px; border-radius:18px; margin-bottom:14px;
}
.hero h1 { color:#fff; margin:0 0 4px; font-size:28px; line-height:1.2; padding:0; }
.hero p  { color:rgba(255,255,255,.82); margin:0; font-size:15px; }
.hero .doc { display:inline-block; margin-top:12px; padding:4px 10px; border-radius:99px;
  background:rgba(255,255,255,.14); color:#fff; font-size:12px; font-weight:600; }
.stat-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; margin:6px 0 14px; }
.stat { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:12px 14px; }
.stat .label { font-size:12px; opacity:.65; margin-bottom:4px; }
.stat .value { font-size:15px; font-weight:600; line-height:1.35; word-break:break-word; }
.empty-hint { opacity:.7; font-size:14px; margin:4px 0 10px; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: var(--accent-soft); border-radius: 14px; }
[data-testid="stChatMessage"] { padding: 12px 14px; }
div.stButton > button { border-radius:10px; text-align:left; justify-content:flex-start; }
div[data-testid="stMetric"] { background:var(--surface); border:1px solid var(--border);
  border-radius:12px; padding:12px 14px; }
.disclaimer { opacity:.6; font-size:12px; text-align:center; margin-top:18px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset e-IPO: pemuatan & pembersihan
# ══════════════════════════════════════════════════════════════════════════════
UW_LIST = "_penjamin_list"           # internal: list nama penjamin per IPO
UW_NAMES = "penjamin_emisi_nama"     # string nama lengkap, dipisah "; "
UW_SINGLE = "penjamin_emisi"         # virtual: satu nama per baris (setelah explode)

_UW_COL = re.compile(r"underwriter|penjamin|(^|_)uw($|_)", re.I)
_UW_SPLIT = re.compile(r"[,;/|&\n]+")
_CODES_SPACED = re.compile(r"^(?:[A-Z]{2}\s+)+[A-Z]{2}$")
_CODE_PREFIX = re.compile(r"^([A-Z]{2})\s*[-:(]")
_DATE_HINT = re.compile(r"date|tanggal|tgl|listing|period|start|end|open|close|waktu|time", re.I)
_ID_HINT = re.compile(r"ticker|kode|code|symbol|(^|_)id($|_)|name|nama|phone|telp|npwp|web|url|email", re.I)


def _as_text(s):
    """Series -> string murni; sel kosong/NaN menjadi "" (aman untuk pandas 2 & 3)."""
    return s.map(lambda v: "" if v is None or (not isinstance(v, (list, tuple)) and pd.isna(v)) else str(v))


def _is_text(s):
    """Kolom teks (pandas 2: object, pandas 3: str)."""
    return s.dtype == object or pd.api.types.is_string_dtype(s)


def underwriter_name(code):
    code = str(code).strip().upper()
    return ALL_UNDERWRITER_NAMES.get(code, code + UNKNOWN_SUFFIX)


def parse_underwriters(value):
    """'CC, OD' / "['CC','OD']" / 'CC OD' / 'CC - Mandiri' -> list nama lengkap."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = re.sub(r"[\[\]'\"]", "", str(value))
    names = []
    for tok in _UW_SPLIT.split(text):
        tok = tok.strip()
        if not tok or tok.lower() in ("nan", "none", "-", "null"):
            continue
        up = tok.upper()
        if len(tok) == 2 and tok.isalpha():
            names.append(underwriter_name(up))
        elif _CODES_SPACED.match(up):
            names += [underwriter_name(c) for c in up.split()]
        elif (m := _CODE_PREFIX.match(tok)) and m.group(1) in ALL_UNDERWRITER_NAMES:
            names.append(ALL_UNDERWRITER_NAMES[m.group(1)])
        else:
            names.append(tok)   # sudah berupa nama
    return list(dict.fromkeys(names))


_ID_THOUSANDS = re.compile(r"^-?\d{1,3}(\.\d{3})+(,\d+)?$")    # 1.000.000,50
_EN_THOUSANDS = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")    # 1,000,000.50
_DEC_COMMA = re.compile(r"^-?\d+,\d+$")                          # 12,5


def _normalize_number(t):
    if not isinstance(t, str):
        return t
    if _ID_THOUSANDS.match(t):
        return t.replace(".", "").replace(",", ".")
    if _EN_THOUSANDS.match(t):
        return t.replace(",", "")
    if _DEC_COMMA.match(t):
        return t.replace(",", ".")
    return t


def _to_number(s):
    txt = _as_text(s).str.strip().str.replace(r"(?i)rp\.?|idr|%|\s", "", regex=True)
    valid = txt.ne("") & ~txt.str.lower().isin(["nan", "none", "-", "null"])
    if not valid.any():
        return None
    num = pd.to_numeric(txt.map(_normalize_number), errors="coerce")
    return num if num[valid].notna().mean() >= 0.9 else None


def _to_date(s):
    try:
        d = pd.to_datetime(s, errors="coerce", dayfirst=True, format="mixed")
    except (TypeError, ValueError):
        d = pd.to_datetime(s, errors="coerce", dayfirst=True)
    nonnull = _as_text(s).str.strip().ne("")
    return d if nonnull.any() and d[nonnull].notna().mean() >= 0.8 else None


def prepare_dataset(raw):
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all").reset_index(drop=True)
    uw_cols = [c for c in df.columns if _UW_COL.search(c)]
    for c in list(df.columns):
        if c in uw_cols:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[f"{c}_tahun"] = df[c].dt.year.astype("Int64")
            continue
        if not _is_text(df[c]):
            continue
        if _DATE_HINT.search(c):
            d = _to_date(df[c])
            if d is not None:
                df[c] = d
                df[f"{c}_tahun"] = d.dt.year.astype("Int64")
                continue
        if not _ID_HINT.search(c):
            n = _to_number(df[c])
            if n is not None:
                df[c] = n
    if uw_cols:
        df[UW_LIST] = df[uw_cols].apply(
            lambda r: list(dict.fromkeys(sum((parse_underwriters(v) for v in r), []))), axis=1)
        df[UW_NAMES] = df[UW_LIST].apply("; ".join)
    return df, uw_cols


def _read_csv(path):
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_eipo():
    """Kembalikan (df, info). Prioritas: EIPO_CSV lokal -> unduh dataset Kaggle."""
    if EIPO_CSV:
        files = [EIPO_CSV]
    else:
        import kagglehub
        root = kagglehub.dataset_download(EIPO_DATASET)
        files = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
        if EIPO_FILE:
            files = [f for f in files if os.path.basename(f) == EIPO_FILE] or files
    if not files:
        raise FileNotFoundError("Tidak ada file CSV pada dataset.")
    path = max(files, key=os.path.getsize)          # bila banyak file, ambil yang terbesar
    df, uw_cols = prepare_dataset(_read_csv(path))
    return df, {"file": os.path.basename(path), "all_files": [os.path.basename(f) for f in files],
                "uw_cols": uw_cols}


def find_company_rows(df):
    """Baris dataset untuk perusahaan dokumen utama (DOC_TICKER persis, lalu DOC_KEYWORD)."""
    text_cols = [c for c in df.columns if c != UW_LIST and _is_text(df[c])]
    if DOC_TICKER:
        for c in text_cols:
            if re.search(r"ticker|kode|code|symbol", c, re.I):
                hit = df[_as_text(df[c]).str.strip().str.upper() == DOC_TICKER.upper()]
                if not hit.empty:
                    return hit
    if DOC_KEYWORD:
        mask = pd.Series(False, index=df.index)
        for c in text_cols:
            mask |= _as_text(df[c]).str.contains(DOC_KEYWORD, case=False, regex=False, na=False)
        return df[mask]
    return df.iloc[0:0]


def _fmt(v):
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, numbers.Real) and float(v).is_integer():
        return str(int(v)) if abs(v) < 10000 else f"{int(v):,}".replace(",", ".")
    return str(v)


def row_to_text(row):
    return "\n".join(f"{k}: {_fmt(v)}" for k, v in row.items()
                     if k != UW_LIST and not (pd.isna(v) if not isinstance(v, list) else False)
                     and str(v).strip() != "")


def dataset_profile(df, max_examples=6):
    lines = [f"Jumlah baris: {len(df)}"]
    for c in df.columns:
        if c == UW_LIST:
            continue
        s = df[c]
        filled = int(s.notna().sum())
        if pd.api.types.is_bool_dtype(s):
            desc = "boolean"
        elif pd.api.types.is_numeric_dtype(s):
            desc = f"angka, min={_fmt(s.min())}, max={_fmt(s.max())}" if filled else "angka"
        elif pd.api.types.is_datetime64_any_dtype(s):
            desc = f"tanggal, {_fmt(s.min())} s.d. {_fmt(s.max())}" if filled else "tanggal"
        else:
            vc = s.dropna().astype(str).value_counts()
            ex = ", ".join(v[:40] for v in vc.index[:max_examples])
            desc = f"teks, {s.nunique()} nilai unik, contoh: {ex}"
        lines.append(f"- {c} ({desc}; terisi {filled})")
    if UW_LIST in df.columns:
        lines.append(f"- {UW_SINGLE} (VIRTUAL: satu nama penjamin emisi per baris; "
                     f"hanya tersedia jika explode_underwriter=true)")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Eksekusi query dataset yang aman (whitelist, tanpa eval)
# ══════════════════════════════════════════════════════════════════════════════
OPS = {"eq", "ne", "gt", "gte", "lt", "lte", "contains", "not_contains", "in", "between", "isnull", "notnull"}
FUNCS = {"count", "sum", "mean", "median", "min", "max", "nunique"}


def _coerce(series, v):
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(v, errors="coerce", dayfirst=True)
    if pd.api.types.is_numeric_dtype(series):
        try:
            return float(v)
        except (TypeError, ValueError):
            return v
    return v


def run_query(df, q):
    d = df
    if q.get("explode_underwriter") and UW_LIST in d.columns:
        d = d.explode(UW_LIST).rename(columns={UW_LIST: UW_SINGLE})
    cols = set(d.columns)

    def need(c):
        if c not in cols:
            raise ValueError(f"kolom tidak dikenal: {c}")
        return c

    for f in q.get("filters") or []:
        c, op, v = need(f.get("column")), f.get("op", "eq"), f.get("value")
        if op not in OPS:
            raise ValueError(f"operasi tidak didukung: {op}")
        s = d[c]
        is_text = not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s))
        if op == "contains":
            m = _as_text(s).str.contains(str(v), case=False, regex=False, na=False)
        elif op == "not_contains":
            m = ~_as_text(s).str.contains(str(v), case=False, regex=False, na=False)
        elif op == "in":
            vals = v if isinstance(v, list) else [v]
            m = _as_text(s).str.lower().isin([str(x).lower() for x in vals]) if is_text \
                else s.isin([_coerce(s, x) for x in vals])
        elif op == "between":
            lo, hi = (_coerce(s, x) for x in (v or [None, None])[:2])
            m = s.between(lo, hi)
        elif op == "isnull":
            m = s.isna()
        elif op == "notnull":
            m = s.notna()
        elif is_text and op in ("eq", "ne"):
            m = _as_text(s).str.strip().str.lower() == str(v).strip().lower()
            m = m if op == "eq" else ~m
        else:
            val = _coerce(s, v)
            m = {"eq": s == val, "ne": s != val, "gt": s > val, "gte": s >= val,
                 "lt": s < val, "lte": s <= val}[op]
        d = d[m.fillna(False)]

    matched = len(d)
    group = [need(c) for c in q.get("group_by") or []]
    metrics = q.get("metrics") or []

    def values_for(col, func):
        vals = d[need(col)]
        if func in ("sum", "mean", "median") and not pd.api.types.is_numeric_dtype(vals):
            vals = pd.to_numeric(vals, errors="coerce")
        return vals

    if group:
        keys = [d[c] for c in group]
        parts = []
        for mt in metrics or [{"column": "*", "func": "count", "alias": "jumlah_ipo"}]:
            func, col = mt.get("func", "count"), mt.get("column") or "*"
            if func not in FUNCS:
                raise ValueError(f"fungsi tidak didukung: {func}")
            alias = mt.get("alias") or f"{func}_{col}"
            if col == "*":
                parts.append(d.groupby(keys, dropna=True).size().rename(alias))
            else:
                parts.append(values_for(col, func).groupby(keys, dropna=True).agg(func).rename(alias))
        res = pd.concat(parts, axis=1).reset_index()
    elif metrics:
        row = {}
        for mt in metrics:
            func, col = mt.get("func", "count"), mt.get("column") or "*"
            if func not in FUNCS:
                raise ValueError(f"fungsi tidak didukung: {func}")
            alias = mt.get("alias") or f"{func}_{col}"
            row[alias] = len(d) if col == "*" else values_for(col, func).agg(func)
        res = pd.DataFrame([row])
    else:
        sel = [need(c) for c in q.get("select") or []] or [c for c in d.columns if c != UW_LIST]
        res = d[sel]

    sort_by = q.get("sort_by")
    if sort_by in res.columns:
        res = res.sort_values(sort_by, ascending=bool(q.get("ascending", False)), na_position="last")
    limit = max(1, min(int(q.get("limit") or 20), 200))
    return res.head(limit).reset_index(drop=True), matched


def table_to_text(title, res, matched):
    t = res.head(MAX_TABLE_ROWS_CONTEXT).copy()
    for c in t.columns:
        if _is_text(t[c]):
            t[c] = t[c].astype(str).str.slice(0, 120)
    body = t.to_csv(index=False, date_format="%Y-%m-%d")
    return f"### {title}\n(baris cocok: {matched}; ditampilkan: {len(t)})\n{body}"


# ══════════════════════════════════════════════════════════════════════════════
# Backend RAG
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Menyiapkan asisten…")
def initialize_rag():
    api_key = get_config("DEEPSEEK_API_KEY")
    supabase_url = get_config("SUPABASE_URL")
    supabase_key = (
        get_config("SUPABASE_SERVICE_ROLE_KEY")
        or get_config("SUPABASE_KEY")
        or get_config("SUPABASE_ANON_KEY")
    )

    missing = [k for k, v in {"DEEPSEEK_API_KEY": api_key, "SUPABASE_URL": supabase_url,
                              "SUPABASE_KEY": supabase_key}.items() if not v]
    if missing:
        raise RuntimeError("Konfigurasi belum lengkap: " + ", ".join(missing))

    class E5Embeddings(Embeddings):
        def __init__(self, base): self.base = base
        def embed_documents(self, texts): return self.base.embed_documents([f"passage: {t}" for t in texts])
        def embed_query(self, text): return self.base.embed_query(f"query: {text}")

    base = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"},
                                 encode_kwargs={"normalize_embeddings": True, "batch_size": 16})
    client = create_client(supabase_url, supabase_key)
    store = SupabaseVectorStore(client=client, embedding=E5Embeddings(base),
                                table_name=TABLE_NAME, query_name=QUERY_NAME)
    llm = ChatOpenAI(model=DEEPSEEK_MODEL, temperature=0, max_tokens=1400, api_key=api_key,
                     base_url=get_config("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    return store, llm


def _parse_json(raw):
    match = re.search(r"\{.*\}", raw or "", re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def llm_json(llm, system, human):
    msgs = [SystemMessage(content=system), HumanMessage(content=human)]
    try:
        raw = llm.bind(response_format={"type": "json_object"}).invoke(msgs).content
    except Exception:
        raw = llm.invoke(msgs).content
    return _parse_json(raw)


# ── Planner ───────────────────────────────────────────────────────────────────
PLANNER_SYSTEM = """Anda adalah perencana query untuk asisten prospektus IPO Indonesia.
Tentukan sumber jawaban dan, bila perlu, susun query terstruktur atas dataset e-IPO.

Sumber:
- "prospektus": detail tentang <<DOC>> (perusahaan dokumen utama): harga, jadwal, penggunaan dana, risiko, operasional, keuangan, manajemen, pemegang saham.
- "dataset": statistik atau perbandingan antar-IPO di Indonesia dari dataset e-IPO (jumlah IPO, penjamin emisi teraktif, rata-rata/tertinggi/terendah suatu nilai, IPO per tahun, daftar IPO tertentu).
- "keduanya": membandingkan <<DOC>> dengan IPO lain, atau butuh data dari kedua sumber.

Skema dataset e-IPO:
<<PROFILE>>

Baris dataset untuk <<DOC>> (bisa kosong):
<<COMPANY>>

Balas HANYA JSON dengan bentuk:
{"sumber": "prospektus|dataset|keduanya",
 "queries": [{"tujuan": "deskripsi singkat",
   "filters": [{"column": "...", "op": "eq|ne|gt|gte|lt|lte|contains|not_contains|in|between|isnull|notnull", "value": ...}],
   "explode_underwriter": false,
   "group_by": ["..."],
   "metrics": [{"column": "* atau nama_kolom", "func": "count|sum|mean|median|min|max|nunique", "alias": "..."}],
   "select": ["..."],
   "sort_by": "...", "ascending": false, "limit": 20}]}

Aturan:
- Pakai hanya nama kolom yang ada di skema, persis sama.
- "queries" = [] jika sumber "prospektus". Maksimal 3 query.
- Statistik per penjamin emisi: explode_underwriter=true dan group_by ["penjamin_emisi"].
- IPO yang ditangani sekuritas tertentu: filter "contains" pada kolom "penjamin_emisi_nama" dengan nama sekuritasnya (tanpa "PT").
- Jika pertanyaan menyangkut penjamin emisi <<DOC>> dan barisnya ada di atas, gunakan nama penjamin dari baris itu.
- Daftar baris: isi "select" dengan maksimal 8 kolom relevan; sort_by kolom yang sesuai.
- Untuk group_by, urutkan menurut alias metrik (sort_by = alias)."""


def plan_question(llm, question, history, profile, company_text):
    system = (PLANNER_SYSTEM.replace("<<DOC>>", DOC_NAME).replace("<<PROFILE>>", profile)
              .replace("<<COMPANY>>", company_text or "(tidak ditemukan)"))
    human = f"Riwayat percakapan:\n{history}\n\nPertanyaan: {question}\n\nKeluarkan JSON."
    plan = llm_json(llm, system, human) or {}
    sumber = plan.get("sumber") if plan.get("sumber") in ("prospektus", "dataset", "keduanya") else "prospektus"
    queries = [q for q in (plan.get("queries") or []) if isinstance(q, dict)][:MAX_QUERIES]
    if sumber != "prospektus" and not queries:
        sumber = "prospektus"
    return sumber, queries


# ── Jawaban ───────────────────────────────────────────────────────────────────
ANSWER_SYSTEM = f"""Anda adalah IPO Insight AI, asisten analisis prospektus IPO Indonesia.
Anda menerima dua jenis sumber:
1) KUTIPAN PROSPEKTUS {DOC_NAME}.
2) HASIL DATA e-IPO: tabel hasil query atas dataset e-IPO publik (berbagai IPO di Indonesia).
Aturan:
- Jawab hanya berdasarkan sumber yang diberikan. Bahasa Indonesia yang profesional, jelas, ringkas. Langsung ke jawaban inti.
- Salin angka, tanggal, persentase, nama, dan satuan persis seperti di sumber.
- Untuk fakta {DOC_NAME}, prospektus diutamakan; bila data e-IPO berbeda, sebutkan perbedaannya.
- Untuk statistik pasar IPO, gunakan tabel data e-IPO dan sebut bahwa angkanya berasal dari data e-IPO. Perhatikan jumlah baris cocok; jangan mengklaim lengkap bila data terpotong.
- Gunakan poin atau tabel Markdown bila membantu keterbacaan.
- Jika informasi tidak didukung sumber, jawab: "{NOT_FOUND}"
- Jangan mengarang. Bedakan fakta dari interpretasi. Jangan menyebut istilah teknis seperti chunk, retrieval, atau query.
- Jika relevan, tutup dengan satu baris "Rujukan:" berisi bagian prospektus dan/atau "data e-IPO".
- Riwayat percakapan hanya untuk memahami pertanyaan lanjutan, bukan sumber fakta.
- Ini bukan nasihat investasi."""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", ANSWER_SYSTEM.replace("{", "{{").replace("}", "}}")),
    ("human", "RIWAYAT PERCAKAPAN:\n{history}\n\nKUTIPAN PROSPEKTUS:\n{context}\n\n"
              "HASIL DATA e-IPO:\n{data}\n\nPERTANYAAN:\n{question}"),
])


def format_context(docs):
    if not docs:
        return "(tidak digunakan)"
    return "\n\n---\n\n".join(
        f"[Bagian: {d.metadata.get('heading_path', 'Tidak tercantum')}]\n{d.page_content}" for d in docs)


def format_history(messages):
    recent = messages[-HISTORY_TURNS * 2:]
    if not recent:
        return "(belum ada)"
    return "\n".join(
        f"{'Pengguna' if m['role'] == 'user' else 'Asisten'}: "
        f"{m['content'][:600] + '…' if len(m['content']) > 600 else m['content']}" for m in recent)


def build_search_query(question, messages):
    prev = [m["content"] for m in messages if m["role"] == "user"]
    return f"{prev[-1]} {question}" if prev and len(question.split()) <= 6 else question


def retrieve(store, query):
    return store.similarity_search(query, k=TOP_K, filter=VECTORSTORE_FILTER)


def stream_answer(llm, docs, data_text, question, history):
    chain = PROMPT | llm | StrOutputParser()
    return chain.stream({"context": format_context(docs), "data": data_text or "(tidak digunakan)",
                         "question": question, "history": history})


def extract_sources(docs):
    seen, sources = set(), []
    for d in docs:
        heading = d.metadata.get("heading_path") or "Bagian tidak bernama"
        if heading in seen:
            continue
        seen.add(heading)
        text = " ".join(d.page_content.split())
        sources.append({"heading": heading, "snippet": text[:280] + ("…" if len(text) > 280 else "")})
        if len(sources) >= MAX_SOURCES_SHOWN:
            break
    return sources


def render_sources(sources, tables=None):
    if sources:
        with st.expander(f"Bagian prospektus yang dirujuk ({len(sources)})"):
            for s in sources:
                st.markdown(f"**{s['heading']}**")
                st.caption(s["snippet"])
    if tables:
        with st.expander(f"Data e-IPO yang digunakan ({len(tables)} tabel)"):
            for t in tables:
                st.markdown(f"**{t['title']}** · {t['matched']} baris cocok")
                st.dataframe(t["df"], hide_index=True, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# Ringkasan fakta perusahaan
# ══════════════════════════════════════════════════════════════════════════════
FACT_QUERIES = [
    "Nama lengkap Perseroan dan kegiatan usaha utama perusahaan",
    "Harga penawaran IPO, nilai nominal saham, jumlah saham ditawarkan, persentase saham",
    "Jadwal masa penawaran umum dan tanggal pencatatan saham",
    "Penggunaan dana hasil IPO dan penjamin pelaksana emisi efek",
    "Faktor risiko utama yang diungkapkan Perseroan",
]
FACT_SYSTEM = """Ekstrak ringkasan fakta perusahaan dari kutipan prospektus. Data e-IPO hanya pelengkap bila prospektus tidak memuatnya.
Keluarkan HANYA JSON valid dengan schema:
{"nama_perusahaan":"...","ticker":"...","kegiatan_usaha":"...","harga_penawaran":"...","nilai_nominal":"...","jumlah_saham":"...","jadwal_ipo":"...","penggunaan_dana":"...","penjamin_emisi":["..."],"risiko_utama":["..."]}
Gunakan "Tidak ditemukan" untuk nilai yang tidak tersedia. Jangan menebak ticker atau fakta."""


@st.cache_data(show_spinner=False)
def get_company_facts(_store, _llm, company_text):
    all_docs, seen = [], set()
    for q in FACT_QUERIES:
        for d in _store.similarity_search(q, k=4, filter=VECTORSTORE_FILTER):
            key = d.metadata.get("chunk_id") or d.page_content[:160]
            if key not in seen:
                seen.add(key)
                all_docs.append(d)
    if not all_docs and not company_text:
        return None
    human = f"KUTIPAN PROSPEKTUS:\n{format_context(all_docs)}\n\nDATA e-IPO:\n{company_text or '(tidak ada)'}"
    return llm_json(_llm, FACT_SYSTEM, human)


def clean(value):
    if isinstance(value, list):
        value = [v for v in value if clean(v) != "—"]
        return value or "—"
    if not value or str(value).strip().lower().startswith("tidak ditemukan"):
        return "—"
    return str(value).strip()


def render_facts(data):
    company = clean(data.get("nama_perusahaan"))
    if company != "—":
        st.markdown(f"#### {company}")
    stats = [("Ticker", data.get("ticker")), ("Harga penawaran", data.get("harga_penawaran")),
             ("Nilai nominal", data.get("nilai_nominal")), ("Jumlah saham", data.get("jumlah_saham"))]
    cards = "".join(f'<div class="stat"><div class="label">{label}</div>'
                    f'<div class="value">{html.escape(str(clean(v)))}</div></div>' for label, v in stats)
    st.markdown(f'<div class="stat-grid">{cards}</div>', unsafe_allow_html=True)
    for label, key in [("Kegiatan usaha", "kegiatan_usaha"), ("Jadwal IPO", "jadwal_ipo"),
                       ("Penggunaan dana", "penggunaan_dana")]:
        val = clean(data.get(key))
        if val != "—":
            st.markdown(f"**{label}**  \n{val}")
    underwriters = clean(data.get("penjamin_emisi"))
    if underwriters != "—":
        st.markdown("**Penjamin emisi**  \n" + ", ".join(underwriters))
    risks = clean(data.get("risiko_utama"))
    if risks != "—":
        st.markdown("**Risiko utama**")
        st.markdown("\n".join(f"- {r}" for r in risks[:3]))


# ══════════════════════════════════════════════════════════════════════════════
# Halaman: Tanya jawab
# ══════════════════════════════════════════════════════════════════════════════
def answer_pipeline(store, llm, question, messages, eipo, company_text):
    """Kembalikan (generator jawaban | str, sources, tables)."""
    history = format_history(messages)
    sumber, queries = "prospektus", []
    if eipo is not None:
        with st.spinner("Memahami pertanyaan…"):
            sumber, queries = plan_question(llm, question, history, dataset_profile(eipo), company_text)

    tables, data_parts = [], []
    if sumber in ("dataset", "keduanya"):
        with st.spinner("Mengolah data e-IPO…"):
            for q in queries:
                try:
                    res, matched = run_query(eipo, q)
                    title = q.get("tujuan") or "Hasil data e-IPO"
                    tables.append({"title": title, "df": res, "matched": matched})
                    data_parts.append(table_to_text(title, res, matched))
                except Exception as e:
                    print(f"[IPO Insight AI] query dataset gagal: {e} | {q}")
        if not tables:
            sumber = "prospektus"   # fallback bila semua query gagal

    if company_text:
        data_parts.insert(0, f"### Baris data e-IPO untuk {DOC_NAME}\n{company_text}")

    docs = []
    if sumber in ("prospektus", "keduanya"):
        with st.spinner("Mencari di prospektus…"):
            docs = retrieve(store, build_search_query(question, messages))

    if not docs and not tables:
        return NOT_FOUND, [], []
    stream = stream_answer(llm, docs, "\n\n".join(data_parts), question, history)
    return stream, extract_sources(docs), tables


def render_chat_page(store, llm, eipo, company_text):
    with st.expander("📋 Ringkasan fakta perusahaan", expanded=False):
        if st.session_state.get("facts_requested"):
            with st.spinner("Merangkum fakta dari prospektus…"):
                try:
                    facts = get_company_facts(store, llm, company_text)
                except Exception:
                    facts = None
            if facts:
                render_facts(facts)
                if st.button("Perbarui ringkasan", key="refresh_facts"):
                    get_company_facts.clear()
                    st.rerun()
            else:
                st.info("Ringkasan belum dapat dibentuk. Anda tetap bisa bertanya langsung di bawah.")
                if st.button("Coba lagi", key="retry_facts"):
                    get_company_facts.clear()
                    st.rerun()
        else:
            st.caption("Ringkasan singkat: harga penawaran, jumlah saham, jadwal, penggunaan dana, dan risiko.")
            if st.button("Buat ringkasan", type="primary", key="make_facts"):
                st.session_state.facts_requested = True
                st.rerun()

    typed = st.chat_input("Tanyakan tentang prospektus atau data IPO…")
    question = typed or st.session_state.pop("pending_question", None)
    messages = st.session_state.messages

    if messages:
        c1, c2 = st.columns([3, 1])
        c1.markdown("##### Percakapan")
        if c2.button("↺ Percakapan baru", key="reset_chat", width="stretch"):
            st.session_state.messages = []
            st.rerun()

    if not messages and not question:
        st.markdown('<div class="empty-hint">Mulai dengan salah satu pertanyaan berikut, '
                    'atau ketik pertanyaan Anda sendiri.</div>', unsafe_allow_html=True)
        quick = QUICK_QUESTIONS if eipo is not None else QUICK_QUESTIONS[:4]
        cols = st.columns(2)
        for i, q in enumerate(quick):
            if cols[i % 2].button(q, key=f"quick_{i}", width="stretch"):
                st.session_state.pending_question = q
                st.rerun()

    for m in messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                render_sources(m.get("sources"), m.get("tables"))

    if question:
        messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            sources, tables = [], []
            try:
                result, sources, tables = answer_pipeline(store, llm, question, messages[:-1], eipo, company_text)
                if isinstance(result, str):
                    answer = result
                    st.markdown(answer)
                else:
                    answer = st.write_stream(result)
                render_sources(sources, tables)
            except Exception as e:
                answer = "Maaf, terjadi kendala saat memproses pertanyaan. Silakan coba lagi."
                st.error(answer)
                print(f"[IPO Insight AI] error: {e}")
        messages.append({"role": "assistant", "content": answer, "sources": sources, "tables": tables})


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
  <h1>IPO Insight AI</h1>
  <p>Tanyakan apa saja tentang prospektus IPO dan data IPO Indonesia.</p>
  <span class="doc">📄 {html.escape(DOC_NAME)}</span>
</div>
""", unsafe_allow_html=True)

try:
    store, llm = initialize_rag()
except Exception as e:
    st.error("Layanan belum siap. Periksa pengaturan environment/secrets aplikasi.")
    with st.expander("Detail"):
        st.code(str(e))
    st.stop()

eipo = None
try:
    with st.spinner("Memuat data e-IPO…"):
        eipo, _ = load_eipo()
except Exception:
    import traceback
    print(f"[IPO Insight AI] data e-IPO tidak dimuat (chat tetap memakai prospektus):\n{traceback.format_exc()}")

company_rows = find_company_rows(eipo) if eipo is not None else pd.DataFrame()
company_text = row_to_text(company_rows.iloc[0]) if not company_rows.empty else ""

if "messages" not in st.session_state:
    st.session_state.messages = []

render_chat_page(store, llm, eipo, company_text)

st.markdown('<div class="disclaimer">Informasi berbasis prospektus dan data e-IPO publik, '
            'bukan rekomendasi atau nasihat investasi.</div>', unsafe_allow_html=True)