# IPO Insight AI

Asisten tanya jawab prospektus IPO Indonesia berbasis **Streamlit**, **DeepSeek**, dan **Supabase pgvector**, diperkaya data pasar IPO dari dataset Kaggle **e-IPO**.

Pengguna cukup bertanya lewat satu kolom chat. Aplikasi menentukan sendiri apakah jawaban diambil dari prospektus, dari data e-IPO, atau keduanya.

## Fitur

- **Tanya jawab prospektus**: jawaban bersumber dari potongan prospektus yang tersimpan di Supabase, lengkap dengan bagian dokumen yang dirujuk.
- **Data e-IPO**: pertanyaan statistik (penjamin emisi teraktif, jumlah IPO per tahun, perbandingan antar-IPO) dijawab dari dataset Kaggle `fahmirk/e-ipo-realtime-data`. Tabel yang dipakai ditampilkan di bawah jawaban.
- **Kode penjamin emisi → nama lengkap**: kamus kode Anggota Bursa BEI bawaan.
- **Ringkasan fakta perusahaan**: harga penawaran, jumlah saham, jadwal, penggunaan dana, dan risiko.
- **Jawaban streaming**, riwayat percakapan, dan pertanyaan lanjutan.
- Tampilan aman untuk **mode terang dan gelap**.

## Struktur

```
.
├── app.py                          # aplikasi Streamlit
├── requirements.txt
├── .env.example                    # contoh konfigurasi lokal
├── .streamlit/secrets.toml.example # contoh secrets untuk Streamlit Cloud
└── .gitignore
```

## Menjalankan secara lokal

```bash
git clone https://github.com/<username>/ipo-insight-ai.git
cd ipo-insight-ai
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
# isi .env dengan kredensial Anda
streamlit run app.py
```

Saat pertama dijalankan, model embedding (`intfloat/multilingual-e5-base`) dan dataset e-IPO diunduh otomatis, lalu disimpan di cache.

## Konfigurasi

| Variabel | Wajib | Keterangan |
|---|---|---|
| `DEEPSEEK_API_KEY` | ✅ | API key DeepSeek |
| `DEEPSEEK_BASE_URL` | | Default `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | | Default `deepseek-v4-flash` |
| `SUPABASE_URL` | ✅ | URL project Supabase |
| `SUPABASE_KEY` | ✅ | Key Supabase (atau `SUPABASE_SERVICE_ROLE_KEY`) |
| `SUPABASE_TABLE` | | Default `documents` |
| `SUPABASE_QUERY_NAME` | | RPC pencarian, default `match_documents` |
| `RAG_TOP_K` | | Jumlah potongan yang diambil, default `10` |
| `DOC_NAME` | | Nama perusahaan prospektus |
| `DOC_KEYWORD` | | Kata kunci untuk menemukan perusahaan di data e-IPO |
| `DOC_TICKER` | | Kode saham perusahaan (opsional) |
| `EIPO_DATASET` | | Dataset Kaggle, default `fahmirk/e-ipo-realtime-data` |
| `EIPO_CSV` | | Path CSV lokal sebagai pengganti unduhan Kaggle |
| `EIPO_FILE` | | Nama file tertentu bila dataset berisi beberapa CSV |
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | | Hanya bila kagglehub meminta login |

Konfigurasi dibaca dari **Streamlit Secrets** terlebih dahulu, lalu dari environment / `.env`.

## Deploy ke Streamlit Community Cloud

1. Push repo ini ke GitHub.
2. Di [share.streamlit.io](https://share.streamlit.io), pilih **Create app**, lalu pilih repo dan file `app.py`.
3. Buka **Advanced settings → Secrets**, lalu tempel isi `.streamlit/secrets.toml.example` dengan nilai asli.
4. Deploy.

## Prasyarat database

Supabase harus sudah berisi potongan prospektus dengan metadata `vectorstore_name = "structure_aware"` dan fungsi RPC `match_documents`. Aplikasi ini hanya membaca vectorstore yang sudah ada dan tidak membangun ulang indeks.

## Keamanan

- **Jangan pernah commit `.env`** atau `secrets.toml`. Keduanya sudah tercantum di `.gitignore`.
- Bila key pernah terekspos (dibagikan, ter-commit, atau dikirim lewat chat), **rotasi** key DeepSeek dan Supabase di dashboard masing-masing.
- Service-role key Supabase melewati Row Level Security. Gunakan hanya di sisi server.

## Catatan

Informasi yang dihasilkan berbasis prospektus dan data e-IPO publik, **bukan rekomendasi atau nasihat investasi**. Kamus kode penjamin emisi disusun dari daftar kode broker BEI (2023–2025). Kode yang belum dikenal akan ditandai "kode tidak dikenal".
