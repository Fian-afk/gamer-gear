import sqlite3
import os

# Pastikan folder 'data' tersedia
os.makedirs('data', exist_ok=True)

# Membuat atau terhubung ke database SQLite
conn = sqlite3.connect('data/gamergear.db')
cursor = conn.cursor()

# 1. Tabel laptops
cursor.execute('''
CREATE TABLE IF NOT EXISTS laptops (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nama TEXT NOT NULL,
    merek TEXT,
    cpu TEXT,
    gpu TEXT,
    ram_gb INTEGER,
    ssd_gb INTEGER,
    layar_inci REAL,
    refresh_rate INTEGER,
    harga_listing INTEGER,
    harga_prediksi INTEGER,
    skor_value REAL,
    tier_performa REAL
)
''')

# 2. Tabel peripherals
cursor.execute('''
CREATE TABLE IF NOT EXISTS peripherals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kategori TEXT,
    nama TEXT NOT NULL,
    merek TEXT,
    spesifikasi TEXT,
    harga INTEGER
)
''')

# 3. Tabel reviews
cursor.execute('''
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    id_sumber TEXT,
    produk_id INTEGER,
    sumber TEXT,
    teks_asli TEXT,
    teks_bersih TEXT,
    tanggal TEXT,
    FOREIGN KEY (produk_id) REFERENCES laptops (id)
)
''')
review_columns = {row[1] for row in cursor.execute("PRAGMA table_info(reviews)")}
if 'id_sumber' not in review_columns:
    cursor.execute('ALTER TABLE reviews ADD COLUMN id_sumber TEXT')
cursor.execute('''
CREATE UNIQUE INDEX IF NOT EXISTS idx_reviews_source_id
ON reviews (sumber, id_sumber)
''')

# 4. Tabel sentiment_results
cursor.execute('''
CREATE TABLE IF NOT EXISTS sentiment_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id INTEGER,
    aspek TEXT,
    label_sentimen TEXT,
    skor REAL,
    FOREIGN KEY (review_id) REFERENCES reviews (id)
)
''')

# 5. Tabel complaint_clusters
cursor.execute('''
CREATE TABLE IF NOT EXISTS complaint_clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_no INTEGER,
    label TEXT,
    kata_kunci TEXT,
    jumlah_ulasan INTEGER
)
''')

# 6. Tabel game_requirements
cursor.execute('''
CREATE TABLE IF NOT EXISTS game_requirements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nama_game TEXT NOT NULL,
    min_cpu_tier INTEGER,
    min_gpu_tier INTEGER,
    min_ram_gb INTEGER,
    rec_cpu_tier INTEGER,
    rec_gpu_tier INTEGER,
    rec_ram_gb INTEGER
)
''')

# 7. Tabel chat_logs
cursor.execute('''
CREATE TABLE IF NOT EXISTS chat_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    waktu TEXT,
    pertanyaan TEXT,
    intent TEXT,
    dijawab BOOLEAN
)
''')

# 8. Tabel unanswered_queries
cursor.execute('''
CREATE TABLE IF NOT EXISTS unanswered_queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    waktu TEXT,
    pertanyaan TEXT,
    intent_terdekat TEXT,
    status_tinjauan TEXT DEFAULT 'pending'
)
''')

conn.commit()
conn.close()
print("Database data/gamergear.db dan seluruh tabel berhasil dibuat!")