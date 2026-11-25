import os
import glob
import re
import json
import joblib
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

# ==============================================================================
# 0. INISIALISASI RESOURCE & KAMUS
# ==============================================================================

# Pastikan nltk resource terunduh
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Stopwords & Stemmer Bahasa Inggris
STOPWORDS_EN = set(stopwords.words('english'))
STEMMER_EN = PorterStemmer()

# Stopwords & Stemmer Bahasa Indonesia (Sastrawi)
stop_factory = StopWordRemoverFactory()
STOPWORDS_ID = set(stop_factory.get_stop_words())
# Tambahan stopword percakapan online & YouTube review
CUSTOM_STOPWORDS_ID = {
    'yg', 'dg', 'dgn', 'ny', 'd', 'k', 'sy', 'gw', 'gua', 'lu', 'lo', 
    'nih', 'tuh', 'aja', 'deh', 'sih', 'nya', 'dong', 'kok', 'lah', 
    'kan', 'ku', 'mu', 'si', 'ya', 'yah', 'halo', 'hai', 'bang', 'mas', 
    'kak', 'gan', 'bro', 'min', 'om', 'pak', 'guys', 'gaes', 'kawan', 'kawan2'
}
STOPWORDS_ID.update(CUSTOM_STOPWORDS_ID)

stem_factory = StemmerFactory()
STEMMER_ID = stem_factory.create_stemmer()

# Cache in-memory untuk Sastrawi Stemmer
SASTRAWI_STEM_CACHE = {}

def get_stem_id(word: str) -> str:
    """Stem kata Bahasa Indonesia menggunakan Sastrawi dengan in-memory cache."""
    if word not in SASTRAWI_STEM_CACHE:
        SASTRAWI_STEM_CACHE[word] = STEMMER_ID.stem(word)
    return SASTRAWI_STEM_CACHE[word]


# ==============================================================================
# KAMUS NORMALISASI BAHASA GAUL (LANGKAH 2)
# ==============================================================================
SLANG_MULTIWORD_MAP = {
    "panas bgt": "panas banget",
    "panas bngt": "panas banget",
    "panas bener": "panas banget",
    "fps drop": "penurunan frame rate",
    "frame drop": "penurunan frame rate",
    "worth it": "sepadan",
    "worth to buy": "sepadan untuk dibeli",
    "worth it lah": "sepadan",
    "rata kanan": "pengaturan grafis maksimal",
    "rata kiri": "pengaturan grafis minimal",
    "gak bisa": "tidak bisa",
    "ga bisa": "tidak bisa",
    "bisa bgt": "bisa banget",
    "drop bgt": "turun drastis",
    "lemot bgt": "lambat banget",
    "lelet bgt": "lambat banget",
    "boros bgt": "boros banget",
    "berisik bgt": "berisik banget",
    "suara mendem": "suara teredam",
    "mendem bgt": "teredam banget",
    "backlight bleed": "kebocoran cahaya layar"
}

SLANG_WORD_MAP = {
    # Kata ganti & penunjuk
    "gw": "saya", "gua": "saya", "gue": "saya", "w": "saya", "sy": "saya",
    "lu": "kamu", "lo": "kamu", "elu": "kamu", "u": "kamu",
    "doi": "dia", "dy": "dia",
    "nih": "ini", "tuh": "itu", "gini": "seperti ini", "gitu": "seperti itu",
    
    # Kata sambung & preposisi
    "yg": "yang", "yng": "yang",
    "dg": "dengan", "dgn": "dengan", "sma": "sama", "sm": "sama",
    "krn": "karena", "karna": "karena", "krna": "karena", "coz": "karena",
    "utk": "untuk", "untk": "untuk", "buat": "untuk",
    "klo": "kalau", "kalo": "kalau", "kl": "kalau", "klw": "kalau",
    "tp": "tetapi", "tpi": "tetapi", "tapi": "tetapi",
    "dr": "dari", "dri": "dari",
    "d": "di", "ke": "ke",
    "atw": "atau",
    "trs": "terus", "trus": "terus",
    
    # Negasi & partikel
    "ga": "tidak", "gak": "tidak", "ngga": "tidak", "nggak": "tidak", "g": "tidak", "tdk": "tidak", "tak": "tidak",
    "bkn": "bukan",
    "blm": "belum", "blom": "belum",
    "udh": "sudah", "udah": "sudah", "sdh": "sudah", "dah": "sudah",
    "aja": "saja", "aj": "saja", "doang": "saja",
    "bgt": "banget", "bngt": "banget", "bgtt": "banget", "bnget": "banget",
    "plis": "tolong", "pliss": "tolong", "please": "tolong",
    
    # Istilah hardware & review
    "batre": "baterai", "batere": "baterai", "batrai": "baterai", "battery": "baterai",
    "boros": "boros", "awet": "awet",
    "berisik": "berisik", "brisik": "berisik",
    "mendem": "teredam",
    "kenceng": "kencang", "ngebut": "cepat", "kencang": "kencang",
    "lemot": "lambat", "lelet": "lambat",
    "ngelag": "macet", "lag": "macet", "stutter": "tersendat", "patah": "patah",
    "adem": "dingin", "anget": "hangat",
    "bodi": "bodi", "body": "bodi", "kokoh": "kokoh", "ringkih": "ringkih",
    "layar": "layar", "screen": "layar",
    "terang": "cerah", "silau": "silau",
    "nyesel": "menyesal", "kecewa": "kecewa",
    "worth": "sepadan", "worthed": "sepadan",
    "overprice": "terlalu mahal", "overpriced": "terlalu mahal",
    "rekomen": "rekomendasi", "recomended": "rekomendasi", "recommended": "rekomendasi",
    "mantap": "mantap", "mantul": "mantap", "jos": "bagus", "josjis": "bagus", "top": "bagus",
    "bgs": "bagus", "apik": "bagus", "keren": "bagus",
    "minus": "kekurangan", "plus": "kelebihan",
    "mending": "lebih baik", "mendingan": "lebih baik",
    "dapet": "dapat", "dpt": "dapat",
    "makasih": "terima kasih", "thx": "terima kasih", "thanks": "terima kasih", "tengkyu": "terima kasih",
    "spek": "spesifikasi", "proc": "prosesor", "procie": "prosesor"
}

# ==============================================================================
# KAMUS KATA KUNCI ASPEK (LANGKAH 4)
# ==============================================================================
ASPECT_KEYWORDS = {
    "build quality": [
        "build", "bodi", "body", "bahan", "plastik", "besi", "metal", "alumunium", 
        "aluminium", "kokoh", "ringkih", "engsel", "hinge", "desain", "design", 
        "berat", "bobot", "tebal", "tipis", "finishing", "casing", "chassis", "material"
    ],
    "thermals": [
        "suhu", "panas", "adem", "dingin", "thermal", "thermals", "kipas", "fan", 
        "cooling", "overheat", "cooler", "ventilasi", "exhaust", "heatsink", 
        "derajat", "celcius", "suhunya", "anget", "hangat"
    ],
    "battery life": [
        "baterai", "batre", "battery", "daya", "charger", "cas", "ngecas", "charging", 
        "watt", "adaptor", "awet", "boros", "tahan", "jam", "mah", "wh", "ketahanan baterai"
    ],
    "performa": [
        "performa", "performance", "fps", "frame", "frame rate", "gaming", "game", 
        "gpu", "cpu", "processor", "prosesor", "ram", "speed", "kecepatan", "kencang", 
        "kenceng", "lemot", "lag", "macet", "stutter", "render", "rendering", 
        "benchmark", "rtx", "gtx", "ryzen", "intel", "core", "geforce", "ddr", "ssd", "tenaga"
    ],
    "layar": [
        "layar", "screen", "display", "panel", "ips", "oled", "srgb", "refresh rate", 
        "hz", "144hz", "165hz", "240hz", "warna", "color", "bezel", "nits", "terang", 
        "cerah", "resolusi", "fhd", "qhd", "ghosting", "bleeding", "backlight bleed"
    ],
    "keyboard": [
        "keyboard", "papan ketik", "tombol", "key", "rgb", "backlight", "backlit", 
        "travel", "tactile", "clicky", "touchpad", "trackpad", "numpad", "feel ketik"
    ]
}

# Lexicon penanda sentimen keluhan/negatif & kepuasan/positif
NEGATIVE_KEYWORDS = [
    "panas", "boros", "berisik", "mendem", "minus", "jelek", "kecewa", "drop", 
    "buruk", "rusak", "lemot", "lelet", "berat", "kurang", "mahal", "stutter", 
    "parah", "nyesel", "menyesal", "patah", "anget", "hangat", "overheat", "ringkih", 
    "cacat", "bocor", "ghosting", "bleed", "mati", "error", "turun", "kalah", "bising",
    "bapuk", "ancur", "hancur", "terlalu mahal"
]
POSITIVE_KEYWORDS = [
    "bagus", "mantap", "mantul", "keren", "jos", "josjis", "puas", "adem", "awet", 
    "kencang", "kenceng", "cepat", "sepadan", "worth", "murah", "terjangkau", 
    "rekomendasi", "rekomen", "rapi", "kokoh", "halus", "lancar", "enak", "senang",
    "terbaik", "juara", "gahar"
]


# ==============================================================================
# PIPELINE FUNCTIONS
# ==============================================================================

def merge_raw_datasets(folder_path: str = "raw-dataset/review-youtube") -> pd.DataFrame:
    """Menggabungkan seluruh file CSV ulasan youtube menjadi satu DataFrame tunggal."""
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"Tidak ada file CSV yang ditemukan di: {folder_path}")
    
    print(f"[*] Ditemukan {len(csv_files)} file CSV. Memulai penggabungan...")
    dfs = []
    for f in csv_files:
        df_part = pd.read_csv(f)
        df_part['file_sumber'] = os.path.basename(f)
        dfs.append(df_part)
        print(f"    - {os.path.basename(f)}: {len(df_part)} baris")
        
    merged_df = pd.concat(dfs, ignore_index=True)
    initial_len = len(merged_df)
    
    # Hapus duplikat ID_Komentar jika ada
    if 'ID_Komentar' in merged_df.columns:
        merged_df = merged_df.drop_duplicates(subset=['ID_Komentar']).reset_index(drop=True)
    
    # Hapus baris dengan teks kosong / NaN
    merged_df = merged_df.dropna(subset=['Teks_Asli']).reset_index(drop=True)
    merged_df['Teks_Asli'] = merged_df['Teks_Asli'].astype(str)
    
    print(f"[+] Total data berhasil digabung: {len(merged_df)} baris (Awal: {initial_len})\n")
    return merged_df


def step1_clean_text(text: str) -> str:
    """
    Langkah 1: Pembersihan Dasar (Text Cleaning)
    - Lowercasing
    - Hapus URL / tautan
    - Hapus emoji & karakter non-ASCII/unicode
    - Hapus tanda baca berlebih dan karakter non-alfanumerik
    - Normalisasi spasi berlebih
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercasing
    text = text.lower()
    
    # 2. Hapus URL / link (http, https, www)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # 3. Hapus tag HTML
    text = re.sub(r'<.*?>', ' ', text)
    
    # 4. Hapus mentions (@user) dan hashtag (#tag)
    text = re.sub(r'[@#]\w+', ' ', text)
    
    # 5. Hapus emoji & karakter unicode non-ASCII
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    # 6. Reduksi tanda baca berulang (e.g. "bagus!!!!" -> "bagus !")
    text = re.sub(r'([!?.,;:])\1+', r'\1', text)
    
    # 7. Ganti karakter non-alfanumerik dengan spasi
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # 8. Normalisasi spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def step2_normalize_slang(text: str) -> str:
    """
    Langkah 2: Normalisasi Bahasa Gaul
    - Menormalkan frasa multi-kata gaul (misal: "panas bgt" -> "panas banget", "fps drop" -> "penurunan frame rate")
    - Menormalkan kata gaul per kata menggunakan SLANG_WORD_MAP
    """
    # 1. Normalisasi frasa multi-kata
    for slang_phrase, formal_phrase in SLANG_MULTIWORD_MAP.items():
        pattern = rf'\b{re.escape(slang_phrase)}\b'
        text = re.sub(pattern, formal_phrase, text)
        
    # 2. Normalisasi kata per kata
    tokens = text.split()
    normalized_tokens = [SLANG_WORD_MAP.get(token, token) for token in tokens]
    return " ".join(normalized_tokens)


def detect_language(tokens: list) -> str:
    """
    Mendeteksi apakah teks lebih dominan Bahasa Inggris ('en') atau Indonesia ('id')
    berdasarkan frekuensi kata stopword masing-masing bahasa.
    """
    en_hits = sum(1 for w in tokens if w in STOPWORDS_EN)
    id_hits = sum(1 for w in tokens if w in STOPWORDS_ID)
    
    if en_hits > id_hits and en_hits >= 2:
        return 'en'
    return 'id'


def step3_tokenize_stopword_stem(text: str) -> dict:
    """
    Langkah 3: Tokenisasi, Stopword Removal, dan Stemming
    - Deteksi bahasa teks (Indonesia vs Inggris)
    - Bahasa Indonesia: Tokenisasi, Stopword Removal Sastrawi, Stemming Sastrawi (dengan cache)
    - Bahasa Inggris: NLTK word_tokenize, Stopwords Inggris, NLTK PorterStemmer
    """
    tokens = text.split()
    if not tokens:
        return {
            "tokens": [],
            "language": "id",
            "stemmed_tokens": [],
            "cleaned_text": ""
        }
    
    lang = detect_language(tokens)
    stemmed_tokens = []
    
    if lang == 'en':
        # Pipeline NLTK untuk Bahasa Inggris
        nltk_tokens = word_tokenize(text)
        filtered_tokens = [
            w for w in nltk_tokens 
            if w.isalpha() and len(w) > 1 and w not in STOPWORDS_EN
        ]
        stemmed_tokens = [STEMMER_EN.stem(w) for w in filtered_tokens]
    else:
        # Pipeline Sastrawi untuk Bahasa Indonesia
        filtered_tokens = [
            w for w in tokens 
            if w.isalpha() and len(w) > 1 and w not in STOPWORDS_ID
        ]
        # Stemming dengan dictionary cache
        stemmed_tokens = [get_stem_id(w) for w in filtered_tokens]
        # Hapus kembali token stopword setelah stemming
        stemmed_tokens = [w for w in stemmed_tokens if w not in STOPWORDS_ID and len(w) > 1]
        
    return {
        "tokens": filtered_tokens,
        "language": lang,
        "stemmed_tokens": stemmed_tokens,
        "cleaned_text": " ".join(stemmed_tokens)
    }


def step4_extract_aspects(raw_text: str) -> dict:
    """
    Langkah 4: Pelabelan Aspek (Aspect Extraction)
    - Pecah ulasan menjadi per kalimat menggunakan NLTK sent_tokenize
    - Petakan kalimat ke aspek tertentu (build quality, thermals, battery life, performa, layar, keyboard)
    berbasis pencocokan kata kunci.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return {"detected_aspects": [], "aspect_sentences": {}}
    
    sentences = sent_tokenize(raw_text)
    review_aspects = set()
    aspect_mapping = {asp: [] for asp in ASPECT_KEYWORDS.keys()}
    
    for sentence in sentences:
        s_clean = sentence.lower()
        words_in_sent = set(re.findall(r'\b\w+\b', s_clean))
        
        for aspect, keywords in ASPECT_KEYWORDS.items():
            for kw in keywords:
                if " " in kw:
                    if kw in s_clean:
                        aspect_mapping[aspect].append(sentence.strip())
                        review_aspects.add(aspect)
                        break
                else:
                    if kw in words_in_sent:
                        aspect_mapping[aspect].append(sentence.strip())
                        review_aspects.add(aspect)
                        break
                        
    filtered_aspect_mapping = {k: list(set(v)) for k, v in aspect_mapping.items() if len(v) > 0}
    
    return {
        "detected_aspects": sorted(list(review_aspects)),
        "aspect_sentences": filtered_aspect_mapping
    }


def label_sentiment_heuristic(text: str) -> str:
    """
    Pelabelan sentimen ulasan (positive, neutral, negative)
    berdasarkan kamus kata sentimen laptop/gadget.
    """
    tokens = set(text.lower().split())
    neg_score = sum(1 for w in tokens if w in NEGATIVE_KEYWORDS)
    pos_score = sum(1 for w in tokens if w in POSITIVE_KEYWORDS)
    
    if neg_score > pos_score:
        return "negative"
    elif pos_score > neg_score:
        return "positive"
    else:
        return "neutral"


def step5_vectorize_tfidf(texts: pd.Series, max_features: int = 1500) -> tuple:
    """
    Langkah 5: Representasi Teks (Vektorisasi TF-IDF)
    - Kombinasi unigram dan bigram (ngram_range=(1, 2))
    - Ringan dan siap untuk Logistic Regression, Naive Bayes, maupun K-Means.
    """
    print("[*] Melakukan Vektorisasi TF-IDF (Unigram + Bigram)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"[+] TF-IDF Matrix selesai dibuat. Shape: {tfidf_matrix.shape}")
    return vectorizer, tfidf_matrix


def step6_clustering_negative_reviews(
    df_subset: pd.DataFrame, 
    tfidf_subset, 
    vectorizer: TfidfVectorizer, 
    n_components: int = 10, 
    n_clusters: int = 3
) -> dict:
    """
    Langkah 6: Reduksi Dimensi & Clustering (Khusus Pipeline Topik Keluhan)
    - Ambil ulasan bersentimen negatif (keluhan)
    - Reduksi dimensi fitur TF-IDF menggunakan TruncatedSVD
    - Terapkan model clustering KMeans untuk mengelompokkan topik keluhan
    - Ekstraksi top kata kunci per klaster
    """
    print("[*] Langkah 6: Reduksi Dimensi & Clustering Ulasan Negatif...")
    
    n_samples = tfidf_subset.shape[0]
    n_features = tfidf_subset.shape[1]
    n_comp = min(n_components, n_features - 1, n_samples - 1)
    
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    reduced_features = svd.fit_transform(tfidf_subset)
    explained_var = svd.explained_variance_ratio_.sum()
    print(f"    - TruncatedSVD ({n_comp} komponen) Cumulative Explained Variance: {explained_var:.2%}")
    
    # Clustering dengan KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(reduced_features)
    
    # Rekonstruksi kata kunci pusat klaster
    cluster_centers = svd.inverse_transform(kmeans.cluster_centers_)
    terms = vectorizer.get_feature_names_out()
    
    cluster_keywords = {}
    for idx, center in enumerate(cluster_centers):
        top_indices = center.argsort()[::-1][:8]
        top_words = [terms[i] for i in top_indices]
        cluster_keywords[f"Cluster_{idx}"] = top_words
        print(f"    [Cluster {idx}] Kata kunci utama keluhan: {', '.join(top_words)}")
        
    return {
        "svd_model": svd,
        "kmeans_model": kmeans,
        "reduced_features": reduced_features,
        "cluster_labels": cluster_labels,
        "cluster_keywords": cluster_keywords
    }


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    print("=" * 70)
    print("PIPELINE PREPROCESSING TEKS ULASAN - GAMERGEAR")
    print("=" * 70)
    
    # 0. Merge Dataset
    df = merge_raw_datasets("raw-dataset/review-youtube")
    
    # Simpan dataset gabungan mentah
    os.makedirs("raw-dataset", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    df.to_csv("raw-dataset/merged_youtube_reviews.csv", index=False)
    print(f"[+] Dataset gabungan mentah tersimpan di: raw-dataset/merged_youtube_reviews.csv")
    
    # Langkah 1: Text Cleaning
    print("\n[*] Menjalankan Langkah 1: Pembersihan Dasar (Text Cleaning)...")
    df['teks_bersih_dasar'] = df['Teks_Asli'].apply(step1_clean_text)
    
    # Langkah 2: Normalisasi Bahasa Gaul
    print("[*] Menjalankan Langkah 2: Normalisasi Bahasa Gaul...")
    df['teks_ternormalisasi'] = df['teks_bersih_dasar'].apply(step2_normalize_slang)
    
    # Langkah 3: Tokenisasi, Stopword Removal, Stemming
    print("[*] Menjalankan Langkah 3: Tokenisasi, Stopword Removal, dan Stemming (ID & EN)...")
    step3_results = df['teks_ternormalisasi'].apply(step3_tokenize_stopword_stem)
    df['bahasa'] = [r['language'] for r in step3_results]
    df['tokens'] = [r['tokens'] for r in step3_results]
    df['stemmed_tokens'] = [r['stemmed_tokens'] for r in step3_results]
    df['teks_final'] = [r['cleaned_text'] for r in step3_results]
    
    # Langkah 4: Aspect Extraction
    print("[*] Menjalankan Langkah 4: Ekstraksi & Pelabelan Aspek (Per Kalimat)...")
    aspect_results = df['Teks_Asli'].apply(step4_extract_aspects)
    df['aspek_terdeteksi'] = [r['detected_aspects'] for r in aspect_results]
    df['detail_kalimat_aspek'] = [json.dumps(r['aspect_sentences']) for r in aspect_results]
    
    # Kolom boolean untuk setiap aspek
    for aspect in ASPECT_KEYWORDS.keys():
        col_name = f"aspek_{aspect.replace(' ', '_')}"
        df[col_name] = df['aspek_terdeteksi'].apply(lambda x: 1 if aspect in x else 0)
        
    # Pelabelan Sentimen (Heuristik Lexicon untuk mendeteksi ulasan keluhan/negatif)
    df['sentimen_label'] = df['teks_ternormalisasi'].apply(label_sentiment_heuristic)
    print(f"[*] Distribusi Sentimen Ulasan:")
    print(df['sentimen_label'].value_counts())
    
    # Langkah 5: TF-IDF Vectorization
    print("\n[*] Menjalankan Langkah 5: Vektorisasi TF-IDF...")
    valid_mask = df['teks_final'].str.strip().str.len() > 0
    vectorizer, tfidf_matrix = step5_vectorize_tfidf(df.loc[valid_mask, 'teks_final'])
    
    # Simpan index yang valid
    valid_indices = df[valid_mask].index
    
    # Langkah 6: Reduksi Dimensi & Clustering Keluhan
    print("\n[*] Menjalankan Langkah 6: Reduksi Dimensi & Clustering Keluhan...")
    neg_sub_mask = df.loc[valid_mask, 'sentimen_label'] == 'negative'
    neg_indices_in_valid = np.where(neg_sub_mask)[0]
    neg_original_indices = valid_indices[neg_sub_mask]
    
    neg_tfidf = tfidf_matrix[neg_indices_in_valid]
    print(f"    - Terpilih {len(neg_indices_in_valid)} ulasan negatif untuk clustering keluhan.")
    
    clustering_results = step6_clustering_negative_reviews(
        df_subset=df.loc[neg_original_indices],
        tfidf_subset=neg_tfidf,
        vectorizer=vectorizer,
        n_components=10,
        n_clusters=3
    )
    
    # Simpan klaster ke dataframe utama
    df['complaint_cluster'] = -1  # -1 untuk bukan ulasan keluhan/negatif
    df.loc[neg_original_indices, 'complaint_cluster'] = clustering_results['cluster_labels']
    
    # Simpan dataset hasil preprocessing lengkap
    output_csv = "raw-dataset/dataset_preprocessed_review.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n[+] Dataset Preprocessed tersimpan di: {output_csv}")
    
    # Simpan model untuk pipeline/inference
    joblib.dump(vectorizer, "model/tfidf_review_vectorizer.pkl")
    joblib.dump(clustering_results['svd_model'], "model/svd_clustering.pkl")
    joblib.dump(clustering_results['kmeans_model'], "model/kmeans_complaints.pkl")
    print("[+] Model TF-IDF Vectorizer, SVD, dan KMeans berhasil disimpan ke folder 'model/'.")
    
    print("\n" + "=" * 70)
    print("PIPELINE SELESAI DENGAN SUKSES!")
    print("=" * 70)


if __name__ == "__main__":
    main()
