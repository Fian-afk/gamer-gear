"""
Modul Mesin Chatbot Rule-Based GamerGear.
Menggunakan regex, RapidFuzz, JSON knowledge base, dan SQLite.
"""

import os
import re
import json
import sqlite3
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from rapidfuzz import process, fuzz


# Path aman berbasis lokasi file ini
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "data", "gamergear.db")
DEFAULT_JSON_PATH = os.path.join(BASE_DIR, "data", "game_requirements.json")


def format_rupiah(amount: int) -> str:
    """Format integer rupiah menjadi string berformat 'Rp 15.000.000'."""
    try:
        return f"Rp {int(amount):,}".replace(",", ".")
    except (ValueError, TypeError):
        return str(amount)


class ChatbotEngine:
    """Mesin Chatbot Rule-Based untuk GamerGear."""

    def __init__(self, db_path: Optional[str] = None, json_path: Optional[str] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.json_path = json_path or DEFAULT_JSON_PATH

        # Muat basis pengetahuan
        self.gpu_tiers: Dict[str, int] = {}
        self.cpu_tiers: Dict[str, int] = {}
        self.games: List[Dict[str, Any]] = []

        self._load_knowledge_base()
        self._init_db_tables()

    def _load_knowledge_base(self) -> None:
        """Memuat gpu_tiers, cpu_tiers, dan games dari file JSON."""
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Knowledge base tidak ditemukan di: {self.json_path}")

        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_gpu = data.get("gpu_tiers", {})
        raw_cpu = data.get("cpu_tiers", {})
        self.games = data.get("games", [])

        # Normalisasi dan perluas dictionary GPU
        self.gpu_tiers = {}
        for k, v in raw_gpu.items():
            k_clean = k.strip().lower()
            self.gpu_tiers[k_clean] = v
            # Variasi tanpa 'geforce ' atau 'nvidia '
            if k_clean.startswith("geforce "):
                self.gpu_tiers[k_clean.replace("geforce ", "", 1)] = v
                self.gpu_tiers["nvidia " + k_clean] = v
            elif k_clean.startswith("amd "):
                self.gpu_tiers[k_clean.replace("amd ", "", 1)] = v

        # Normalisasi dan perluas dictionary CPU
        self.cpu_tiers = {}
        for k, v in raw_cpu.items():
            k_clean = k.strip().lower()
            self.cpu_tiers[k_clean] = v
            # Dukungan format seperti "i5 gen 12" dari "core i5 gen 12"
            if k_clean.startswith("core "):
                without_core = k_clean.replace("core ", "", 1)
                self.cpu_tiers[without_core] = v
                self.cpu_tiers["intel " + k_clean] = v
                self.cpu_tiers["intel " + without_core] = v
            if k_clean.startswith("intel "):
                self.cpu_tiers[k_clean.replace("intel ", "", 1)] = v
            if k_clean.startswith("amd "):
                self.cpu_tiers[k_clean.replace("amd ", "", 1)] = v

        # Siapkan corpus pencarian game (nama_game + aliases)
        self.game_corpus: Dict[str, Dict[str, Any]] = {}
        for g in self.games:
            self.game_corpus[g["nama_game"].strip().lower()] = g
            for alias in g.get("aliases", []):
                self.game_corpus[alias.strip().lower()] = g

        # Urutkan kunci pencarian berdasarkan panjang menurun untuk prioritas substring
        self.sorted_gpu_keys = sorted(self.gpu_tiers.keys(), key=len, reverse=True)
        self.sorted_cpu_keys = sorted(self.cpu_tiers.keys(), key=len, reverse=True)
        self.sorted_game_phrases = sorted(self.game_corpus.keys(), key=len, reverse=True)

    def _init_db_tables(self) -> None:
        """Memastikan tabel chat_logs dan unanswered_queries tersedia."""
        if not os.path.exists(os.path.dirname(self.db_path)):
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                waktu TEXT,
                pertanyaan TEXT,
                intent TEXT,
                dijawab BOOLEAN
            )
            """)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS unanswered_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                waktu TEXT,
                pertanyaan TEXT,
                intent_terdekat TEXT,
                status_tinjauan TEXT DEFAULT 'pending'
            )
            """)
            conn.commit()

    def _log_chat(self, pertanyaan: str, intent: str, dijawab: bool) -> None:
        """Menyimpan setiap interaksi chat ke tabel chat_logs."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO chat_logs (waktu, pertanyaan, intent, dijawab) VALUES (?, ?, ?, ?)",
                    (datetime.now().isoformat(), pertanyaan, intent, 1 if dijawab else 0)
                )
                conn.commit()
        except Exception as e:
            print(f"[Error _log_chat] {e}")

    def _log_unanswered(self, pertanyaan: str, intent_terdekat: str = "unknown") -> None:
        """Menyimpan query yang tidak dapat dijawab ke tabel unanswered_queries."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO unanswered_queries (waktu, pertanyaan, intent_terdekat, status_tinjauan) VALUES (?, ?, ?, 'pending')",
                    (datetime.now().isoformat(), pertanyaan, intent_terdekat)
                )
                conn.commit()
        except Exception as e:
            print(f"[Error _log_unanswered] {e}")

    # =========================================================================
    # EKSTRAKSI ENTITAS (REGEX + RAPIDFUZZ)
    # =========================================================================

    def extract_ram(self, text: str) -> Optional[int]:
        """
        Tangkap angka kapasitas RAM via regex.
        Contoh: '8gb', '16 gb', 'ram 32', '32g', 'ram: 16', '16gb ram' -> int
        """
        # 1. Format eksplisit dengan kata 'ram'
        m1 = re.search(r'\bram\s*[:=]?\s*(\d{1,3})\s*(?:gb|g)?\b', text, re.IGNORECASE)
        if m1:
            return int(m1.group(1))

        m2 = re.search(r'\b(\d{1,3})\s*(?:gb|g)?\s*ram\b', text, re.IGNORECASE)
        if m2:
            return int(m2.group(1))

        # 2. Format angka + GB / G umum
        matches = list(re.finditer(r'\b(\d{1,3})\s*(?:gb|g)\b', text, re.IGNORECASE))
        if matches:
            # Jika ada beberapa (misal 'rtx 3050 4gb ram 16gb'), ambil angka yang relevan untuk RAM
            # Biasanya RAM adalah nilai 4, 8, 12, 16, 24, 32, 64
            # Jika ada kata vram di dekat angka sebelumnya, kita ambil angka yang lain
            candidates = [int(m.group(1)) for m in matches]
            # Ambil candidate terakhir jika bukan vram
            return candidates[-1]

        return None

    def extract_budget(self, text: str) -> Optional[int]:
        """
        Tangkap nominal anggaran Rupiah via regex dengan dukungan format:
        '12 juta', '12jt', '10.5 jt', '10,5jt', '15000000', '15.000.000' -> int Rupiah penuh
        """
        # 1. Pola dengan 'juta' atau 'jt' (mendukung desimal dengan koma atau titik)
        m_juta = re.search(r'(\d+(?:[.,]\d+)?)\s*(?:juta|jt)\b', text, re.IGNORECASE)
        if m_juta:
            val_str = m_juta.group(1).replace(",", ".")
            return int(float(val_str) * 1_000_000)

        # 2. Pola angka ribuan dengan pemisah titik atau koma (misal: 15.000.000 atau 15,000,000)
        m_fmt = re.search(r'\b(\d{1,3}(?:[.,]\d{3}){2,3})\b', text)
        if m_fmt:
            num_clean = re.sub(r'[.,]', '', m_fmt.group(1))
            return int(num_clean)

        # 3. Pola digit murni (6 hingga 10 digit, misal 15000000)
        m_digits = re.search(r'\b(\d{6,10})\b', text)
        if m_digits:
            return int(m_digits.group(1))

        return None

    def extract_game(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Cocokkan teks dengan nama_game dan aliases pada games di JSON.
        Menggunakan regex dan RapidFuzz (threshold >= 78) agar toleran typo.
        """
        text_lower = text.lower()

        # 1. Pencocokan langsung dengan boundary kata (dari frase terpanjang)
        for phrase in self.sorted_game_phrases:
            pattern = r'(?<!\w)' + re.escape(phrase) + r'(?!\w)'
            if re.search(pattern, text_lower):
                return self.game_corpus[phrase]

        # 2. Fuzzy matching dengan RapidFuzz (threshold >= 78)
        # Ambil n-gram kata dari teks input (panjang 1 s.d. 4 kata)
        words = re.findall(r'[a-zA-Z0-9]+', text_lower)
        best_match = None
        best_score = 0.0

        for n in range(1, 5):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i + n])
                # Filter kandidat dengan panjang yang sebanding untuk akurasi tinggi
                filtered_cands = [
                    c for c in self.sorted_game_phrases
                    if abs(len(c) - len(ngram)) <= max(4, int(len(ngram) * 0.5)) and len(c) >= 3
                ]
                if not filtered_cands:
                    continue

                res = process.extractOne(ngram, filtered_cands, scorer=fuzz.ratio)
                if res and res[1] >= 78 and res[1] > best_score:
                    best_score = res[1]
                    best_match = self.game_corpus[res[0]]

        return best_match

    def extract_gpus(self, text: str) -> List[Dict[str, Any]]:
        """
        Cari kecocokan seri GPU dari kunci gpu_tiers di JSON menggunakan regex
        dan/atau RapidFuzz (threshold >= 80). Mengembalikan list GPU yang ditemukan.
        """
        text_lower = text.lower()
        # Normalisasi spasi rapat seperti rtx3050 -> rtx 3050, gtx1650 -> gtx 1650
        text_normalized = re.sub(r'\b(rtx|gtx|rx|mx|gt)(\d+)', r'\1 \2', text_lower)

        found_gpus: List[Tuple[int, int, str, int]] = []
        matched_spans: List[Tuple[int, int]] = []

        # 1. Direct regex match berdasarkan sorted_gpu_keys (terpanjang dulu)
        for k in self.sorted_gpu_keys:
            pattern = r'(?<!\w)' + re.escape(k) + r'(?!\w)'
            for m in re.finditer(pattern, text_normalized):
                s, e = m.span()
                # Cek overlap dengan span yang sudah ditemukan
                if not any(s < oe and e > os_ for os_, oe in matched_spans):
                    matched_spans.append((s, e))
                    found_gpus.append((s, e, k, self.gpu_tiers[k]))

        # 2. Jika tidak ada yang cocok langsung, coba fuzzy matching token per token
        if not found_gpus:
            tokens = re.findall(r'[a-zA-Z0-9]+(?:\s+[a-zA-Z0-9]+)?', text_normalized)
            for token in tokens:
                token_clean = token.strip()
                if len(token_clean) < 4:
                    continue
                res = process.extractOne(token_clean, self.sorted_gpu_keys, scorer=fuzz.ratio)
                if res and res[1] >= 80:
                    matched_key = res[0]
                    found_gpus.append((0, 0, matched_key, self.gpu_tiers[matched_key]))
                    break

        # Urutkan berdasarkan posisi kemunculan dalam teks
        found_gpus.sort(key=lambda x: x[0])

        # Hapus duplikat nama GPU dengan tetap mempertahankan urutan
        result: List[Dict[str, Any]] = []
        seen_names = set()
        for _, _, name, tier in found_gpus:
            if name not in seen_names:
                seen_names.add(name)
                result.append({"name": name, "tier": tier})

        return result

    def extract_cpus(self, text: str) -> List[Dict[str, Any]]:
        """
        Cari kecocokan seri CPU dari kunci cpu_tiers di JSON (misal: 'i5 gen 12', 'ryzen 5', 'i3', 'core i7').
        Mendukung ekstraksi multiple CPU untuk perbandingan.
        """
        text_lower = text.lower()
        # Normalisasi tanda hubung: misal 'i5-12400f' -> 'i5 12400f'
        text_normalized = re.sub(r'([a-z0-9])-([a-z0-9])', r'\1 \2', text_lower)
        text_normalized = re.sub(r'\b(core|ryzen|intel|amd)\s+', r'\1 ', text_normalized)

        found_cpus: List[Tuple[int, int, str, int]] = []
        matched_spans: List[Tuple[int, int]] = []

        # 1. Direct regex match berdasarkan sorted_cpu_keys
        for k in self.sorted_cpu_keys:
            pattern = r'(?<!\w)' + re.escape(k) + r'(?!\w)'
            for m in re.finditer(pattern, text_normalized):
                s, e = m.span()
                if not any(s < oe and e > os_ for os_, oe in matched_spans):
                    matched_spans.append((s, e))
                    found_cpus.append((s, e, k, self.cpu_tiers[k]))

        # 2. Fuzzy match jika tidak ada CPU yang terdeteksi secara langsung
        if not found_cpus:
            tokens = re.findall(r'[a-zA-Z0-9]+(?:\s+[a-zA-Z0-9]+)?', text_normalized)
            for token in tokens:
                token_clean = token.strip()
                if len(token_clean) < 3:
                    continue
                res = process.extractOne(token_clean, self.sorted_cpu_keys, scorer=fuzz.ratio)
                if res and res[1] >= 80:
                    matched_key = res[0]
                    found_cpus.append((0, 0, matched_key, self.cpu_tiers[matched_key]))
                    break

        found_cpus.sort(key=lambda x: x[0])

        result: List[Dict[str, Any]] = []
        seen_names = set()
        for _, _, name, tier in found_cpus:
            if name not in seen_names:
                seen_names.add(name)
                result.append({"name": name, "tier": tier})

        return result

    def extract_laptops_from_db(self, text: str) -> List[Dict[str, Any]]:
        """Mencari laptop di database SQLite yang sesuai dengan nama/merek yang disebut pengguna."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, nama, merek, cpu, gpu, ram_gb, ssd_gb, layar_inci,
                           refresh_rate, harga_listing, skor_value, tier_performa
                    FROM laptops
                """)
                all_laptops = [dict(r) for r in cursor.fetchall()]

            text_lower = text.lower()
            matched = []

            # Periksa pemisahan kata kunci 'vs', 'versus', 'banding', 'atau'
            parts = re.split(r'\b(?:vs|versus|atau|bandingkan|dibanding|sama|dengan)\b', text_lower)
            cleaned_parts = [p.strip() for p in parts if p.strip()]

            if len(cleaned_parts) >= 2:
                for part in cleaned_parts[:2]:
                    # Cari laptop yang namanya paling mendekati part
                    best_match = None
                    best_score = 0
                    for lap in all_laptops:
                        full_name = f"{lap['merek']} {lap['nama']}".lower()
                        score = fuzz.token_set_ratio(part, full_name)
                        if score > best_score and score >= 60:
                            best_score = score
                            best_match = lap
                    if best_match and best_match not in matched:
                        matched.append(best_match)
            else:
                for lap in all_laptops:
                    full_name = f"{lap['merek']} {lap['nama']}".lower()
                    if fuzz.partial_ratio(lap["nama"].lower(), text_lower) >= 80:
                        matched.append(lap)
                        if len(matched) >= 2:
                            break

            return matched
        except Exception as e:
            print(f"[Error extract_laptops_from_db] {e}")
            return []

    def extract_all_entities(self, text: str) -> Dict[str, Any]:
        """Mengekstrak seluruh entitas dari teks pesan pengguna."""
        ram = self.extract_ram(text)
        budget = self.extract_budget(text)
        game = self.extract_game(text)
        gpus = self.extract_gpus(text)
        cpus = self.extract_cpus(text)
        laptops = self.extract_laptops_from_db(text)

        # Format entitas sesuai spesifikasi
        # Jika ada 2 GPU, simpan sebagai list. Jika 1, simpan sebagai string nama.
        gpu_entity = None
        if len(gpus) == 1:
            gpu_entity = gpus[0]["name"]
        elif len(gpus) >= 2:
            gpu_entity = [g["name"] for g in gpus]

        cpu_entity = None
        if len(cpus) == 1:
            cpu_entity = cpus[0]["name"]
        elif len(cpus) >= 2:
            cpu_entity = [c["name"] for c in cpus]

        game_entity = game["nama_game"] if game else None

        return {
            "ram_gb": ram,
            "gpu": gpu_entity,
            "cpu": cpu_entity,
            "game": game_entity,
            "budget": budget,
            "_raw_gpus": gpus,
            "_raw_cpus": cpus,
            "_raw_game": game,
            "_raw_laptops": laptops,
        }

    # =========================================================================
    # INTENT HANDLERS
    # =========================================================================

    def handle_sapaan_bantuan(self, entities: Dict[str, Any]) -> str:
        """Handler untuk intent sapaan_bantuan."""
        return (
            "Halo Gamers! 👋 Selamat datang di **GamerGear AI Assistant**.\n"
            "Saya siap membantu kamu memilih dan menganalisis perangkat gaming terbaik! "
            "Berikut adalah 4 fitur utama yang bisa kamu tanyakan:\n\n"
            "1. 🎮 **Cek Kompatibilitas Game**\n"
            "   Cek apakah spek laptop/PC kamu kuat menjalankan game tertentu.\n"
            "   *Contoh:* \"Apakah laptop i5 gen 12 RTX 3050 RAM 16GB kuat main Cyberpunk 2077?\"\n\n"
            "2. 💰 **Rekomendasi Laptop Sesuai Budget**\n"
            "   Temukan laptop gaming dengan performa dan value terbaik sesuai anggaranmu.\n"
            "   *Contoh:* \"Rekomendasi laptop gaming budget 15 juta\"\n\n"
            "3. ⚖️ **Bandingkan Komponen & Laptop**\n"
            "   Bandingkan performa 2 GPU, 2 CPU, atau produk laptop spesifik.\n"
            "   *Contoh:* \"Mending RTX 3050 vs RTX 4060?\" atau \"Bandingkan Lenovo LOQ vs HP Victus\"\n\n"
            "4. ⚡ **Analisis Bottleneck CPU & GPU**\n"
            "   Cek apakah kombinasi prosesor dan kartu grafismu sudah seimbang atau terjadi bottleneck.\n"
            "   *Contoh:* \"Apakah i3 gen 10 dan RTX 4070 bakal bottleneck?\"\n\n"
            "Silakan ketik pertanyaanmu langsung ya!"
        )

    def handle_cek_game(self, entities: Dict[str, Any], raw_text: str) -> Tuple[str, bool]:
        """Handler untuk intent cek_game."""
        game_data = entities.get("_raw_game")
        if not game_data:
            # Game tidak ditemukan, minta pengguna menyebutkan nama game
            response = (
                "Kamu ingin mengecek kemampuan spek untuk game apa nih? 🎮\n"
                "Sebutkan nama game dan spesifikasi komputermu (CPU, GPU, RAM) ya.\n"
                "*Contoh:* \"Kuat main Black Myth Wukong dengan i5 gen 12, RTX 3050, RAM 16GB?\""
            )
            return response, False

        game_name = game_data["nama_game"]
        kategori = game_data.get("kategori", "Gaming")
        storage = game_data.get("storage_gb", 50)
        min_cpu = game_data.get("min_cpu_tier", 2)
        min_gpu = game_data.get("min_gpu_tier", 3)
        min_ram = game_data.get("min_ram_gb", 8)
        rec_cpu = game_data.get("rec_cpu_tier", 4)
        rec_gpu = game_data.get("rec_gpu_tier", 5)
        rec_ram = game_data.get("rec_ram_gb", 16)

        raw_cpus = entities.get("_raw_cpus", [])
        raw_gpus = entities.get("_raw_gpus", [])
        user_ram = entities.get("ram_gb")

        user_cpu = raw_cpus[0] if raw_cpus else None
        user_gpu = raw_gpus[0] if raw_gpus else None

        # Evaluasi masing-masing komponen
        evaluasi = []
        asumsi_catatan = []
        komponen_kurang = []
        semua_rekomendasi = True
        semua_minimum = True

        # 1. CPU
        if user_cpu:
            cpu_name = user_cpu["name"].title()
            cpu_tier = user_cpu["tier"]
            if cpu_tier >= rec_cpu:
                evaluasi.append(f"• **CPU**: {cpu_name} (Tier {cpu_tier}) ➔ ✅ **Memenuhi Rekomendasi** (Tier Rec: {rec_cpu})")
            elif cpu_tier >= min_cpu:
                semua_rekomendasi = False
                evaluasi.append(f"• **CPU**: {cpu_name} (Tier {cpu_tier}) ➔ ⚠️ **Memenuhi Minimum** (Tier Min: {min_cpu}, Rec: {rec_cpu})")
            else:
                semua_rekomendasi = False
                semua_minimum = False
                komponen_kurang.append(f"CPU ({cpu_name} Tier {cpu_tier} < Min Tier {min_cpu})")
                evaluasi.append(f"• **CPU**: {cpu_name} (Tier {cpu_tier}) ➔ ❌ **Di Bawah Minimum** (Min Tier: {min_cpu})")
        else:
            asumsi_catatan.append("CPU tidak disebutkan (diasumsikan setara tier rekomendasi)")

        # 2. GPU
        if user_gpu:
            gpu_name = user_gpu["name"].upper()
            gpu_tier = user_gpu["tier"]
            if gpu_tier >= rec_gpu:
                evaluasi.append(f"• **GPU**: {gpu_name} (Tier {gpu_tier}) ➔ ✅ **Memenuhi Rekomendasi** (Tier Rec: {rec_gpu})")
            elif gpu_tier >= min_gpu:
                semua_rekomendasi = False
                evaluasi.append(f"• **GPU**: {gpu_name} (Tier {gpu_tier}) ➔ ⚠️ **Memenuhi Minimum** (Tier Min: {min_gpu}, Rec: {rec_gpu})")
            else:
                semua_rekomendasi = False
                semua_minimum = False
                komponen_kurang.append(f"GPU ({gpu_name} Tier {gpu_tier} < Min Tier {min_gpu})")
                evaluasi.append(f"• **GPU**: {gpu_name} (Tier {gpu_tier}) ➔ ❌ **Di Bawah Minimum** (Min Tier: {min_gpu})")
        else:
            asumsi_catatan.append("GPU tidak disebutkan (diasumsikan setara tier rekomendasi)")

        # 3. RAM
        if user_ram is not None:
            if user_ram >= rec_ram:
                evaluasi.append(f"• **RAM**: {user_ram} GB ➔ ✅ **Memenuhi Rekomendasi** (Rec: {rec_ram} GB)")
            elif user_ram >= min_ram:
                semua_rekomendasi = False
                evaluasi.append(f"• **RAM**: {user_ram} GB ➔ ⚠️ **Memenuhi Minimum** (Min: {min_ram} GB, Rec: {rec_ram} GB)")
            else:
                semua_rekomendasi = False
                semua_minimum = False
                komponen_kurang.append(f"RAM ({user_ram} GB < Min {min_ram} GB)")
                evaluasi.append(f"• **RAM**: {user_ram} GB ➔ ❌ **Di Bawah Minimum** (Min: {min_ram} GB)")
        else:
            asumsi_catatan.append("RAM tidak disebutkan (diasumsikan setara tier rekomendasi)")

        # Tentukan status akhir
        if not user_cpu and not user_gpu and user_ram is None:
            # Tidak ada komponen yang disebutkan sama sekali
            return (
                f"### 🎮 Spesifikasi Game: **{game_name}** ({kategori})\n"
                f"• **Kebutuhan Minimum**: CPU Tier {min_cpu}, GPU Tier {min_gpu}, RAM {min_ram} GB\n"
                f"• **Kebutuhan Rekomendasi**: CPU Tier {rec_cpu}, GPU Tier {rec_gpu}, RAM {rec_ram} GB\n"
                f"• **Perkiraan Storage**: ±{storage} GB\n\n"
                "Untuk mengetahui apakah laptop/PC kamu kuat atau tidak, silakan sebutkan spesifikasi kamu "
                "(prosesor, kartu grafis, dan kapasitas RAM) ya!\n"
                f"*Contoh:* \"Kuat main {game_name} dengan i5 gen 12, RTX 3050, RAM 16GB?\""
            ), True

        if not semua_minimum:
            status_badge = "❌ **Status: Tidak Memenuhi Syarat Minimum**"
            kesimpulan = (
                f"Perangkatmu belum memenuhi syarat minimum untuk menjalankan **{game_name}**. "
                f"Komponen yang perlu di-upgrade: {', '.join(komponen_kurang)}. "
                "Jika dipaksakan, game berisiko mengalami lag parah, patah-patah, atau crash."
            )
        elif semua_rekomendasi:
            status_badge = "🎉 **Status: Rekomendasi (Lancar)**"
            kesimpulan = (
                f"Kabar gembira! Spesifikasimu sangat mumpuni untuk memainkan **{game_name}** secara lancar. "
                "Kamu bisa menikmati resolusi 1080p dengan setting grafis High atau 60+ FPS!"
            )
        else:
            status_badge = "⚡ **Status: Minimum (Bisa Jalan)**"
            kesimpulan = (
                f"Spesifikasimu memenuhi syarat minimum untuk **{game_name}**. "
                "Game bisa berjalan cukup baik (estimasi 30-45 FPS pada resolusi 1080p setting Low-Medium). "
                "Untuk pengalaman yang lebih mulus 60 FPS, pertimbangkan upgrade komponen yang masih di batas minimum."
            )

        resp_lines = [
            f"### 🎮 Analisis Kompatibilitas: **{game_name}**",
            status_badge,
            "",
            "**Detail Komponen:**",
            *evaluasi,
        ]
        if asumsi_catatan:
            resp_lines.append("")
            resp_lines.append(f"*Catatan: {'; '.join(asumsi_catatan)}.*")

        resp_lines.append("")
        resp_lines.append(f"**Kesimpulan:** {kesimpulan}")
        resp_lines.append(f"*(Tips: Pastikan juga ruang penyimpanan mencukupi ±{storage} GB SSD).*")

        return "\n".join(resp_lines), True

    def handle_rekomendasi_budget(self, entities: Dict[str, Any]) -> Tuple[str, bool]:
        """Handler untuk intent rekomendasi_budget."""
        budget = entities.get("budget")
        if not budget:
            return (
                "Berapa anggaran/budget yang kamu siapkan untuk membeli laptop gaming? 💰\n"
                "*Contoh:* \"Rekomendasi laptop gaming budget 15 juta\" atau \"Saran laptop under 12jt\""
            ), False

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Query laptop dengan harga <= budget, urutkan berdasarkan skor gabungan tertinggi
                cursor.execute("""
                    SELECT id, nama, merek, cpu, gpu, ram_gb, ssd_gb, layar_inci,
                           refresh_rate, harga_listing, skor_value, tier_performa
                    FROM laptops
                    WHERE harga_listing <= ?
                    ORDER BY (COALESCE(skor_value, 0) + COALESCE(tier_performa, 0)) DESC
                    LIMIT 5
                """, (budget,))
                laptops = [dict(r) for r in cursor.fetchall()]

                is_alternative = False
                if not laptops:
                    # Ambil 3 laptop termurah sebagai alternatif jika tidak ada di bawah budget
                    is_alternative = True
                    cursor.execute("""
                        SELECT id, nama, merek, cpu, gpu, ram_gb, ssd_gb, layar_inci,
                               refresh_rate, harga_listing, skor_value, tier_performa
                        FROM laptops
                        ORDER BY harga_listing ASC
                        LIMIT 3
                    """)
                    laptops = [dict(r) for r in cursor.fetchall()]

            if not laptops:
                return "Maaf, database laptop saat ini sedang tidak memiliki data.", True

            lines = []
            if is_alternative:
                lines.append(
                    f"Saat ini belum ada laptop gaming di database kami dengan harga di bawah **{format_rupiah(budget)}**.\n"
                    "Namun, berikut adalah **3 rekomendasi laptop termurah** dengan performa gaming terbaik sebagai alternatif:"
                )
            else:
                lines.append(
                    f"Berikut adalah rekomendasi laptop gaming terbaik untuk budget hingga **{format_rupiah(budget)}** "
                    "(diurutkan berdasarkan kombinasi performa dan value terbaik):"
                )

            lines.append("")
            for idx, lap in enumerate(laptops, 1):
                nama = lap.get("nama", "Gaming Laptop").strip()
                merek = lap.get("merek", "")
                cpu = lap.get("cpu", "-")
                gpu = lap.get("gpu", "-")
                ram = lap.get("ram_gb", "-")
                ssd = lap.get("ssd_gb", "-")
                layar = lap.get("layar_inci", "-")
                hz = lap.get("refresh_rate", "-")
                harga = format_rupiah(lap.get("harga_listing", 0))

                lines.append(
                    f"**{idx}. {nama}**\n"
                    f"   • **Harga**: {harga}\n"
                    f"   • **Spesifikasi**: {cpu} | {gpu} | RAM {ram}GB | SSD {ssd}GB\n"
                    f"   • **Layar**: {layar}\" ({hz}Hz)"
                )

            lines.append("\n*Catatan: Harga dan ketersediaan stok dapat berubah sewaktu-waktu di pasaran.*")
            return "\n".join(lines), True

        except Exception as e:
            return f"Terjadi kesalahan saat mencari rekomendasi laptop: {e}", False

    def handle_bandingkan(self, entities: Dict[str, Any], raw_text: str) -> Tuple[str, bool]:
        """Handler untuk intent bandingkan (2 GPU, 2 CPU, atau 2 Laptop)."""
        raw_gpus = entities.get("_raw_gpus", [])
        raw_cpus = entities.get("_raw_cpus", [])

        # KASUS 1: Membandingkan 2 GPU
        if len(raw_gpus) >= 2:
            gpu1, gpu2 = raw_gpus[0], raw_gpus[1]
            t1, t2 = gpu1["tier"], gpu2["tier"]
            diff = abs(t1 - t2)
            name1 = gpu1["name"].upper()
            name2 = gpu2["name"].upper()

            if t1 > t2:
                winner = name1
                loser = name2
                win_text = f"**{name1} lebih unggul** sebesar {diff} tier dibanding **{name2}**."
            elif t2 > t1:
                winner = name2
                loser = name1
                win_text = f"**{name2} lebih unggul** sebesar {diff} tier dibanding **{name1}**."
            else:
                win_text = f"Kedua kartu grafis berada di **Tier yang sama (Tier {t1})**, sehingga performanya relatif setara."

            lines = [
                f"### ⚖️ Perbandingan GPU: **{name1}** vs **{name2}**",
                f"• **{name1}**: Tier {t1} / 10",
                f"• **{name2}**: Tier {t2} / 10",
                f"• **Selisih Performa**: {diff} Tier",
                "",
                f"**Kesimpulan & Saran:**",
                win_text,
            ]
            if t1 != t2:
                lines.append(
                    f"Untuk game modern resolusi 1080p/1440p dengan framerate lebih tinggi dan ketahanan masa depan, "
                    f"**{winner}** sangat direkomendasikan jika selisih harganya masuk akal."
                )
            else:
                lines.append("Pilihlah opsi yang menawarkan harga lebih terjangkau, sistem pendingin lebih baik, atau konsumsi daya yang lebih efisien.")

            return "\n".join(lines), True

        # KASUS 2: Membandingkan 2 CPU
        if len(raw_cpus) >= 2:
            cpu1, cpu2 = raw_cpus[0], raw_cpus[1]
            t1, t2 = cpu1["tier"], cpu2["tier"]
            diff = abs(t1 - t2)
            name1 = cpu1["name"].title()
            name2 = cpu2["name"].title()

            if t1 > t2:
                winner = name1
                win_text = f"**{name1} lebih unggul** sebesar {diff} tier dibanding **{name2}**."
            elif t2 > t1:
                winner = name2
                win_text = f"**{name2} lebih unggul** sebesar {diff} tier dibanding **{name1}**."
            else:
                win_text = f"Kedua prosesor berada di **Tier yang sama (Tier {t1})**, performanya sangat berimbang."

            lines = [
                f"### ⚖️ Perbandingan CPU: **{name1}** vs **{name2}**",
                f"• **{name1}**: Tier {t1} / 10",
                f"• **{name2}**: Tier {t2} / 10",
                f"• **Selisih Performa**: {diff} Tier",
                "",
                f"**Kesimpulan & Saran:**",
                win_text,
            ]
            if t1 != t2:
                lines.append(f"**{winner}** akan memberikan performa multi-tasking dan kestabilan 1% low FPS yang lebih baik.")
            return "\n".join(lines), True

        # KASUS 3: Membandingkan Produk Laptop di SQLite
        matched_laptops = entities.get("_raw_laptops") or self.extract_laptops_from_db(raw_text)
        if len(matched_laptops) >= 2:
            lap1, lap2 = matched_laptops[0], matched_laptops[1]
            h1 = format_rupiah(lap1.get("harga_listing", 0))
            h2 = format_rupiah(lap2.get("harga_listing", 0))

            val1 = lap1.get("skor_value") or 0
            val2 = lap2.get("skor_value") or 0
            perf1 = lap1.get("tier_performa") or 0
            perf2 = lap2.get("tier_performa") or 0

            lines = [
                f"### ⚖️ Perbandingan Laptop: **{lap1['nama'].strip()}** vs **{lap2['nama'].strip()}**\n",
                f"**1. {lap1['nama'].strip()}**",
                f"   • **Harga**: {h1}",
                f"   • **Spesifikasi**: {lap1['cpu']} | {lap1['gpu']} | RAM {lap1['ram_gb']}GB | SSD {lap1['ssd_gb']}GB",
                f"   • **Layar**: {lap1['layar_inci']}\" {lap1['refresh_rate']}Hz",
                f"   • **Skor Value**: {val1:.2f} | **Tier Performa**: {perf1}",
                "",
                f"**2. {lap2['nama'].strip()}**",
                f"   • **Harga**: {h2}",
                f"   • **Spesifikasi**: {lap2['cpu']} | {lap2['gpu']} | RAM {lap2['ram_gb']}GB | SSD {lap2['ssd_gb']}GB",
                f"   • **Layar**: {lap2['layar_inci']}\" {lap2['refresh_rate']}Hz",
                f"   • **Skor Value**: {val2:.2f} | **Tier Performa**: {perf2}",
                "",
                "**Kesimpulan:**",
            ]

            if perf1 > perf2:
                lines.append(f"• Dari sisi **performa gaming murni**, **{lap1['nama'].strip()}** lebih unggul.")
            elif perf2 > perf1:
                lines.append(f"• Dari sisi **performa gaming murni**, **{lap2['nama'].strip()}** lebih unggul.")

            if val1 > val2:
                lines.append(f"• Dari sisi **value for money**, **{lap1['nama'].strip()}** menawarkan perbandingan harga dan performa yang lebih efisien.")
            elif val2 > val1:
                lines.append(f"• Dari sisi **value for money**, **{lap2['nama'].strip()}** menawarkan perbandingan harga dan performa yang lebih efisien.")

            return "\n".join(lines), True

        # Jika entitas perbandingan belum lengkap
        return (
            "Kamu ingin membandingkan apa nih? ⚖️\n"
            "Kamu bisa membandingkan:\n"
            "• **2 GPU**: misal *\"Mending RTX 3050 vs RTX 4060?\"*\n"
            "• **2 CPU**: misal *\"Bagus mana i5 12400F vs Ryzen 5 5600?\"*\n"
            "• **2 Laptop**: misal *\"Bandingkan Lenovo LOQ vs HP Victus\"*"
        ), False

    def handle_bottleneck(self, entities: Dict[str, Any]) -> Tuple[str, bool]:
        """Handler untuk intent bottleneck (evaluasi keseimbangan CPU dan GPU)."""
        raw_cpus = entities.get("_raw_cpus", [])
        raw_gpus = entities.get("_raw_gpus", [])

        if not raw_cpus or not raw_gpus:
            return (
                "Untuk menghitung potensi bottleneck, saya membutuhkan informasi **CPU** dan **GPU** kamu. ⚡\n"
                "*Contoh:* \"Apakah i3 gen 10 dan RTX 4070 bakal bottleneck?\""
            ), False

        cpu_item = raw_cpus[0]
        gpu_item = raw_gpus[0]

        cpu_name = cpu_item["name"].title()
        gpu_name = gpu_item["name"].upper()
        cpu_tier = cpu_item["tier"]
        gpu_tier = gpu_item["tier"]

        diff = abs(cpu_tier - gpu_tier)

        lines = [
            f"### ⚡ Analisis Bottleneck: **{cpu_name}** + **{gpu_name}**",
            f"• **CPU**: {cpu_name} ➔ **Tier {cpu_tier}** / 10",
            f"• **GPU**: {gpu_name} ➔ **Tier {gpu_tier}** / 10",
            f"• **Selisih Level**: {diff} Tier",
            "",
        ]

        if diff <= 1:
            status = "✅ **Kombinasi Seimbang (Optimal)**"
            penjelasan = (
                f"Kombinasi prosesor **{cpu_name}** dan kartu grafis **{gpu_name}** sangat serasi dan seimbang! "
                "Kedua komponen dapat saling mendukung secara maksimal tanpa adanya penahanan performa "
                "yang berarti pada berbagai jenis game."
            )
            saran = "Tidak perlu melakukan upgrade, racikan spek ini sudah sangat proporsional untuk gaming!"
        else:
            status = "⚠️ **Terjadi Bottleneck Signifikan**"
            if cpu_tier < gpu_tier:
                penjelasan = (
                    f"**CPU ({cpu_name}) lebih lemah** dibanding **GPU ({gpu_name})** dengan selisih {diff} tier. "
                    "Kondisi ini menyebabkan *CPU Bottleneck* (CPU bekerja 100% sementara GPU tidak dapat mengeluarkan "
                    "seluruh potensinya, terutama pada game bertempo cepat atau resolusi 1080p)."
                )
                target_tier = max(1, gpu_tier - 1)
                saran = (
                    f"Disarankan untuk meng-upgrade prosesor ke minimal **Tier {target_tier}** atau setara "
                    f"**Tier {gpu_tier}** (misal keluarga Core i5/i7 generasi lebih baru atau Ryzen 5/7) "
                    "agar performa kartu grafis terbuka penuh."
                )
            else:
                penjelasan = (
                    f"**GPU ({gpu_name}) lebih lemah** dibanding **CPU ({cpu_name})** dengan selisih {diff} tier. "
                    "Meskipun tidak merugikan secara sistem, potensi prosesor yang kencang ini tertahan oleh kemampuan grafis."
                )
                target_tier = max(1, cpu_tier - 1)
                saran = (
                    f"Jika ingin performa gaming lebih maksimal, disarankan meningkatkan GPU ke minimal **Tier {target_tier}** "
                    f"atau **Tier {cpu_tier}**."
                )

        lines.append(f"**Status:** {status}")
        lines.append(f"**Analisis:** {penjelasan}")
        lines.append(f"**Saran Upgrade:** {saran}")

        return "\n".join(lines), True

    def handle_fallback(self, user_message: str, intent_terdekat: str = "unknown") -> str:
        """Handler untuk pertanyaan di luar jangkauan atau entitas tidak lengkap."""
        return (
            "Maaf, saya belum memahami pertanyaan Anda atau informasinya kurang lengkap. 🙏\n\n"
            "Sebagai **GamerGear AI**, saya dapat membantu Anda seputar topik berikut:\n"
            "1. **Cek Kemampuan Game**: *\"Kuat main GTA V pakai i5 gen 12 RTX 3050 RAM 16GB?\"*\n"
            "2. **Rekomendasi Laptop Budget**: *\"Rekomendasi laptop gaming budget 15 juta\"*\n"
            "3. **Perbandingan Komponen**: *\"Mending RTX 3050 vs RTX 4060?\"*\n"
            "4. **Cek Bottleneck**: *\"Apakah i3 gen 10 dan RTX 4070 bakal bottleneck?\"*\n\n"
            "Silakan coba tanyakan kembali dengan format di atas ya!"
        )

    # =========================================================================
    # METODE UTAMA: PROCESS_MESSAGE
    # =========================================================================

    def process_message(self, user_message: str) -> Dict[str, Any]:
        """
        Memproses pesan teks pengguna dan mengembalikan respons terstruktur.

        Returns:
            dict: {
                "intent": "nama_intent",
                "response": "teks jawaban ramah dalam Bahasa Indonesia",
                "entities": {...},
                "answered": True/False
            }
        """
        raw_text = (user_message or "").strip()
        if not raw_text:
            resp = self.handle_sapaan_bantuan({})
            return {
                "intent": "sapaan_bantuan",
                "response": resp,
                "entities": {},
                "answered": True
            }

        text_lower = raw_text.lower()
        entities = self.extract_all_entities(raw_text)

        # Siapkan dictionary entitas publik (tanpa internal key dengan awalan '_')
        public_entities = {
            "ram_gb": entities["ram_gb"],
            "gpu": entities["gpu"],
            "cpu": entities["cpu"],
            "game": entities["game"],
            "budget": entities["budget"]
        }

        intent = "unknown"
        response_text = ""
        answered = False

        # ---------------------------------------------------------------------
        # 1. CEK INTENT: sapaan_bantuan
        # ---------------------------------------------------------------------
        # Trigger: halo, hai, hi, pagi, siang, malam, help, bantuan, fitur, bisa apa
        is_greeting = bool(re.search(
            r'\b(halo|hai|hi|pagi|siang|sore|malam|help|bantuan|fitur|bisa apa|menu|panduan|assalamualaikum)\b',
            text_lower
        ))
        has_specific_action = bool(
            entities["budget"] or entities["_raw_game"] or
            re.search(r'\b(vs|versus|banding|mending|bottleneck|kuat|lancar|fps)\b', text_lower)
        )

        if is_greeting and not has_specific_action:
            intent = "sapaan_bantuan"
            response_text = self.handle_sapaan_bantuan(public_entities)
            answered = True

        # ---------------------------------------------------------------------
        # 2. CEK INTENT: bottleneck
        # ---------------------------------------------------------------------
        # Trigger: kata kunci (bottleneck, tertahan, imbang, seimbang, cocok gak, pasang) + cpu + gpu
        elif (
            re.search(r'\b(bottleneck|tertahan|imbang|seimbang|cocok gak|cocok ga|pasang|ngangkat)\b', text_lower)
            and entities["_raw_cpus"] and entities["_raw_gpus"]
        ):
            intent = "bottleneck"
            response_text, answered = self.handle_bottleneck(entities)

        # ---------------------------------------------------------------------
        # 3. CEK INTENT: bandingkan
        # ---------------------------------------------------------------------
        # Trigger: kata kunci (banding, vs, versus, mending, bagus mana, beda) + minimal 2 entitas GPU/CPU atau laptop
        elif (
            re.search(r'\b(banding|bandingkan|vs|versus|mending|bagus mana|beda|perbedaan|perbandingan|pilih mana)\b', text_lower)
            and (
                len(entities["_raw_gpus"]) >= 2
                or len(entities["_raw_cpus"]) >= 2
                or len(entities.get("_raw_laptops", [])) >= 2
            )
        ):
            intent = "bandingkan"
            response_text, answered = self.handle_bandingkan(entities, raw_text)

        # ---------------------------------------------------------------------
        # 4. CEK INTENT: rekomendasi_budget
        # ---------------------------------------------------------------------
        # Trigger: kata kunci (rekomendasi, saran, budget, harga, dibawah, under, beli) + entitas budget
        elif (
            entities["budget"] is not None
            and (
                re.search(r'\b(rekomendasi|rekomendasikan|saran|sarankan|budget|harga|dibawah|under|beli|laptop murah|cari laptop)\b', text_lower)
                or not entities["_raw_game"]
            )
        ):
            intent = "rekomendasi_budget"
            response_text, answered = self.handle_rekomendasi_budget(entities)

        # ---------------------------------------------------------------------
        # 5. CEK INTENT: cek_game
        # ---------------------------------------------------------------------
        # Trigger: kata kunci (kuat, lancar, bisa main, spek, spec, jalan, fps) ATAU terdeteksi entitas game
        elif (
            entities["_raw_game"] is not None
            or re.search(r'\b(kuat|lancar|bisa main|spek|spec|spesifikasi|jalan|fps|ngangkat|mampu main)\b', text_lower)
        ):
            intent = "cek_game"
            response_text, answered = self.handle_cek_game(entities, raw_text)

        # ---------------------------------------------------------------------
        # 6. FALLBACK / UNKNOWN
        # ---------------------------------------------------------------------
        else:
            intent = "unknown"
            answered = False
            # Tebak intent terdekat untuk pencatatan
            intent_terdekat = "unknown"
            if re.search(r'\b(laptop|spek|rekomendasi|budget|harga)\b', text_lower):
                intent_terdekat = "rekomendasi_budget"
            elif re.search(r'\b(game|main|fps)\b', text_lower):
                intent_terdekat = "cek_game"
            elif re.search(r'\b(vs|banding|mending)\b', text_lower):
                intent_terdekat = "bandingkan"
            elif re.search(r'\b(bottleneck|cocok)\b', text_lower):
                intent_terdekat = "bottleneck"

            response_text = self.handle_fallback(raw_text, intent_terdekat)
            self._log_unanswered(raw_text, intent_terdekat)

        # Catat setiap chat ke chat_logs
        self._log_chat(raw_text, intent, answered)

        return {
            "intent": intent,
            "response": response_text,
            "entities": public_entities,
            "answered": answered
        }


# =============================================================================
# SIMULASI TESTING VIA TERMINAL
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🤖 SIMULASI CHATBOT ENGINE GAMERGEAR (6 CONTOH UJI)")
    print("=" * 70)

    engine = ChatbotEngine()

    test_queries = [
        # 1. Sapaan / Bantuan
        "Halo selamat pagi, fitur apa saja yang bisa dibantu?",
        # 2. Cek Game
        "Laptop saya Core i5 gen 12, RTX 3050, dan RAM 16GB apakah kuat main Black Myth Wukong?",
        # 3. Rekomendasi Budget
        "Rekomendasi laptop gaming budget 15 juta yang bagus apa ya?",
        # 4. Bandingkan
        "Mending mana RTX 3050 vs RTX 4060 untuk laptop gaming?",
        # 5. Bottleneck
        "Apakah Core i3 gen 10 dan RTX 4070 bakal bottleneck?",
        # 6. Fallback / Unknown
        "Berapa harga tiket pesawat ke Tokyo bulan depan?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}] USER: {query}")
        result = engine.process_message(query)
        print(f"INTENT  : {result['intent']}")
        print(f"ANSWERED: {result['answered']}")
        print(f"ENTITIES: {result['entities']}")
        print("-" * 50)
        print("BOT RESPONSE:")
        print(result["response"])
        print("=" * 70)
