"""
Modul Endpoint REST API Flask untuk GamerGear (app/routes.py).
Menyediakan endpoint web, estimasi harga ML, rekomendasi laptop,
analisis sentimen ulasan & klaster, chatbot rule-based, dan komparasi laptop.
"""

import os
import re
import sqlite3
import joblib
import pandas as pd
from typing import Optional, Dict, Any, List
from flask import Blueprint, render_template, request, jsonify, current_app

from app.services.chatbot_engine import ChatbotEngine, format_rupiah


# Blueprint Flask
main_bp = Blueprint("main", __name__)

# Lokasi file aman
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(BASE_DIR, "data", "gamergear.db")
MODEL_PATH = os.path.join(BASE_DIR, "model", "laptop_price_model.pkl")

# Cache model ML dan mesin Chatbot agar efisien
_cached_price_model = None
_cached_chatbot_engine = None


def get_db_connection() -> sqlite3.Connection:
    """Membuat koneksi ke database SQLite lokal gamergear.db."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_price_model():
    """Memuat model regresi estimasi harga (laptop_price_model.pkl) secara lazy-load."""
    global _cached_price_model
    if _cached_price_model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file tidak ditemukan di: {MODEL_PATH}")
        _cached_price_model = joblib.load(MODEL_PATH)
    return _cached_price_model


def get_chatbot_engine() -> ChatbotEngine:
    """Memuat instance ChatbotEngine secara lazy-load."""
    global _cached_chatbot_engine
    if _cached_chatbot_engine is None:
        _cached_chatbot_engine = ChatbotEngine(db_path=DB_PATH)
    return _cached_chatbot_engine


def parse_budget_input(budget_raw: Any) -> Optional[int]:
    """Helper untuk mem-parsing input budget angka maupun teks."""
    if budget_raw is None:
        return None
    if isinstance(budget_raw, (int, float)):
        return int(budget_raw)

    text = str(budget_raw).strip()
    # Gunakan extractor dari chatbot engine untuk mendukung '15 juta', '15jt', '15.000.000'
    engine = get_chatbot_engine()
    budget_parsed = engine.extract_budget(text)
    if budget_parsed is not None:
        return budget_parsed

    # Fallback: hilangkan karakter non-digit
    digits_only = re.sub(r"\D", "", text)
    if digits_only:
        return int(digits_only)
    return None


# =============================================================================
# 1. GET /: Merender Halaman Utama (index.html)
# =============================================================================

@main_bp.route("/", methods=["GET"])
def index():
    """Merender halaman antarmuka web utama GamerGear."""
    return render_template("index.html")


# =============================================================================
# 2. POST /api/predict: Estimasi Harga Wajar Laptop Gaming (ML Model)
# =============================================================================

@main_bp.route("/api/predict", methods=["POST"])
def predict_price():
    """
    Memuat laptop_price_model.pkl untuk menghitung estimasi harga wajar laptop
    dari spesifikasi input form (RAM, SSD, Layar, Refresh Rate, Tier Performa).
    """
    try:
        data = request.get_json(silent=True) or request.form.to_dict()
        if not data:
            return jsonify({
                "success": False,
                "error": "Body request tidak boleh kosong. Kirimkan data JSON atau Form."
            }), 400

        # Validasi & parsing fitur spesifikasi
        try:
            ram_gb = int(data.get("ram_gb", 0))
            ssd_gb = int(data.get("ssd_gb", 0))
            layar_inci = float(data.get("layar_inci", 0))
            refresh_rate = int(data.get("refresh_rate", 0))
        except (ValueError, TypeError) as e:
            return jsonify({
                "success": False,
                "error": f"Format spesifikasi numerik tidak valid: {e}"
            }), 400

        if ram_gb <= 0 or ssd_gb <= 0 or layar_inci <= 0 or refresh_rate <= 0:
            return jsonify({
                "success": False,
                "error": "Nilai spesifikasi (RAM, SSD, Layar, Refresh Rate) harus lebih besar dari 0."
            }), 400

        # Tier Performa: jika diberikan langsung, gunakan nilainya.
        # Jika tidak, coba estimasi dari CPU dan GPU yang dikirimkan.
        tier_performa = data.get("tier_performa")
        if tier_performa is not None and str(tier_performa).strip() != "":
            try:
                tier_performa = float(tier_performa)
            except (ValueError, TypeError):
                tier_performa = 3.0
        else:
            cpu_name = str(data.get("cpu", "")).strip()
            gpu_name = str(data.get("gpu", "")).strip()
            engine = get_chatbot_engine()
            c_info = engine.extract_cpus(cpu_name) if cpu_name else []
            g_info = engine.extract_gpus(gpu_name) if gpu_name else []

            c_tier = c_info[0]["tier"] if c_info else 3
            g_tier = g_info[0]["tier"] if g_info else 3
            tier_performa = round((c_tier + g_tier) / 2.0, 1)

        # Siapkan DataFrame input sesuai urutan fitur saat training
        features = ["ram_gb", "ssd_gb", "layar_inci", "refresh_rate", "tier_performa"]
        input_df = pd.DataFrame([{
            "ram_gb": ram_gb,
            "ssd_gb": ssd_gb,
            "layar_inci": layar_inci,
            "refresh_rate": refresh_rate,
            "tier_performa": tier_performa
        }])[features]

        # Prediksi harga wajar dengan model Random Forest Regressor
        model = get_price_model()
        predicted_raw = model.predict(input_df)[0]
        harga_prediksi = int(round(predicted_raw))

        # Analisis Value Deal jika pengguna menyertakan harga_listing
        skor_value = None
        status_value = None
        harga_listing_raw = data.get("harga_listing")
        harga_listing = parse_budget_input(harga_listing_raw)

        if harga_listing and harga_prediksi > 0:
            skor_value = round((harga_prediksi - harga_listing) / harga_prediksi, 4)
            if skor_value > 0.15:
                status_value = "Sangat Menguntungkan (Great Deal / Undervalued)"
            elif skor_value >= 0.0:
                status_value = "Harga Wajar (Fair Price)"
            elif skor_value >= -0.15:
                status_value = "Agak Mahal (Slightly Overpriced)"
            else:
                status_value = "Kurang Disarankan (Overpriced)"

        return jsonify({
            "success": True,
            "harga_prediksi": harga_prediksi,
            "harga_prediksi_formatted": format_rupiah(harga_prediksi),
            "harga_listing": harga_listing,
            "harga_listing_formatted": format_rupiah(harga_listing) if harga_listing else None,
            "skor_value": skor_value,
            "status_value": status_value,
            "features_used": {
                "ram_gb": ram_gb,
                "ssd_gb": ssd_gb,
                "layar_inci": layar_inci,
                "refresh_rate": refresh_rate,
                "tier_performa": tier_performa
            }
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Terjadi kesalahan saat memproses estimasi harga: {str(e)}"
        }), 500


# =============================================================================
# 3. GET /api/recommend?budget=: Rekomendasi Laptop Berdasarkan Anggaran
# =============================================================================

@main_bp.route("/api/recommend", methods=["GET"])
def recommend_laptops():
    """
    Mengambil daftar laptop dari tabel laptops di gamergear.db sesuai anggaran (harga_listing <= budget)
    dan mengurutkannya berdasarkan skor gabungan tertinggi (skor_value + tier_performa).
    Jika tidak ada di bawah budget, tampilkan 3 laptop termurah sebagai alternatif.
    """
    budget_raw = request.args.get("budget")
    if not budget_raw:
        return jsonify({
            "success": False,
            "error": "Parameter query 'budget' wajib disertakan. Contoh: /api/recommend?budget=15000000"
        }), 400

    budget = parse_budget_input(budget_raw)
    if budget is None or budget <= 0:
        return jsonify({
            "success": False,
            "error": "Format parameter 'budget' tidak valid. Contoh: 15000000 atau 15jt"
        }), 400

    limit = request.args.get("limit", default=5, type=int)
    limit = max(1, min(limit, 20))

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Query laptop di bawah budget
            cursor.execute("""
                SELECT id, nama, merek, cpu, gpu, ram_gb, ssd_gb, layar_inci,
                       refresh_rate, harga_listing, harga_prediksi, skor_value, tier_performa,
                       (COALESCE(skor_value, 0) + COALESCE(tier_performa, 0)) AS skor_gabungan
                FROM laptops
                WHERE harga_listing <= ?
                ORDER BY (COALESCE(skor_value, 0) + COALESCE(tier_performa, 0)) DESC
                LIMIT ?
            """, (budget, limit))
            rows = cursor.fetchall()

            is_alternative = False
            if not rows:
                is_alternative = True
                cursor.execute("""
                    SELECT id, nama, merek, cpu, gpu, ram_gb, ssd_gb, layar_inci,
                           refresh_rate, harga_listing, harga_prediksi, skor_value, tier_performa,
                           (COALESCE(skor_value, 0) + COALESCE(tier_performa, 0)) AS skor_gabungan
                    FROM laptops
                    ORDER BY harga_listing ASC
                    LIMIT 3
                """)
                rows = cursor.fetchall()

        laptops_data = []
        for r in rows:
            laptops_data.append({
                "id": r["id"],
                "nama": r["nama"].strip(),
                "merek": r["merek"],
                "cpu": r["cpu"],
                "gpu": r["gpu"],
                "ram_gb": r["ram_gb"],
                "ssd_gb": r["ssd_gb"],
                "layar_inci": r["layar_inci"],
                "refresh_rate": r["refresh_rate"],
                "harga_listing": r["harga_listing"],
                "harga_listing_formatted": format_rupiah(r["harga_listing"]),
                "harga_prediksi": r["harga_prediksi"],
                "harga_prediksi_formatted": format_rupiah(r["harga_prediksi"]) if r["harga_prediksi"] else None,
                "skor_value": round(r["skor_value"], 4) if r["skor_value"] is not None else None,
                "tier_performa": r["tier_performa"],
                "skor_gabungan": round(r["skor_gabungan"], 4) if r["skor_gabungan"] is not None else None,
            })

        message = (
            f"Ditemukan {len(laptops_data)} rekomendasi laptop untuk budget {format_rupiah(budget)}."
            if not is_alternative
            else f"Tidak ada laptop di bawah budget {format_rupiah(budget)}. Menampilkan 3 laptop termurah sebagai alternatif."
        )

        return jsonify({
            "success": True,
            "budget": budget,
            "budget_formatted": format_rupiah(budget),
            "is_alternative": is_alternative,
            "message": message,
            "total": len(laptops_data),
            "data": laptops_data
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Gagal mengambil rekomendasi laptop: {str(e)}"
        }), 500


# =============================================================================
# 4. GET /api/sentiment/<produk_id> & GET /api/clusters: Analisis Sentimen & Klaster
# =============================================================================

@main_bp.route("/api/sentiment/<produk_id>", methods=["GET"])
@main_bp.route("/api/sentiment", methods=["GET"])
def get_sentiment(produk_id=None):
    """
    Mengambil ringkasan sentimen aspek untuk produk laptop tertentu atau keseluruhan ulasan.
    Parameter `produk_id` dapat berupa integer id laptop atau string 'all'.
    """
    if produk_id is None:
        produk_id = request.args.get("produk_id", default="all")

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            laptop_info = None
            # Jika memfilter produk spesifik
            if str(produk_id).lower() != "all":
                try:
                    p_id = int(produk_id)
                except ValueError:
                    return jsonify({
                        "success": False,
                        "error": "Parameter produk_id harus berupa angka integer atau 'all'."
                    }), 400

                cursor.execute("SELECT id, nama, merek FROM laptops WHERE id = ?", (p_id,))
                laptop_row = cursor.fetchone()
                if not laptop_row:
                    return jsonify({
                        "success": False,
                        "error": f"Produk laptop dengan ID {p_id} tidak ditemukan."
                    }), 404

                laptop_info = {
                    "id": laptop_row["id"],
                    "nama": laptop_row["nama"].strip(),
                    "merek": laptop_row["merek"]
                }

                # Cek ulasan langsung terhubung ke produk_id
                cursor.execute("""
                    SELECT s.aspek, s.label_sentimen, COUNT(*) as count, AVG(s.skor) as avg_score
                    FROM sentiment_results s
                    JOIN reviews r ON s.review_id = r.id
                    WHERE r.produk_id = ?
                    GROUP BY s.aspek, s.label_sentimen
                """, (p_id,))
                stats_rows = cursor.fetchall()

                # Jika relasi produk_id belum dipetakan, cari ulasan yang menyebut merek/nama laptop
                if not stats_rows:
                    merek = laptop_row["merek"] or ""
                    cursor.execute("""
                        SELECT s.aspek, s.label_sentimen, COUNT(*) as count, AVG(s.skor) as avg_score
                        FROM sentiment_results s
                        JOIN reviews r ON s.review_id = r.id
                        WHERE r.teks_asli LIKE ?
                        GROUP BY s.aspek, s.label_sentimen
                    """, (f"%{merek}%",))
                    stats_rows = cursor.fetchall()

            else:
                # Keseluruhan sentimen di database
                cursor.execute("""
                    SELECT aspek, label_sentimen, COUNT(*) as count, AVG(skor) as avg_score
                    FROM sentiment_results
                    GROUP BY aspek, label_sentimen
                """)
                stats_rows = cursor.fetchall()

            # Susun rekapitulasi data sentimen
            summary = {"positif": 0, "netral": 0, "negatif": 0}
            aspek_map = {}
            total_reviews = 0
            total_score_sum = 0.0

            for r in stats_rows:
                lbl = str(r["label_sentimen"]).lower()
                cnt = r["count"]
                aspek = r["aspek"]
                avg_skor = r["avg_score"] or 0.0

                if lbl in summary:
                    summary[lbl] += cnt
                total_reviews += cnt
                total_score_sum += (avg_skor * cnt)

                if aspek not in aspek_map:
                    aspek_map[aspek] = {"positif": 0, "netral": 0, "negatif": 0, "total": 0}
                if lbl in aspek_map[aspek]:
                    aspek_map[aspek][lbl] += cnt
                aspek_map[aspek]["total"] += cnt

            persentase = {
                k: round((v / total_reviews * 100), 2) if total_reviews > 0 else 0.0
                for k, v in summary.items()
            }
            skor_rata_rata = round(total_score_sum / total_reviews, 4) if total_reviews > 0 else 0.0

        return jsonify({
            "success": True,
            "produk_id": produk_id,
            "laptop": laptop_info,
            "total_ulasan": total_reviews,
            "ringkasan_sentimen": summary,
            "persentase": persentase,
            "skor_rata_rata": skor_rata_rata,
            "aspek": [
                {"nama_aspek": k, **v} for k, v in aspek_map.items()
            ]
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Gagal mengambil data sentimen: {str(e)}"
        }), 500


@main_bp.route("/api/clusters", methods=["GET"])
def get_complaint_clusters():
    """Mengambil data klaster keluhan pengguna dari tabel complaint_clusters di SQLite."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT cluster_no, label, kata_kunci, jumlah_ulasan
                FROM complaint_clusters
                ORDER BY cluster_no ASC
            """)
            rows = cursor.fetchall()

        clusters_data = []
        for r in rows:
            clusters_data.append({
                "cluster_no": r["cluster_no"],
                "label": r["label"],
                "kata_kunci": [k.strip() for k in str(r["kata_kunci"]).split(",") if k.strip()],
                "kata_kunci_str": r["kata_kunci"],
                "jumlah_ulasan": r["jumlah_ulasan"]
            })

        return jsonify({
            "success": True,
            "total_klaster": len(clusters_data),
            "data": clusters_data
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Gagal mengambil data klaster keluhan: {str(e)}"
        }), 500


# =============================================================================
# 5. POST /api/chat: Penghubung Antarmuka Web ke Chatbot Engine
# =============================================================================

@main_bp.route("/api/chat", methods=["POST"])
def chat():
    """Menghubungkan pesan chat pengguna ke mesin chatbot rule-based (chatbot_engine.py)."""
    try:
        data = request.get_json(silent=True) or request.form.to_dict()
        if not data:
            return jsonify({
                "success": False,
                "error": "Body request harus berupa JSON atau form berisi field 'message'."
            }), 400

        user_message = data.get("message") or data.get("pesan")
        if not user_message or not str(user_message).strip():
            return jsonify({
                "success": False,
                "error": "Field 'message' tidak boleh kosong."
            }), 400

        # Panggil ChatbotEngine
        engine = get_chatbot_engine()
        result = engine.process_message(str(user_message).strip())

        return jsonify({
            "success": True,
            "intent": result["intent"],
            "response": result["response"],
            "entities": result["entities"],
            "answered": result["answered"]
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Terjadi kesalahan pada mesin chatbot: {str(e)}"
        }), 500


# =============================================================================
# 6. GET /api/compare?ids= & GET /admin/unanswered
# =============================================================================

@main_bp.route("/api/compare", methods=["GET"])
def compare_laptops():
    """
    Endpoint perbandingan laptop berdasarkan daftar ID.
    Contoh: /api/compare?ids=1,2 atau /api/compare?ids=1,2,3
    """
    ids_param = request.args.get("ids")
    if not ids_param:
        return jsonify({
            "success": False,
            "error": "Parameter query 'ids' wajib diisi (minimal 2 ID dipisahkan koma). Contoh: /api/compare?ids=1,2"
        }), 400

    # Parse daftar ID
    try:
        id_list = [int(i.strip()) for i in ids_param.split(",") if i.strip()]
    except ValueError:
        return jsonify({
            "success": False,
            "error": "Format parameter 'ids' tidak valid. Gunakan angka yang dipisahkan koma."
        }), 400

    if len(id_list) < 2:
        return jsonify({
            "success": False,
            "error": "Harap sertakan minimal 2 ID laptop untuk dibandingkan."
        }), 400

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join(["?"] * len(id_list))
            cursor.execute(f"""
                SELECT id, nama, merek, cpu, gpu, ram_gb, ssd_gb, layar_inci,
                       refresh_rate, harga_listing, harga_prediksi, skor_value, tier_performa
                FROM laptops
                WHERE id IN ({placeholders})
            """, id_list)
            rows = cursor.fetchall()

        if len(rows) < 2:
            return jsonify({
                "success": False,
                "error": f"Hanya ditemukan {len(rows)} laptop dari ID yang diberikan. Minimal diperlukan 2 laptop."
            }), 404

        laptops_data = []
        for r in rows:
            laptops_data.append({
                "id": r["id"],
                "nama": r["nama"].strip(),
                "merek": r["merek"],
                "cpu": r["cpu"],
                "gpu": r["gpu"],
                "ram_gb": r["ram_gb"],
                "ssd_gb": r["ssd_gb"],
                "layar_inci": r["layar_inci"],
                "refresh_rate": r["refresh_rate"],
                "harga_listing": r["harga_listing"],
                "harga_listing_formatted": format_rupiah(r["harga_listing"]),
                "harga_prediksi": r["harga_prediksi"],
                "harga_prediksi_formatted": format_rupiah(r["harga_prediksi"]) if r["harga_prediksi"] else None,
                "skor_value": round(r["skor_value"], 4) if r["skor_value"] is not None else 0.0,
                "tier_performa": r["tier_performa"] or 0.0,
            })

        # Komparasi metrik
        cheapest = min(laptops_data, key=lambda x: x["harga_listing"])
        highest_perf = max(laptops_data, key=lambda x: x["tier_performa"])
        best_value = max(laptops_data, key=lambda x: x["skor_value"])

        return jsonify({
            "success": True,
            "count": len(laptops_data),
            "data": laptops_data,
            "analysis": {
                "cheapest": {
                    "id": cheapest["id"],
                    "nama": cheapest["nama"],
                    "harga": cheapest["harga_listing_formatted"]
                },
                "highest_performance": {
                    "id": highest_perf["id"],
                    "nama": highest_perf["nama"],
                    "tier_performa": highest_perf["tier_performa"]
                },
                "best_value": {
                    "id": best_value["id"],
                    "nama": best_value["nama"],
                    "skor_value": best_value["skor_value"]
                }
            }
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Gagal membandingkan laptop: {str(e)}"
        }), 500


@main_bp.route("/admin/unanswered", methods=["GET"])
def get_unanswered_queries():
    """
    Endpoint untuk melihat log pertanyaan yang belum terjawab atau tidak dipahami
    dari tabel unanswered_queries di SQLite.
    Dapat difilter dengan query param: ?status=pending
    """
    status_filter = request.args.get("status")
    limit = request.args.get("limit", default=50, type=int)

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            if status_filter:
                cursor.execute("""
                    SELECT id, waktu, pertanyaan, intent_terdekat, status_tinjauan
                    FROM unanswered_queries
                    WHERE status_tinjauan = ?
                    ORDER BY id DESC
                    LIMIT ?
                """, (status_filter, limit))
            else:
                cursor.execute("""
                    SELECT id, waktu, pertanyaan, intent_terdekat, status_tinjauan
                    FROM unanswered_queries
                    ORDER BY id DESC
                    LIMIT ?
                """, (limit,))

            rows = cursor.fetchall()

        queries_data = []
        for r in rows:
            queries_data.append({
                "id": r["id"],
                "waktu": r["waktu"],
                "pertanyaan": r["pertanyaan"],
                "intent_terdekat": r["intent_terdekat"],
                "status_tinjauan": r["status_tinjauan"]
            })

        return jsonify({
            "success": True,
            "total": len(queries_data),
            "status_filter": status_filter or "all",
            "data": queries_data
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Gagal mengambil log unanswered queries: {str(e)}"
        }), 500
