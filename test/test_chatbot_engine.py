"""
Unit tests untuk ChatbotEngine GamerGear.
"""

import os
import sys

# Tambahkan root folder ke sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import sqlite3
from app.services.chatbot_engine import ChatbotEngine, format_rupiah


@pytest.fixture(scope="module")
def engine():
    return ChatbotEngine()


def test_format_rupiah():
    assert format_rupiah(15000000) == "Rp 15.000.000"
    assert format_rupiah(12500000) == "Rp 12.500.000"
    assert format_rupiah(500000) == "Rp 500.000"


def test_extract_ram(engine):
    assert engine.extract_ram("laptop ini 8gb ram") == 8
    assert engine.extract_ram("punya spek 16 gb") == 16
    assert engine.extract_ram("laptop dengan ram 32") == 32
    assert engine.extract_ram("spek 32g mantap") == 32
    assert engine.extract_ram("ram: 16 gb") == 16
    assert engine.extract_ram("ram 16gb") == 16
    assert engine.extract_ram("laptop i5 rtx 3050 4gb ram 16gb") == 16


def test_extract_budget(engine):
    assert engine.extract_budget("budget 12 juta") == 12000000
    assert engine.extract_budget("cari laptop 12jt") == 12000000
    assert engine.extract_budget("rekomendasi under 10.5 jt") == 10500000
    assert engine.extract_budget("dibawah 10,5jt") == 10500000
    assert engine.extract_budget("harga 15000000 rupiah") == 15000000
    assert engine.extract_budget("budget 15.000.000") == 15000000


def test_extract_game(engine):
    g1 = engine.extract_game("apakah kuat main Black Myth Wukong?")
    assert g1 is not None
    assert g1["nama_game"] == "Black Myth: Wukong"

    g2 = engine.extract_game("bisa main wukong gak?")
    assert g2 is not None
    assert g2["nama_game"] == "Black Myth: Wukong"

    # Toleransi typo dengan rapidfuzz (threshold >= 78)
    g3 = engine.extract_game("kuat ga buat maen valoren?")
    assert g3 is not None
    assert g3["nama_game"] == "Valorant"


def test_extract_gpu(engine):
    # Single GPU
    res_single = engine.extract_gpus("laptop dengan rtx 3050")
    assert len(res_single) == 1
    assert res_single[0]["name"] == "rtx 3050"

    # Two GPUs (untuk perbandingan)
    res_multi = engine.extract_gpus("mending rtx 3050 vs rtx 4060?")
    assert len(res_multi) >= 2
    names = [g["name"] for g in res_multi]
    assert "rtx 3050" in names
    assert "rtx 4060" in names


def test_extract_cpu(engine):
    # Tes 'i5 gen 12', 'ryzen 5', 'i3', 'core i7'
    c1 = engine.extract_cpus("spek i5 gen 12 mantap")
    assert len(c1) >= 1
    assert "i5 gen 12" in c1[0]["name"]

    c2 = engine.extract_cpus("laptop ryzen 5")
    assert len(c2) >= 1
    assert "ryzen 5" in c2[0]["name"]

    c3 = engine.extract_cpus("apakah i3 kuat?")
    assert len(c3) >= 1
    assert "i3" in c3[0]["name"]

    c4 = engine.extract_cpus("prosesor core i7")
    assert len(c4) >= 1
    assert "core i7" in c4[0]["name"]


def test_intent_sapaan_bantuan(engine):
    res = engine.process_message("Halo, selamat pagi!")
    assert res["intent"] == "sapaan_bantuan"
    assert res["answered"] is True
    assert "GamerGear AI" in res["response"]
    assert "Cek Kompatibilitas Game" in res["response"]


def test_intent_cek_game(engine):
    query = "Laptop i5 gen 12 RTX 3050 RAM 16GB kuat main Cyberpunk 2077?"
    res = engine.process_message(query)
    assert res["intent"] == "cek_game"
    assert res["answered"] is True
    assert res["entities"]["game"] == "Cyberpunk 2077"
    assert res["entities"]["ram_gb"] == 16
    assert "Cyberpunk 2077" in res["response"]


def test_intent_rekomendasi_budget(engine):
    # Normal budget
    res = engine.process_message("Rekomendasi laptop gaming budget 15 juta")
    assert res["intent"] == "rekomendasi_budget"
    assert res["answered"] is True
    assert res["entities"]["budget"] == 15000000
    assert "Rp 15.000.000" in res["response"]

    # Budget sangat kecil -> 3 alternatif termurah
    res_low = engine.process_message("Laptop gaming budget 5jt")
    assert res_low["intent"] == "rekomendasi_budget"
    assert res_low["answered"] is True
    assert "termurah" in res_low["response"]


def test_intent_bandingkan_gpu(engine):
    res = engine.process_message("Mending mana RTX 3050 vs RTX 4060?")
    assert res["intent"] == "bandingkan"
    assert res["answered"] is True
    assert isinstance(res["entities"]["gpu"], list)
    assert len(res["entities"]["gpu"]) == 2
    assert "Perbandingan GPU" in res["response"]


def test_intent_bandingkan_laptop(engine):
    res = engine.process_message("Bandingkan Lenovo LOQ vs HP Victus")
    assert res["intent"] == "bandingkan"
    assert res["answered"] is True
    assert "Perbandingan Laptop" in res["response"]


def test_intent_bottleneck(engine):
    res = engine.process_message("Apakah Core i3 gen 10 dan RTX 4070 bakal bottleneck?")
    assert res["intent"] == "bottleneck"
    assert res["answered"] is True
    assert "Bottleneck" in res["response"]
    assert res["entities"]["cpu"] is not None
    assert res["entities"]["gpu"] is not None


def test_intent_fallback_and_logging(engine):
    test_query = "Resep membuat nasi goreng spesial pedas"
    res = engine.process_message(test_query)
    assert res["intent"] == "unknown"
    assert res["answered"] is False
    assert "Maaf, saya belum memahami" in res["response"]

    # Periksa logging di SQLite
    with sqlite3.connect(engine.db_path) as conn:
        c = conn.cursor()
        c.execute("SELECT intent, dijawab FROM chat_logs WHERE pertanyaan = ?", (test_query,))
        row = c.fetchone()
        assert row is not None
        assert row[0] == "unknown"
        assert row[1] == 0

        c.execute("SELECT status_tinjauan FROM unanswered_queries WHERE pertanyaan = ?", (test_query,))
        unans = c.fetchone()
        assert unans is not None
        assert unans[0] == "pending"
