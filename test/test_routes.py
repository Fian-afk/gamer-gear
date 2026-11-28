"""
Unit test suite untuk Flask API endpoints (app/routes.py).
"""

import os
import sys

# Tambahkan root path proyek ke sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from app import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app({"TESTING": True})
    with app.test_client() as client:
        yield client


# 1. GET /
def test_get_index(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"GamerGear" in response.data


# 2. POST /api/predict
def test_predict_price_valid(client):
    payload = {
        "ram_gb": 16,
        "ssd_gb": 512,
        "layar_inci": 15.6,
        "refresh_rate": 144,
        "tier_performa": 2.5,
        "harga_listing": 18000000
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "harga_prediksi" in data
    assert data["harga_prediksi"] > 0
    assert "harga_prediksi_formatted" in data
    assert "skor_value" in data
    assert "status_value" in data


def test_predict_price_invalid(client):
    # Payload kosong
    r1 = client.post("/api/predict", json={})
    assert r1.status_code == 400

    # Nilai non-positif
    r2 = client.post("/api/predict", json={"ram_gb": -4, "ssd_gb": 512, "layar_inci": 15.6, "refresh_rate": 144})
    assert r2.status_code == 400


# 3. GET /api/recommend
def test_recommend_laptops_valid(client):
    response = client.get("/api/recommend?budget=15000000")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["budget"] == 15000000
    assert len(data["data"]) > 0
    assert data["data"][0]["harga_listing"] <= 15000000


def test_recommend_laptops_text_budget(client):
    response = client.get("/api/recommend?budget=15jt")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["budget"] == 15000000


def test_recommend_laptops_alternative(client):
    # Budget sangat kecil -> 3 alternatif termurah
    response = client.get("/api/recommend?budget=5000000")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["is_alternative"] is True
    assert len(data["data"]) == 3


def test_recommend_laptops_missing_budget(client):
    response = client.get("/api/recommend")
    assert response.status_code == 400


# 4. GET /api/sentiment/<produk_id> & GET /api/clusters
def test_get_sentiment_product(client):
    response = client.get("/api/sentiment/1")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "ringkasan_sentimen" in data
    assert "persentase" in data


def test_get_sentiment_all(client):
    response = client.get("/api/sentiment/all")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["total_ulasan"] > 0


def test_get_sentiment_not_found(client):
    response = client.get("/api/sentiment/999999")
    assert response.status_code == 404


def test_get_clusters(client):
    response = client.get("/api/clusters")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["total_klaster"] == 4
    assert len(data["data"]) == 4
    assert "label" in data["data"][0]
    assert "kata_kunci" in data["data"][0]


# 5. POST /api/chat
def test_chat_endpoint_valid(client):
    payload = {"message": "Halo, selamat pagi!"}
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["intent"] == "sapaan_bantuan"
    assert "GamerGear AI" in data["response"]


def test_chat_endpoint_empty(client):
    response = client.post("/api/chat", json={"message": ""})
    assert response.status_code == 400


# 6. GET /api/compare?ids= & GET /admin/unanswered
def test_compare_laptops_valid(client):
    response = client.get("/api/compare?ids=1,2")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["count"] == 2
    assert "analysis" in data
    assert "cheapest" in data["analysis"]
    assert "highest_performance" in data["analysis"]
    assert "best_value" in data["analysis"]


def test_compare_laptops_invalid(client):
    # Kurang dari 2 ID
    r1 = client.get("/api/compare?ids=1")
    assert r1.status_code == 400

    # Tanpa parameter
    r2 = client.get("/api/compare")
    assert r2.status_code == 400


def test_admin_unanswered(client):
    response = client.get("/admin/unanswered")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "total" in data
    assert isinstance(data["data"], list)
