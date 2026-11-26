import requests
import pandas as pd
from config import GOOGLE_API_KEY

# 1. Masukkan API Key YouTube Anda
API_KEY = GOOGLE_API_KEY

# 2. Masukkan ID Video YouTube (Contoh: video review ASUS TUF atau Lenovo Legion)
VIDEO_ID = "rW37y2_cEs8"

def get_youtube_comments(video_id, api_key, max_pages=5):
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    aspek_kunci = ['build quality', 'thermal', 'panas', 'suhu', 'baterai', 'battery', 'performa', 'fps', 'layar', 'keyboard', 
                   'engsel', 'lemot', 'batre', 'kipas', 'minus', 'suara', 'overheat', 'hang', 'lag', 'stuck', 'freeze', 
                   'fps drop', 'frame drop', 'lagging', 'throttling', 'throttle', 'overheating', 'hotspot', 'fan noise', 
                   'battery drain', 'performance', 'rekomendasi', 'saran', 'review', 'ulasan', 'masalah', 'problem', 
                   'issue', 'bug', 'error', 'crash', 'shutdown', 'restart', 'lagging', 'stuttering', 'flickering', 'artifacting', 
                   'mending', 'ram', 'storage', 'ssd', 'hdd', 'vga', 'gpu', 'cpu', 'processor', 'core', 'thread', 'clock speed', 
                   'overclock', 'undervolt', 'thermal paste', 'cooling pad', 'heatsink', 'fan speed', 'noise level', 'spek', 
                   'temperature', 'heat dissipation', 'worth it', 'kondisi', 'garansi', 'service center', 'customer support', 
                   'after sales', 'warranty', 'replacement', 'refund', 'return policy', 'shipping', 'delivery', 'packaging', 
                   'unboxing', 'setup', 'installation', 'driver update', 'firmware update', 'software update', 'gaming', 'saran', 
                   'tips', 'trik', 'cara', 'tutorial', 'panduan', 'guide', 'manual', 'budget', 'harga', 'monitor', 'laptop']
    
    dataset_komentar = []
    next_page_token = None
    
    print(f"Mulai mengambil komentar dari video ID: {video_id}...")
    
    for halaman in range(max_pages):
        params = {
            'part': 'snippet',
            'videoId': video_id,
            'key': api_key,
            'maxResults': 100, # Maksimal 100 komentar per permintaan
            'textFormat': 'plainText',
            'pageToken': next_page_token
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            items = data.get('items', [])
            
            for item in items:
                # Mengekstrak teks komentar
                komentar = item['snippet']['topLevelComment']['snippet']['textDisplay']
                komentar_lower = komentar.lower()
                
                # Memastikan komentar panjangnya wajar dan mengandung kata kunci aspek
                mengandung_aspek = any(aspek in komentar_lower for aspek in aspek_kunci)
                
                if mengandung_aspek and len(komentar) > 20:
                    # HANYA mengambil teks dan metrik keluhan, TANPA authorDisplayName (Anonim)
                    data_bersih = {
                        "ID_Komentar": item['snippet']['topLevelComment']['id'],
                        "Sumber": "YouTube",
                        "Teks_Asli": komentar,
                        "Skor_Like": item['snippet']['topLevelComment']['snippet']['likeCount']
                    }
                    dataset_komentar.append(data_bersih)
            
            next_page_token = data.get('nextPageToken')
            print(f"✔️ Halaman {halaman + 1} selesai diproses...")
            
            if not next_page_token:
                break # Berhenti jika komentar sudah habis
        else:
            print(f"Gagal memanggil API: {response.status_code} - {response.text}")
            break

    return dataset_komentar

# --- Eksekusi Program ---
if __name__ == "__main__":
    if API_KEY != "MASUKKAN_API_KEY_YOUTUBE_ANDA_DISINI":
        hasil_komentar = get_youtube_comments(VIDEO_ID, API_KEY)
        
        if hasil_komentar:
            df = pd.DataFrame(hasil_komentar)
            df.to_csv("dataset_youtube_indo_7.csv", index=False, encoding='utf-8')
            print(f"\nSelesai! {len(hasil_komentar)} ulasan anonim berhasil disimpan ke 'dataset_youtube_indo_7.csv'")
        else:
            print("\nTidak ada komentar yang sesuai kriteria aspek ditemukan.")
    else:
        print("Tolong masukkan API Key YouTube Anda ke dalam variabel API_KEY terlebih dahulu.")