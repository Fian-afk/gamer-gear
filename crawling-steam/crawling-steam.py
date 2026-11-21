import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def get_popular_games():
    # 1. Meminta daftar 100 game terpopuler saat ini dari SteamSpy
    print("Meminta daftar 100 game populer...")
    url = "https://steamspy.com/api.php?request=top100forever"
    response = requests.get(url)
    return response.json() 

def parse_requirements(html_content):
    # 2. Alat bedah teks (BeautifulSoup)
    if not html_content:
        return {}
    
    soup = BeautifulSoup(html_content, 'html.parser')
    req_dict = {}
    
    # Steam selalu menaruh spesifikasi di dalam list bullet HTML (tag <li>)
    list_items = soup.find_all('li')
    
    for item in list_items:
        text = item.get_text(strip=True)
        # Mencari kata kunci spesifik dan memotong teksnya
        if "Processor:" in text or "CPU:" in text:
            req_dict["CPU"] = text.split(":", 1)[1].strip()
        elif "Memory:" in text or "RAM:" in text:
            req_dict["RAM"] = text.split(":", 1)[1].strip()
        elif "Graphics:" in text or "GPU:" in text:
            req_dict["GPU"] = text.split(":", 1)[1].strip()
        elif "Storage:" in text or "Hard Drive:" in text:
            req_dict["Storage"] = text.split(":", 1)[1].strip()
            
    return req_dict

def get_steam_game_data(app_id):
    # 3. Meminta detail spesifikasi langsung ke database internal Steam
    url = f"https://store.steampowered.com/api/appdetails?appids={app_id}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if data and str(app_id) in data and data[str(app_id)]['success']:
            game_info = data[str(app_id)]['data']
            game_name = game_info.get('name', 'Unknown')
            
            # Mengambil teks kotor spesifikasi minimum dan rekomendasi
            pc_reqs = game_info.get('pc_requirements', {})
            if isinstance(pc_reqs, dict):
                min_html = pc_reqs.get('minimum', '')
                rec_html = pc_reqs.get('recommended', '')
                
                # Mengirim teks kotor ke fungsi pemotong teks (parse_requirements)
                return {
                    "Game": game_name,
                    "Minimum": parse_requirements(min_html),
                    "Recommended": parse_requirements(rec_html)
                }
    return None

# --- PROGRAM UTAMA DIMULAI DI SINI ---
if __name__ == "__main__":
    popular_games = get_popular_games()
    database_chatbot = [] # Tempat kita menampung hasil
    
    # Kita batasi 10 game saja dulu sebagai percobaan agar cepat selesai
    # Hapus "[:10]" jika ingin mengambil seluruh 100 game
    daftar_id_game = list(popular_games.keys()) 
    
    for app_id in daftar_id_game:
        nama_game = popular_games[app_id].get('name', 'Unknown')
        print(f"Sedang mencatat spesifikasi: {nama_game}...")
        
        game_data = get_steam_game_data(app_id)
        
        if game_data and (game_data["Minimum"] or game_data["Recommended"]):
            # Menyusun data yang berantakan menjadi rapi siap masuk tabel Excel/CSV
            data_rapi = {
                "Game Name": game_data["Game"],
                "Min CPU": game_data["Minimum"].get("CPU", "Tidak Ada Info"),
                "Min RAM": game_data["Minimum"].get("RAM", "Tidak Ada Info"),
                "Min GPU": game_data["Minimum"].get("GPU", "Tidak Ada Info"),
                "Min Storage": game_data["Minimum"].get("Storage", "Tidak Ada Info"),
                "Rec CPU": game_data["Recommended"].get("CPU", "Tidak Ada Info"),
                "Rec RAM": game_data["Recommended"].get("RAM", "Tidak Ada Info"),
                "Rec GPU": game_data["Recommended"].get("GPU", "Tidak Ada Info"),
                "Rec Storage": game_data["Recommended"].get("Storage", "Tidak Ada Info"),
            }
            database_chatbot.append(data_rapi)
        
        # Jeda 1.5 detik sangat wajib agar koneksi kita tidak diblokir oleh Steam
        time.sleep(1.5)

    # 4. Menyimpan hasil tangkapan ke dalam file tabel CSV
    if database_chatbot:
        tabel = pd.DataFrame(database_chatbot)
        tabel.to_csv("database_spesifikasi_game.csv", index=False)
        print("Selesai! Buka file 'database_spesifikasi_game.csv' untuk melihat hasilnya.")