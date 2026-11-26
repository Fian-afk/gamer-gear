import os
from dotenv import load_dotenv

# Membaca konfigurasi dari file .env
load_dotenv()

# Mengambil API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Memastikan API Key tersedia
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY belum dikonfigurasi!")
