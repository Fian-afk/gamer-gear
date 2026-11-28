"""
File entry point untuk menjalankan server aplikasi Flask GamerGear.
"""

import os
from app import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Server GamerGear berjalan di http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
