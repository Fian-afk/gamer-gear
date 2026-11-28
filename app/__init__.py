"""
Inisialisasi paket aplikasi Flask GamerGear.
"""

import os
from flask import Flask


def create_app(test_config=None):
    """Factory function untuk membuat instance aplikasi Flask GamerGear."""
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static"
    )

    if test_config:
        app.config.update(test_config)

    # Daftarkan blueprint utama
    from app.routes import main_bp
    app.register_blueprint(main_bp)

    return app
