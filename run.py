"""Application entry point."""

import os

from app import create_app
from config import get_config

app = create_app(get_config())


if __name__ == "__main__":
    app.run(
        host=os.environ.get("FLASK_HOST", "127.0.0.1"),
        port=int(os.environ.get("FLASK_PORT", 5000)),
        debug=app.config.get("DEBUG", False),
    )
