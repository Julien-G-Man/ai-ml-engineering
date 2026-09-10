import uvicorn
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        app_dir=str(APP_DIR),
        reload_dirs=[str(APP_DIR)],
    )
