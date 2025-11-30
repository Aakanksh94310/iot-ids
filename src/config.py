# src/config.py

from pathlib import Path

# I want this file to be the single source of truth for paths and
# some global settings. That way if you move the project somewhere else,
# you only fix things here, not in 20 different scripts.

# This points to the project root folder:
# D:\MASTERS\CIS600-IOT INTERNET\Project\iot-ids
# I am assuming the structure:
#   project_root/
#     src/
#       config.py  <-- this file
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Where trained models and exported artifacts will go
MODELS_DIR = PROJECT_ROOT / "models"

# Where you might store experiment logs, metrics, etc.
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Main dataset paths.
# Right now these are just placeholders. Once you decide the exact filenames
# (for Bot-IoT, N-BaIoT, or your own capture), you can update them here.
BOT_IOT_CSV = RAW_DATA_DIR / "bot_iot_reduced.csv"
N_BAIOT_CSV = RAW_DATA_DIR / "n_baiot.csv"          

# Name of the label column that will exist in your processed dataset.
# We will adjust this later once you finalise your preprocessing.
LABEL_COLUMN = "label"


def ensure_directories():
    """
    Make sure the important folders exist.

    I like having this small helper so that any script
    can safely call it at the start and not worry about missing folders.
    """
    for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, EXPERIMENTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # If you run this file directly with:
    #   uv run python -m src.config
    # it will just make sure all the folders exist.
    print(f"Project root: {PROJECT_ROOT}")
    ensure_directories()
    print("Checked/created data, models, experiments folders.")
