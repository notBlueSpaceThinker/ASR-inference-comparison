from pathlib import Path

ROOT_PATH = Path(__file__).parent
DATA_PATH = ROOT_PATH / "data"
DATA_PATH.mkdir(parents=True, exist_ok=True)