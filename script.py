from config import DATA_PATH
from utils import data
from models.whisper import faster_model
from utils.data import data_loader
from models.inference import run_inference

PATH_X = DATA_PATH / "data_x"
PATH_Y = DATA_PATH / "data_y"

def main() -> None:
    infer = run_inference("whisper", faster_model, data_loader)
    for sample in infer:
        print(f"Predicted {sample["prediction"]}")
        print(f"Actual {sample["true_value"]}")
        print(f"wer: {sample["wer"]}")
        print(f"cer: {sample["cer"]}")
        print(f"Infer: {sample["infer_runtime"]}")
        print()

if __name__ == "__main__":
    main()