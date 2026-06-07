from models.whisper import faster_model
from utils.data import data_loader
from utils.io import save_inference, load_inference
from models.inference import run_inference


def main() -> None:
    infer = run_inference("whisper", faster_model, data_loader)
    save_inference(infer)
    infer = load_inference(infer.model_name)
    sample = infer[0]
    print(f"Predicted {sample["prediction"]}")
    print(f"Actual {sample["true_value"]}")
    print(f"wer: {sample["wer"]}")
    print(f"cer: {sample["cer"]}")
    print(f"Infer: {sample["infer_runtime"]}")


if __name__ == "__main__":
    main()