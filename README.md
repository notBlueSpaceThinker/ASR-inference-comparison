
# Whisper Model Inference
The project focuses on the automated inference of the model. For each sample, the project calculates WER (Word Error Rate) and CER (Character Error Rate) metrics, as well as the model inference time.
## Project Structure
```text
.
├── data/                   # Directory for storing datasets
│   ├── data_x/             # Input data (audio files)
│   └── data_y/             # Target variables (text transcriptions)
├── models/                 # Model logic
│   ├── __init__.py
│   ├── inference.py        # Classes and functions for inference (ModelInference, run_inference)
│   └── whisper.py          # Initialization and settings for the Faster Whisper model
├── output/                 # Directory for saving execution results
│   └── whisper.csv         # Saved predictions, inference time, and metrics (WER, CER)
├── utils/                  # Auxiliary modules
│   ├── __init__.py
│   ├── data.py             # Data processing, DataLoader initialization
│   └── io.py               # Functions for saving and loading results (save_inference, load_inference)
├── config.py               # Configuration file (constants, paths, settings)
└── script.py               # Entry point to the project
````
## Environment Setup

Clone the repository:
```Bash
git clone https://github.com/notBlueSpaceThinker/ASR-inference-comparison
cd ASR-inference-comparison
```

Create and activate a virtual environment:
```Bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux/macOS:
source .venv/bin/activate
```

Install dependencies:
```Bash
python -m pip install -r requirements.txt
```
## Usage
An example is defined in the main file `script.py`.
```Python
from models.whisper import faster_model
from utils.data import data_loader
from utils.io import save_inference, load_inference
from models.inference import run_inference

def main() -> None:
    infer = run_inference("whisper", faster_model, data_loader)
    save_inference(infer)
    infer = load_inference(infer.model_name)
    
    sample = infer[0]
    print(f"Predicted: {sample['prediction']}")
    print(f"Actual: {sample['true_value']}")
    print(f"wer: {sample['wer']}")
    print(f"cer: {sample['cer']}")
    print(f"Infer: {sample['infer_runtime']}")

if __name__ == "__main__":
    main()
```
### Code Explanation:
- `infer = run_inference("whisper", faster_model, data_loader)` — runs the model inference and creates a `ModelInference` instance.
- `save_inference(infer)` — saves the model inference result to the `output` folder.
- `infer = load_inference(infer.model_name)` — creates a `ModelInference` instance from previously saved data.
- `sample = infer[0]` — retrieves a single observation (followed by visualization of the inference output values).
