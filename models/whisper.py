import torch
from faster_whisper import WhisperModel as FasterWhisperModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

faster_model = FasterWhisperModel("base", device=device.type, compute_type="float32")

         