import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import jiwer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

@dataclass
class ModelInference:
    model_name: str
    predictions: list[str]
    true_values: list[str]
    wer: list[float]
    cer: list[float]
    sample_infer_runtime: list[float]
    total_infer_runtime: float

    def __len__(self):
        return len(self.predictions)

    def __getitem__(self, index: int) -> dict:
        return {
            "model_name": self.model_name,
            "prediction": self.predictions[index],
            "true_value": self.true_values[index],
            "wer": self.wer[index],
            "cer": self.cer[index],
            "infer_runtime": self.sample_infer_runtime[index]
        }

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

def run_inference(model_name: str, model: Any, data_loader: DataLoader):
    predictions = []
    wer = []
    cer = []
    true_values = []
    infer_runtimes = []
    for data_x, data_y in tqdm(data_loader):
        waveform, sample_rate = data_x
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.squeeze().numpy()
        waveform = waveform.astype(np.float32)
        start_time = time.time()
        segments, info = model.transcribe(
            waveform,
            beam_size=5
        )
        infer_runtimes.append(time.time() - start_time)
        predicted_text = " ".join([segment.text for segment in segments])
        predictions.append(predicted_text)
        true_value = str(data_y)
        true_values.append(true_value)
        wer.append(jiwer.wer(true_value, predicted_text))
        cer.append(jiwer.cer(true_value, predicted_text))

    return ModelInference(
        model_name=model_name,
        predictions=predictions,
        true_values=true_values,
        wer=wer,
        cer=cer,
        sample_infer_runtime=infer_runtimes,
        total_infer_runtime=sum(infer_runtimes)
    )
