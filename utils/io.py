from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from config import OUTPUT_PATH
from models.inference import ModelInference


def save_inference(inference: ModelInference, path: Path | str = OUTPUT_PATH) -> None:
    output_df = pd.DataFrame()
    for attr in inference.__dict__:
        attr_value = getattr(inference, attr)
        if isinstance(attr_value, Iterable) and not isinstance(attr_value, str):
            output_df[attr] = list(attr_value)
    path = Path(path)
    output_df.to_csv(path / f"{inference.model_name}.csv", index=False)

def load_inference(model_name, path: Path | str = OUTPUT_PATH) -> ModelInference:
    path = Path(path)
    input_df = pd.read_csv(OUTPUT_PATH / f"{model_name}.csv")
    return ModelInference(
        model_name,
        **input_df.to_dict(orient="list")
    )
