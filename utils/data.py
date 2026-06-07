import re

import soundfile as sf
import torch
from torch.utils import data

from config import DATA_PATH

PATH_X = DATA_PATH / "data_x"
PATH_Y = DATA_PATH / "data_y"

class InconsistentDatasetError(Exception):
    """
    Raised when IDs contain slips,
    number of meta and raw files is not equal,
    files are empty.
    """

class EmptyDirectoryError(Exception):
    """
    Raised when directory is empty.
    """

class EmptyFileError(Exception):
    """
    Raised when file is empty
    """


class Dataset(data.Dataset):
    def __init__(self, path_audio, path_text) -> None:
        super().__init__()
        self.path_audio = path_audio
        self.path_text = path_text
        self._validate_dataset()

        self.data_x = sorted(list(path_audio.glob("*.wav")))
        self.lenght = len(self.data_x)

    def __getitem__(self, index: int) -> tuple[tuple, str]:
        data, sample_rate = sf.read(self.data_x[index], dtype="float32")
        waveform = torch.from_numpy(data).T
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        x = (waveform, sample_rate)
        y = (self.path_text  / f"{self.data_x[index].stem}.txt").read_text(encoding="utf-8")
        return x, y

    def __len__(self):
        return self.lenght

    def _validate_dataset(self):
        """
        Validate data folders.
        """
        if not self.path_audio.exists() or not self.path_text.exists():
            raise EmptyDirectoryError(
                "No x or y data directory"
            )

        found_audio = []
        found_text = []
        for path in self.path_audio.glob("*.wav"):
            file_name = path.name
            if not path.stat().st_size:
                raise InconsistentDatasetError(
                    f"File is empty or corrupted: {file_name}"
                )
            found_audio.append(int(re.sub(r"\D", "", file_name)))
        for path in self.path_text.glob("*.txt"):
            file_name = path.name
            if not path.stat().st_size:
                raise InconsistentDatasetError(
                    f"File is empty or corrupted: {file_name}"
                )
            found_text.append(int(re.sub(r"\D", "", file_name)))

        if not found_audio or not found_text:
            raise EmptyDirectoryError(
                "Directory is empty"
            )
        if len(found_audio) != len(found_text):
            raise InconsistentDatasetError(
                "Number of meta and raw files is not equal"
            )
        for idx, file_id in enumerate(sorted(found_audio), start=1):
            if idx != file_id:
                raise InconsistentDatasetError(
                    "Raw file IDs contain slips"
                )
        for idx, file_id in enumerate(sorted(found_text), start=1):
            if idx != file_id:
                raise InconsistentDatasetError(
                    "Meta file IDs contain slips"
                )

data_loader = data.DataLoader(Dataset(PATH_X, PATH_Y))
