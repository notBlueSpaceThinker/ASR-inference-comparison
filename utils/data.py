from torch.utils import data
import torch
import soundfile as sf
import re


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
    def __init__(self, path_x, path_y) -> None:
        super().__init__()
        self.path_x = path_x
        self.path_y = path_y
        self._validate_dataset()

        self.data_x = sorted(list(path_x.glob("*.wav")))
        self.lenght = len(self.data_x)

    def __getitem__(self, index: int) -> tuple[tuple, str]:
        data, sample_rate = sf.read(self.data_x[index], dtype="float32")
        waveform = torch.from_numpy(data).T
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        x = (waveform, sample_rate)
        y = (self.path_y  / f"{self.data_x[index].stem}.txt").read_text(encoding="utf-8")
        return x, y

    def __len__(self):
        return self.lenght

    def _validate_dataset(self):
        """
        Validate data folders.
        """
        if not self.path_x.exists() or not self.path_y.exists():
            raise EmptyDirectoryError(
                "No x or y data directory"
            )

        found_x = []
        found_y = []
        for path in self.path_x.glob("*.wav"):
            file_name = path.name
            if not path.stat().st_size:
                raise InconsistentDatasetError(
                    f"File is empty or corrupted: {file_name}"
                )
            found_x.append(int(re.sub(r"\D", "", file_name)))
        for path in self.path_y.glob("*.txt"):
            file_name = path.name
            if not path.stat().st_size:
                raise InconsistentDatasetError(
                    f"File is empty or corrupted: {file_name}"
                )
            found_y.append(int(re.sub(r"\D", "", file_name)))

        if not found_x or not found_y:
            raise EmptyDirectoryError(
                "Directory is empty"
            )
        if len(found_x) != len(found_y):
            raise InconsistentDatasetError(
                "Number of meta and raw files is not equal"
            )
        for idx, file_id in enumerate(sorted(found_y), start=1):
            if idx != file_id:
                raise InconsistentDatasetError(
                    "Raw file IDs contain slips"
                )
        for idx, file_id in enumerate(sorted(found_y), start=1):
            if idx != file_id:
                raise InconsistentDatasetError(
                    "Meta file IDs contain slips"
                )
