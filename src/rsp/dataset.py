"""Dataset class used with the Action Unit Predictor."""

from typing import Any, TypedDict

from torch import Tensor
from torch.utils.data import Dataset


class Sample(TypedDict):
    """Sample with image tensor and id.

    Keys:
        id: The sample id.
        image: The image tensor.
    """

    id: int
    image: Tensor


class Prediction(Sample):
    """A sample with id and predictions.

    Keys:
        predictions: The AU predictions for the sample.
    """

    predictions: Tensor


class TensorDataset(Dataset):
    """Dataset for storing tensors.

    Args:
        data: The generated samples.
    """

    def __init__(self, data: Tensor):
        """Initialize the dataset with the given data."""
        self.data = data
        if len(self.data.shape) == 3:
            self.data = self.data.unsqueeze(0)
            # Ensures batch shape if not batch shape

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.data.size(0)

    def __getitem__(self, idx) -> dict[str, Any]:
        """Return a sample from the dataset."""
        temp = {
            "Image": self.data[idx],
            "Frame": idx,
            "FileName": "tensor",
            "Scale": 1.0,
            "Padding": {"Left": 0, "Top": 0, "Right": 0, "Bottom": 0},
        }
        return temp
