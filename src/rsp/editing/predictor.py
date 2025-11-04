"""AU Prediction module."""

import numpy as np
import pandas as pd
import torch
from feat import Detector
from feat.data import Fex
from feat.utils.image_operations import compute_original_image_size
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from rsp.constants import SUPPORTED_AUS

from ..dataset import TensorDataset


class AUPredictor:
    """Predictor class for predicting Facial Action Units (AUs)."""

    def __init__(
        self,
        landmark_model: str = "mobilefacenet",
        au_model: str = "xgb",
        emotion_model: str = "resmasknet",
        identity_model: str = "facenet",
        face_threshold: float = 0.9,
        face_identity_threshold: float = 0.8,
        num_workers: int = 0,
        pin_memory: bool = False,
        batch_size: int = 32,
        progress_bar: bool = True,
        device: str = "cuda",
    ):
        """Initialize the AUPredictor class.

        Args:
            landmark_model:
                The landmark detection model to use.
            au_model:
                The AU detection model to use.
            emotion_model:
                The emotion detection model to use.
            identity_model:
                The identity recognition model to use.
            face_threshold:
                The minimum confidence threshold for face detection.
            face_identity_threshold:
                The minimum confidence threshold for face identity.
            num_workers:
                The number of workers to use for data loading.
            pin_memory:
                Whether to pin memory for data loading.
            batch_size:
                The batch size to use for prediction.
            progress_bar:
                Whether to show a progress bar during prediction.
            device:
                The device to use for computation.
        """
        self.device = device
        if "cuda" in self.device:
            detector_device = "cuda"
        else:
            detector_device = "cpu"

        self.face_threshold = face_threshold
        self.face_identity_threshold = face_identity_threshold
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.progress_bar = progress_bar

        self.model = Detector(
            landmark_model=landmark_model,
            au_model=au_model,
            emotion_model=emotion_model,
            identity_model=identity_model,
            device=detector_device,
        )

    def predict(
        self,
        images: torch.Tensor | None = None,
        dataset: TensorDataset | None = None,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """Predicts the Facial Action Units (AUs) in a batch of images.

        Args:
            images:
                The images to predict on. Mutually exclusive with 'dataset'.
            dataset:
                The dataset to predict on. Mutually exclusive with 'images'.
            batch_size:
                The batch size to use for prediction.

        Returns:
            The predicted Facial Action Units (AUs) for the input images.

        Raises:
            ValueError:
                If both 'images' and 'dataset' are provided, or if neither are provided.
        """
        if dataset is not None and images is not None:
            raise ValueError("Only one of 'images' or 'dataset' should be provided.")
        elif dataset is None and images is None:
            raise ValueError("One of 'images' or 'dataset' must be provided.")
        elif dataset is None and isinstance(images, torch.Tensor):
            dataset = TensorDataset(images)
        else:
            raise ValueError(
                f"Invalid input type ({type(images)} for 'images', should be "
                "'torch.Tensor')"
            )

        if batch_size is not None:
            self.batch_size = batch_size

        data_loader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        self.data_loader = tqdm(data_loader) if self.progress_bar else data_loader

        results = self._predict_batches()

        batch_results = torch.full((len(dataset), len(SUPPORTED_AUS)), fill_value=-1.0)

        # Handle each image individually
        for image_idx in range(len(dataset)):
            image_results = results[results["frame"] == image_idx]
            if not image_results.isna().any().any():  # pyright: ignore[reportAttributeAccessIssue]
                largest_face = self._filter_largest_faces(image_results)
                au_scores = largest_face.aus
                batch_results[image_idx] = torch.tensor(
                    au_scores.values, dtype=torch.float32
                )
            else:
                # Probably no face detected, skip this image and set -1 for all AUs
                continue

        return batch_results

    def _filter_largest_faces(self, results: pd.DataFrame | pd.Series) -> pd.DataFrame:
        """Filter results to keep only the largest face per image."""
        largest_idx = (
            results.assign(
                face_area=results["FaceRectWidth"] * results["FaceRectHeight"]
            )
            .groupby("input")["face_area"]
            .idxmax()
        )
        return results.loc[largest_idx].reset_index(drop=True)

    def _predict_batches(self) -> pd.DataFrame:
        """Predicts the Facial Action Units (AUs) in a batch of images.

        modified from https://github.com/cosanlab/py-feat/blob/c4f6364299ea2258ae1e73ed73c95750a18bff3e/feat/detector.py#L513.

        Returns:
            torch.Tensor: Predicted Facial Action Units (AUs).
        """
        frame_counter = 0
        batch_output = []

        for _, batch_data in enumerate(self.data_loader):
            faces_data = self.model.detect_faces(
                batch_data["Image"],
                face_size=self.model.face_size if hasattr(self, "face_size") else 112,
                face_detection_threshold=self.face_threshold,
            )
            batch_results = self.model.forward(faces_data)

            # Create metadata for each frame
            file_names = []
            frame_ids = []
            for i, face in enumerate(faces_data):
                n_faces = len(face["scores"])
                current_frame_id = frame_counter + i
                frame_ids.append(np.repeat(current_frame_id, n_faces))
                file_names.append(np.repeat(batch_data["FileName"][i], n_faces))
            batch_results["input"] = np.concatenate(file_names)
            batch_results["frame"] = np.concatenate(frame_ids)

            # since height and width are given from the diffusion model and
            # we have no scaling or padding we just extract the original image size
            # and since no padding or scaling we also dont need any inversion of
            # face height + width or landmarks - was part of pyfeat code
            _, _, height, width = batch_data["Image"].shape
            batch_data["FrameHeight"] = height
            batch_data["FrameWidth"] = width

            batch_output.append(batch_results)
            frame_counter += 1 * self.batch_size

        batch_output = pd.concat(batch_output).reset_index(drop=True)

        batch_output.compute_identities(
            threshold=self.face_identity_threshold, inplace=True
        )
        return batch_output
