"""AU Prediction module."""

import numpy as np
import pandas as pd
import torch
from feat import Detector
from feat.utils.image_operations import compute_original_image_size
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from ..dataset import GenDataset


class AUPredictor:
    """Predictor class for predicting Facial Action Units (AUs)."""

    def __init__(
        self,
        landmark_model: str = "mobilefacenet",
        au_model: str = "xgb",
        emotion_model: str = "resmasknet",
        identity_model: str = "facenet",
        device: str = "cuda",
    ):
        """Initialize the AUPredictor class."""
        self.device = device
        self.model = Detector(
            landmark_model=landmark_model,
            au_model=au_model,
            emotion_model=emotion_model,
            identity_model=identity_model,
            device=self.device,
        )

    def predict_single(
        self,
        image: torch.Tensor,
        face_threshold: float = 0.9,
        face_identity_threshold: float = 0.8,
        num_workers: int = 0,
        pin_memory: bool = False,
        progress_bar: bool = True,
    ):
        """Predicts the Facial Action Units (AUs) in a given image.

        Args:
            image: The image to predict on.
            face_threshold: The minimum confidence threshold for face detection.
            face_identity_threshold: The minimum confidence threshold for face identity.
            num_workers: The number of workers to use for data loading.
            pin_memory: Whether to pin memory for faster data transfer.
            progress_bar: Whether to show a progress bar.
        """
        dataset = GenDataset(image)
        predictions = self.predict(
            dataset=dataset,
            face_threshold=face_threshold,
            face_identity_threshold=face_identity_threshold,
            num_workers=num_workers,
            pin_memory=pin_memory,
            batch_size=1,
            progress_bar=progress_bar,
        )
        return predictions[0]

    def predict(
        self,
        dataset: Dataset,
        face_threshold: float = 0.9,
        face_identity_threshold: float = 0.8,
        num_workers: int = 0,
        pin_memory: bool = False,
        batch_size: int = 32,
        progress_bar: bool = True,
    ):
        """Predicts the Facial Action Units (AUs) in a batch of images.

        Args:
            dataset: The dataset to predict on.
            face_threshold: The minimum confidence threshold for face detection.
            face_identity_threshold: The minimum confidence threshold for face identity.
            num_workers: The number of workers to use for data loading.
            pin_memory: Whether to pin memory for data loading.
            batch_size: The batch size to use for prediction.
            progress_bar: Whether to show a progress bar during prediction.
        """
        self.batch_size = batch_size
        data_loader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.data_loader = tqdm(data_loader) if progress_bar else data_loader

        results = self._predict_batches(
            face_threshold=face_threshold,
            face_identity_threshold=face_identity_threshold,
        )

        # Filter for largest face per image
        largest_faces = self._filter_largest_faces(results)

        # Extract AU predictions
        au_scores = largest_faces.aus
        return torch.tensor(au_scores.values, dtype=torch.float32)

    def _filter_largest_faces(self, results: pd.DataFrame):
        """Filter results to keep only the largest face per image."""
        results["face_area"] = results["FaceRectWidth"] * results["FaceRectHeight"]

        # Group by image ID and keep only the row with the largest face area
        largest_faces = results.loc[results.groupby("input")["face_area"].idxmax()]

        # Drop the temporary face_area column
        largest_faces = largest_faces.drop("face_area", axis=1).reset_index(drop=True)

        return largest_faces

    def _predict_batches(
        self, face_threshold: float = 0.9, face_identity_threshold: float = 0.8
    ):
        """Predicts the Facial Action Units (AUs) in a batch of images.

        modified from https://github.com/cosanlab/py-feat/blob/c4f6364299ea2258ae1e73ed73c95750a18bff3e/feat/detector.py#L513.

        Args:
            face_threshold: Threshold for face detection.
            face_identity_threshold: Threshold for face identity.

        Returns:
            torch.Tensor: Predicted Facial Action Units (AUs).
        """
        frame_counter = 0
        batch_output = []

        for _, batch_data in enumerate(self.data_loader):
            faces_data = self.model.detect_faces(
                batch_data["Image"],
                face_size=self.model.face_size if hasattr(self, "face_size") else 112,
                face_detection_threshold=face_threshold,
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

            # Invert the face boxes and landmarks based on the padded output size
            for j, frame_idx in enumerate(batch_results["frame"].unique()):
                batch_results.loc[
                    batch_results["frame"] == frame_idx, ["FrameHeight", "FrameWidth"]
                ] = (
                    compute_original_image_size(batch_data)[j, :]
                    .repeat(
                        len(
                            batch_results.loc[
                                batch_results["frame"] == frame_idx, "frame"
                            ]
                        ),
                        1,
                    )
                    .numpy()
                )
                batch_results.loc[batch_results["frame"] == frame_idx, "FaceRectX"] = (
                    batch_results.loc[batch_results["frame"] == frame_idx, "FaceRectX"]
                    - batch_data["Padding"]["Left"].detach().numpy()[j]
                ) / batch_data["Scale"].detach().numpy()[j]
                batch_results.loc[batch_results["frame"] == frame_idx, "FaceRectY"] = (
                    batch_results.loc[batch_results["frame"] == frame_idx, "FaceRectY"]
                    - batch_data["Padding"]["Top"].detach().numpy()[j]
                ) / batch_data["Scale"].detach().numpy()[j]
                batch_results.loc[
                    batch_results["frame"] == frame_idx, "FaceRectWidth"
                ] = (
                    (
                        batch_results.loc[
                            batch_results["frame"] == frame_idx, "FaceRectWidth"
                        ]
                    )
                    / batch_data["Scale"].detach().numpy()[j]
                )
                batch_results.loc[
                    batch_results["frame"] == frame_idx, "FaceRectHeight"
                ] = (
                    (
                        batch_results.loc[
                            batch_results["frame"] == frame_idx, "FaceRectHeight"
                        ]
                    )
                    / batch_data["Scale"].detach().numpy()[j]
                )

                for i in range(68):
                    batch_results.loc[batch_results["frame"] == frame_idx, f"x_{i}"] = (
                        batch_results.loc[batch_results["frame"] == frame_idx, f"x_{i}"]
                        - batch_data["Padding"]["Left"].detach().numpy()[j]
                    ) / batch_data["Scale"].detach().numpy()[j]
                    batch_results.loc[batch_results["frame"] == frame_idx, f"y_{i}"] = (
                        batch_results.loc[batch_results["frame"] == frame_idx, f"y_{i}"]
                        - batch_data["Padding"]["Top"].detach().numpy()[j]
                    ) / batch_data["Scale"].detach().numpy()[j]

            batch_output.append(batch_results)
            frame_counter += 1 * self.batch_size

        batch_output = pd.concat(batch_output).reset_index(drop=True)

        batch_output.compute_identities(threshold=face_identity_threshold, inplace=True)
        return batch_output
