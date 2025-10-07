import torch
from datasets import load_dataset
from dotenv import load_dotenv
from feat import Detector
from torch.utils.data import DataLoader

load_dotenv()


def detect(batch, detector, batch_size):
    detections = detector.detect(
        batch,
        data_type="tensor",
        face_detection_threshold=0.9,
        batch_size=batch_size,
    )
    return detections


def main():
    batch_size = 512
    dataset = load_dataset("viga-itu/celeba-hq-512")
    dataset_torch = dataset["train"].with_format("torch")["image"]
    dataloader = DataLoader(dataset_torch, batch_size=batch_size)

    first_batch = next(iter(dataloader))

    detector = Detector(
        landmark_model="mobilefacenet",
        au_model="xgb",
        emotion_model="resmasknet",
        identity_model="facenet",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    detections = detect(first_batch, detector, batch_size)
    detections.to_csv("results/detections.csv", index=False)

    for batch in dataloader:
        detections = detect(batch, detector, batch_size)
        detections.to_csv("results/detections.csv", mode="a", header=False, index=False)


if __name__ == "__main__":
    main()
