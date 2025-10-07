import os
from pathlib import Path

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from feat import Detector

load_dotenv()


def main():
    DATA_PATH = Path("data/celeba-hq-512")

    if not DATA_PATH.is_dir():
        DATA_PATH.mkdir(parents=True, exist_ok=True)
        dataset = load_dataset("viga-itu/celeba-hq-512")
        trainset = dataset["train"]["image"]
        for i, img in enumerate(trainset):
            path = f"{i}.jpg"
            img.save(DATA_PATH / path)

    images = [
        str(DATA_PATH / img_name)
        for img_name in os.listdir(DATA_PATH)
        if img_name.endswith(".jpg")
    ]
    batch_size = 512
    detector = Detector(
        landmark_model="mobilefacenet",
        au_model="xgb",
        emotion_model="resmasknet",
        identity_model="facenet",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    detector.detect(
        images,
        face_detection_threshold=0.9,
        save="results/detections.csv",
        batch_size=batch_size,
    )


if __name__ == "__main__":
    main()
