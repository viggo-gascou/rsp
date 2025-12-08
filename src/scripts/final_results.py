import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import load_dataset
from torchvision.transforms import PILToTensor
from tqdm import tqdm

from rsp.constants import *
from rsp.constants import AU_SUBSET
from rsp.editing import AnycostDirections
from rsp.editing.predictor import AUPredictor
from rsp.loading import load_model
from rsp.log_utils import set_logging_level

set_logging_level(logging.INFO)

path = RESULTS_DIR / "final_results/"

num_images = 100
default_seed = 10000
curr_aus = ["AU12", "AU04"]
# check if folder exists
if not path.exists():
    os.makedirs(path)
    # create subfolder for each AU
    for au in curr_aus:
        os.makedirs(path / au)
    os.makedirs(path / "Original")


if len(os.listdir(path / "Original")) > 0:
    print("Images already generated, skipping...")
else:
    sd = load_model("pixel", device="cuda", h_space="after", num_inference_steps=100)

    ad = AnycostDirections(
        sd,
        etas=None,
        num_examples=100,
        idx_size=10000,  # Size of index. The number of images sampled in total
        batch_size=32,
    )

    for i in tqdm(range(num_images)):
        curr_seed = default_seed + i
        q_original = sd.sample(seed=curr_seed)
        img_original = sd.show(q_original)
        img_original.save(path / "Original" / f"image_{i}_original.png")
        for au in curr_aus:
            n = ad.get_direction(au)
            q_edit = sd.apply_direction(q_original.copy(), n, scale=1)
            img_edit_au = sd.show(q_edit)
            img_edit_au.save(path / au / f"image_{i}_{au}_edited.png")


# original first
dataset = load_dataset(
    "imagefolder",
    data_dir=path / "Original",
)
dataset = load_dataset("imagefolder", data_dir=path / "Original")
pil_to_tensor = PILToTensor()
images_tensor = torch.stack([pil_to_tensor(img) for img in dataset["train"]["image"]])

predictor = AUPredictor(device="cuda", batch_size=16, progress_bar=True)
preds = predictor.predict(images=images_tensor, batch_size=16)
df = pd.DataFrame(preds.numpy(), columns=SUPPORTED_AUS)
df.to_csv(path / "Original" / "au_predictions.csv", index=False)

# then, get the edited images
for au in curr_aus:
    dataset = load_dataset(
        "imagefolder",
        data_dir=path / au,
    )
    dataset = load_dataset("imagefolder", data_dir=path / au)
    pil_to_tensor = PILToTensor()
    images_tensor = torch.stack(
        [pil_to_tensor(img) for img in dataset["train"]["image"]]
    )
    preds = predictor.predict(images=images_tensor, batch_size=16)
    df = pd.DataFrame(preds.numpy(), columns=SUPPORTED_AUS)
    df.to_csv(path / au / "au_predictions.csv", index=False)
