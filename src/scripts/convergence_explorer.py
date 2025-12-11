import logging
import os

import matplotlib.pyplot as plt
import torch
from safetensors.torch import load_file
from torchvision.transforms import PILToTensor
from tqdm import tqdm

from rsp.constants import *
from rsp.editing import AnycostDirections
from rsp.loading import load_model
from rsp.log_utils import set_logging_level
from rsp.stateclass import Q

path = RESULTS_DIR / "anycost"

aus = ["AU04", "AU12"]
sd = load_model("pixel", device="cuda", h_space="after", num_inference_steps=100)

ad = AnycostDirections(
    sd,
    etas=None,
    num_examples=100,
    idx_size=10000,
    batch_size=32,
)

scale = 0.5

curr_seed = 100123012
sample_img = sd.sample(seed=curr_seed)

# save image
img = sd.show(sample_img)
figures_path = RESULTS_DIR / "figures" / "convergence_explorer"
# make the folder convergence_explorer if it does not exist
if not figures_path.exists():
    os.makedirs(figures_path)
img.save(figures_path / "unchanged_image.png")


for au in aus:
    file_path = path / f"convergence_{au}.safetensors"
    if not file_path.exists():
        print(f"File {file_path} does not exist. Skipping.")

        continue
    tensor = load_file(file_path, device="cuda")
    # if the figure_path / au folder does not exist, create it
    if not (figures_path / au).exists():
        os.makedirs(figures_path / au)

    for i in tqdm(range(0, tensor["steps_delta_hs"].shape[0], 2)):
        curr_direction = tensor["steps_delta_hs"][i]

        apply_direction = Q(delta_hs=curr_direction)
        apply_direction.hs = curr_direction
        img_edit_au = sd.apply_direction(
            sample_img.copy(), apply_direction, scale=scale
        )
        img_show = sd.show(img_edit_au)
        img_show.save(figures_path / au / f"image_{i}_{au}_edited.png")
