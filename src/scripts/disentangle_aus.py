import logging

import matplotlib.pyplot as plt

from rsp.constants import SUPPORTED_AUS
from rsp.editing import AnycostDirections
from rsp.loading import load_model
from rsp.log_utils import set_logging_level

set_logging_level(logging.INFO)

sd = load_model("pixel", device="cuda", h_space="after", num_inference_steps=100)

# h_space Is the sementic latent space defined as ["before", middle","after"] the
# middle convolution in the U-net


ad = AnycostDirections(
    sd,
    etas=None,  # etas Noise schedule. None/0 for DDIM, 1 for DDPM. or list of eta_t's
    num_examples=100,
    idx_size=10000,  # Size of index. The number of images sampled in total
    batch_size=32,
)

q_original = sd.sample(
    seed=100123012
)  # All information about a sample is contained in the Q object
img_original = sd.show(q_original)
img_original.save(f"pixel_original_image.png")

# for each AU cond it on all other AUs
for au in SUPPORTED_AUS:
    other_aus = [au2 for au2 in SUPPORTED_AUS if au2 != au]
    dir = ad.get_cond_dir(au, other_aus)
    q_edit = sd.apply_direction(q_original.copy(), dir, scale=1)
    img_edit_age = sd.show(q_edit)
    img_edit_age.save(f"pixel_{au}_edited_image.png")
