import logging

import matplotlib.pyplot as plt

from rsp.constants import AU_SUBSET
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
    seed=42
)  # All information about a sample is contained in the Q object
img_original = sd.show(q_original)
img_original.save(f"pixel_original_image.png")

# Edit AU01
label = "AU12"
n = ad.get_direction(label)
q_edit = sd.apply_direction(q_original.copy(), n, scale=5)
img_edit_age = sd.show(q_edit)
img_edit_age.save("pixel_AU12_edited_image.png")

clabels = ["AU04"]
n = ad.get_cond_dir(
    label, clabels
)  # Direction for smile with the direction for gender projected away
q_edit_smile_cond_au = sd.apply_direction(q_original.copy(), n, scale=0.3)
img_edit_cond_au = sd.show(q_edit_smile_cond_au)
img_edit_cond_au.save("pixel_AU12_cond_AU04_image.png")
