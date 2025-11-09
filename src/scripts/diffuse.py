import logging

import matplotlib.pyplot as plt

from rsp.editing import AnycostDirections
from rsp.loading import load_model
from rsp.log_utils import set_logging_level

set_logging_level(logging.INFO)

sd = load_model("ldm", device="cuda", h_space="after", num_inference_steps=100)

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
img_original.save(f"original_image.png")

# Edit AU01
label = "AU01"
n = ad.get_direction(label)
q_edit = sd.apply_direction(q_original.copy(), n, scale=0.6)
img_edit_age = sd.show(q_edit)
plt.imshow(img_edit_age)
plt.savefig("edit_AU01.png")
plt.show()
