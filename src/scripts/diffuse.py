from rsp.editing import AnycostDirections
from rsp.loading import load_model

sd = load_model("pixel", device="cuda", h_space="after", num_inference_steps=10)

# h_space Is the sementic latent space defined as ["before", middle","after"] the
# middle convolution in the U-net

ad = AnycostDirections(
    sd,
    etas=None,  # Noise schedule. None/0 for DDIM, 1 for DDPM. List if eta_t values is also supported
    num_examples=100,  # Choose num_examples top(bottom), positive(negative) examples for a given attribute from the sampled images
    idx_size=10000,  # Size of index. The number of images sampled in total
)

import matplotlib.pyplot as plt

q_original = sd.sample(
    seed=42
)  # All information about a sample is contained in the Q object
img_original = sd.show(q_original)  # Easy decoding and conversion to PIL.Image
plt.imshow(img_original)
plt.show()

# Edit Age
label = "AU01"
n = ad.get_direction(label)
q_edit = sd.apply_direction(q_original.copy(), n, scale=-0.6)
img_edit_age = sd.show(q_edit)

plt.imshow(img_edit_age)
plt.show()
print(q_original.hs.shape)
