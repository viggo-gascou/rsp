import logging

import torch
from safetensors.torch import load_file
from tqdm import tqdm

from rsp.constants import AU_SUBSET, FIGURES_DIR, RESULTS_DIR, SUPPORTED_AUS
from rsp.editing import AnycostDirections, AnycostPredictor
from rsp.loading import load_model
from rsp.log_utils import log
from rsp.stateclass import Q
from rsp.utils import image_grid

FINAL_RESULTS_PATH = RESULTS_DIR / "au_editing"
NUM_IMAGES = 100
DEFAULT_SEED = 10000
LATENTS_FILE = FINAL_RESULTS_PATH / "original_latents.safetensors"

if not FINAL_RESULTS_PATH.exists():
    FINAL_RESULTS_PATH.mkdir(parents=True, exist_ok=True)

sd = load_model("pixel", device="cuda", h_space="after", num_inference_steps=100)

ad = AnycostDirections(
    sd,
    etas=None,
    num_examples=100,
    idx_size=10000,
    batch_size=32,
)

# Initialize predictor
predictor = AnycostPredictor(device="cuda", batch_size=32)

# Check if latents already exist
q_originals = []
if LATENTS_FILE.exists():
    log("Loading existing latents from safetensors...", logging.INFO)
    latents_dict = load_file(str(LATENTS_FILE))

    # Reconstruct q objects from state dicts
    for i in range(NUM_IMAGES):
        if i == 91 or i == 49:
            # Extract state dict for this image
            state_dict = {}
            prefix = f"latent_{i}_"
            for key in latents_dict.keys():
                if key.startswith(prefix):
                    attr_name = key[len(prefix) :]
                    state_dict[attr_name] = latents_dict[key].to("cuda")

            # Reconstruct q from state dict using Q class
            q = Q().from_state_dict(state_dict)
            q_originals.append(q)


def show(q: Q | list[Q]):
    """Show the output image the diffusion model."""
    if isinstance(q, Q):
        q = [q]
    imgs = [q_.x0 for q_ in q]
    imgs = torch.cat(imgs)

    return image_grid(imgs, size=225, rows=2, cols=3)


for au in tqdm(AU_SUBSET):
    q_edits = []
    q_edits_disentangled = []
    for i, q in enumerate(q_originals):
        for scale in [0.0, 0.5, 1.0]:
            dir = ad.get_direction(au)
            rest_aus = [au_ for au_ in SUPPORTED_AUS if au_ != au]
            cond_dir = ad.get_cond_dir(au, rest_aus)
            if scale == 0.0:
                q_edit = q
                q_edit_disentangled = q
            else:
                q_edit = sd.apply_direction(q, dir, scale)
                q_edit_disentangled = sd.apply_direction(q, cond_dir, scale)
            q_edits.append(q_edit)
            q_edits_disentangled.append(q_edit_disentangled)
    imgs = show(q_edits)
    imgs_disentangled = show(q_edits_disentangled)
    imgs.save(FIGURES_DIR / f"au_{au}_editing.pdf")
    imgs_disentangled.save(FIGURES_DIR / f"au_{au}_disentangled_editing.pdf")

# combs = list(itertools.combinations(AU_SUBSET, 2))
# for au1, au2 in tqdm(combs):
#     print(au1, au2)
#     if au1 == au2:
#         continue
#     for img_id, q in enumerate(q_originals):
#         q_edits = []
#         for scale in [0.0, 0.5, 1.0]:
#             dir1 = ad.get_cond_dir(au1, [au2])
#             dir2 = ad.get_cond_dir(au2, [au1])
#             final_dir = dir1 + dir2
#             if scale == 0.0:
#                 q_edit = q
#             else:
#                 q_edit = sd.apply_direction(q, final_dir, scale)
#             q_edits.append(q_edit)
#         imgs = sd.show(q_edits)
#         imgs.save(FINAL_RESULTS_PATH / f"au_{au1}_{au2}_cond_{img_id}_editing.png")
