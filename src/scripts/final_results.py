import logging
from pathlib import Path

import pandas as pd
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from rsp.constants import RESULTS_DIR, SUPPORTED_AUS
from rsp.editing import AnycostDirections, AnycostPredictor
from rsp.loading import load_model
from rsp.log_utils import log, logger, set_logging_level
from rsp.stateclass import Q

set_logging_level(logging.INFO)

FINAL_RESULTS_PATH = RESULTS_DIR / "final_results"
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
if LATENTS_FILE.exists():
    log("Loading existing latents from safetensors...", logging.INFO)
    latents_dict = load_file(str(LATENTS_FILE))
    q_originals = []

    # Reconstruct q objects from state dicts
    for i in range(NUM_IMAGES):
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
else:
    log("Generating original latent representations...", logging.INFO)
    q_originals = []
    latents_to_save = {}

    curr_seed = DEFAULT_SEED
    for i in tqdm(range(NUM_IMAGES), desc="Generating latents", unit="image"):
        curr_seed += i
        q_original = sd.sample(seed=curr_seed)
        q_originals.append(q_original)

        # Save state dict for this q
        state_dict = q_original.to_state_dict()
        for key, value in state_dict.items():
            latents_to_save[f"latent_{i}_{key}"] = value.cpu()

    log("Saving latents to safetensors...", logging.INFO)
    save_file(latents_to_save, str(LATENTS_FILE))

original_x0_stack = torch.cat([q.x0.float() for q in q_originals], dim=0)
if not Path(FINAL_RESULTS_PATH, f"original_au_predictions.csv").exists():
    log("\nPredicting AUs for original images...", logging.INFO)
    preds_original = predictor(original_x0_stack, batch_size=32)
    df_original = pd.DataFrame(preds_original.cpu().numpy(), columns=SUPPORTED_AUS)
    df_original.to_csv(
        FINAL_RESULTS_PATH / f"original_au_predictions.csv",
        index=False,
    )
    del df_original
    del preds_original


def predict_aus(qs, predictor, prefix, au, scale):
    x0_stack = torch.cat([q for q in qs], dim=0)
    preds = predictor(x0_stack, batch_size=32)
    df = pd.DataFrame(preds.cpu().numpy(), columns=SUPPORTED_AUS)
    out_path = FINAL_RESULTS_PATH / au / prefix / f"au_predictions_scale{scale}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        out_path,
        index=False,
    )


def apply_edits_and_predict(qs, mode, scale):
    for au in tqdm(SUPPORTED_AUS, desc="Processing", position=0):
        edits = []
        for q_original in tqdm(qs, desc="Editing", position=1):
            if mode == "simple":
                n = ad.get_direction(au)
                q_edit = sd.apply_direction(q_original, n, scale=scale)
            elif mode == "disentangled":
                other_aus = [au2 for au2 in SUPPORTED_AUS if au2 != au]
                dir_disentangled = ad.get_cond_dir(au, other_aus)
                q_edit = sd.apply_direction(q_original, dir_disentangled, scale=scale)
            edits.append(q_edit.x0)
        predict_aus(edits, predictor, mode, au, scale)


for scale in [1.0, -1.0, 0.25, 0.50, 0.75]:
    log(f"Applying edits and predicting AUs for scale {scale}", logging.INFO)
    apply_edits_and_predict(q_originals, "simple", scale)
    if scale == 1.0:
        apply_edits_and_predict(q_originals, "disentangled", scale)
