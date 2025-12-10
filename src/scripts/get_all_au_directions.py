import logging

from tqdm import tqdm

from rsp.constants import RESULTS_DIR, SUPPORTED_AUS
from rsp.editing import AnycostDirections
from rsp.loading import load_model
from rsp.log_utils import log, set_logging_level

set_logging_level(logging.INFO)

sd = load_model("pixel", device="cuda", h_space="after", num_inference_steps=100)

ad = AnycostDirections(
    sd,
    etas=None,  # etas Noise schedule. None/0 for DDIM, 1 for DDPM. or list of eta_t's
    num_examples=100,
    idx_size=10000,  # Size of index. The number of images sampled in total
    batch_size=32,
)

for au in tqdm(
    SUPPORTED_AUS,
    desc="Extracting directions",
):
    ad.get_direction(au)

log(f"Successfully, extracted directions for all AUs ({SUPPORTED_AUS})", logging.INFO)
