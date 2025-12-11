import matplotlib.pyplot as plt
import seaborn as sns
import torch
from safetensors.torch import load_file

from rsp.constants import FIGURES_DIR, RESULTS_DIR, SUPPORTED_AUS

pre_path = f"{RESULTS_DIR}/anycost"
p = "/google-ddpm-ema-celebahq-256steps100-"
pth = "hspace-after-etasNone-idxsize10000.safetensors"

data = load_file(f"{pre_path}{p}{pth}", device="cpu")


attr = data["attr"]
attr = attr[attr[:, 0] != -1]

aus = SUPPORTED_AUS


# function that given a column of data returns the histogram of data
def get_au_histogram(au_column, name):
    plt.figure()
    sns.histplot(
        au_column.numpy(), bins=100, stat="percent", kde=False, element="step", alpha=1
    )
    plt.xlabel("Intensity", fontsize=20)
    plt.ylabel("Percent", fontsize=20)
    plt.grid(False)
    plt.xlim(-0.01, 1.01)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "au_distributions_generated" / f"{name}_histogram.png",
        bbox_inches="tight",
    )
    plt.close()
    return


for i, au in enumerate(aus):
    au_column = attr[:, i]
    get_au_histogram(au_column, au)
