import matplotlib.pyplot as plt
import torch
from safetensors.torch import load_file

from rsp.constants import *

pre_path = "../rsp/results/anycost"
p = "/google-ddpm-ema-celebahq-256steps100-"
pth = "hspace-after-etasNone-idxsize10000.safetensors"

# open the torch file its not weights its just tensors
data = load_file(f"{pre_path}{p}{pth}", device="cpu")

attr = data["attr"]

aus = SUPPORTED_AUS


# function that given a column of data returns the histogram of data
def get_au_histogram(au_column, name):
    plt.figure()
    plt.hist(au_column.numpy(), bins=50, density=True, alpha=0.75)
    plt.title(f"Histogram of {name}")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.grid(True)
    plt.savefig(f"{name}_histogram.png")
    plt.close()
    return


for i, au in enumerate(aus):
    au_column = attr[:, i]
    get_au_histogram(au_column, au)
