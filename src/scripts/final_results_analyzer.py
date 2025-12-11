import os

import matplotlib.pyplot as plt
import pandas

from rsp.constants import *

path = RESULTS_DIR / "final_results/"

# get all subfolders in path except "Original"
subfolders = [f.name for f in os.scandir(path) if f.is_dir() and f.name != "Original"]

# get the original path
original_path = path / "Original"

# get the bar plot data for the original images (its in the) au_predictions.csv file
original_csv_path = original_path / "au_predictions.csv"
original_df = pandas.read_csv(original_csv_path)

# save the plot of the csv (its just the au values for
# each au in AU_SUBSET) so a bar plot
# should be made for each au in AU_SUBSET
# (the avg value for each au across all images)
fig, ax = plt.subplots()
original_df[AU_SUBSET].mean().plot.bar(ax=ax)
ax.set_title("Original Images - Average AU Predictions")
ax.set_ylabel("Average AU Value")
ax.set_xlabel("Action Units (AUs)")
fig.savefig(path / "Original" / "original_au_predictions.png")
plt.close(fig)

for au in subfolders:
    # repeat but also add difference from original
    au_path = path / au
    au_csv_path = au_path / "au_predictions.csv"
    au_df = pandas.read_csv(au_csv_path)
    fig, ax = plt.subplots()
    au_df[AU_SUBSET].mean().plot.bar(ax=ax)
    ax.set_title(f"{au} Edited Images - Average AU Predictions")
    ax.set_ylabel("Average AU Value")
    ax.set_xlabel("Action Units (AUs)")
    fig.savefig(au_path / f"{au}_au_predictions.png")
    plt.close(fig)
    # now plot the difference from original
    diff = au_df[AU_SUBSET].mean() - original_df[AU_SUBSET].mean()
    fig, ax = plt.subplots()
    diff.plot.bar(ax=ax)
    ax.set_title(
        f"{au} Edited Images - Difference in Average AU Predictions from Original"
    )
    ax.set_ylabel("Difference in Average AU Value")
    ax.set_xlabel("Action Units (AUs)")
    fig.savefig(au_path / f"{au}_au_predictions_difference.png")
    plt.close(fig)
