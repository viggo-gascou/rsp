import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from rsp.constants import AU_SUBSET, FIGURES_DIR, RESULTS_DIR, SUPPORTED_AUS

AU_EDITING_DIR = RESULTS_DIR / "au_editing"


no_edit_df = pd.read_csv(AU_EDITING_DIR / "original_au_predictions.csv")
# drop rows where any AU is -1
no_edit_df = no_edit_df[(no_edit_df[SUPPORTED_AUS] != -1.0).all(axis=1)]

sns.set_style("ticks")
au_folders = [
    path
    for path in AU_EDITING_DIR.iterdir()
    if path.is_dir() and not path.name.startswith(".")
]

au_folders = [
    path
    for path in AU_EDITING_DIR.iterdir()
    if path.is_dir() and not path.name.startswith(".")
]


def plot_au_distribution(df, au_name, title, fname, scale):
    # compare AU from filename with no_edit_df
    fig, ax = plt.subplots(figsize=(20, 12), nrows=1)

    common_kwargs = {
        "x": au_name,
        "ax": ax,
        "stat": "percent",
    }
    # calc actual bins
    bins_edit = np.histogram_bin_edges(df[au_name], bins=20, range=(0, 1))
    bins_no_edit = np.histogram_bin_edges(no_edit_df[au_name], bins=20, range=(0, 1))
    sns.histplot(
        data=no_edit_df, color="tab:blue", alpha=1, **common_kwargs, bins=bins_no_edit
    )
    sns.histplot(
        data=df, color="darkorange", alpha=0.7, **common_kwargs, bins=bins_edit
    )
    bins_edit = df[au_name].value_counts(bins=bins_edit, sort=False).sort_index()
    bins_no_edit = (
        no_edit_df[au_name].value_counts(bins=bins_no_edit, sort=False).sort_index()
    )

    ax.set_xlim(-0.01, 1.01)
    if au_name == "AU12":
        ax.set_ylim(0, 100)
    else:
        ax.set_ylim(0, 20)
    # ax.set_title(title)
    ax.set_xlabel("AU Score", fontsize=30)
    ax.set_ylabel("Percentage (%)", fontsize=30)
    ax.legend(["Not Edited", f"Edited (Scale={scale})"], fontsize=30)

    # set tickparams
    ax.tick_params(axis="both", which="major", labelsize=30)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / fname, bbox_inches="tight")
    plt.show()
    return bins_edit, bins_no_edit


def plot_mode_distributions(mode):
    for au in au_folders:
        for result in au.glob(f"{mode}/*.csv"):
            if au.name not in SUPPORTED_AUS:
                continue
            scale = result.stem.split("_")[-1].removeprefix("scale")
            if scale != "1.0":
                continue
            df = pd.read_csv(result)
            df = df[(df[SUPPORTED_AUS] != -1.0).all(axis=1)]
            title = (
                f"Distribution of AU Scores for {au.name} - Mode: {mode.capitalize()}"
            )
            fname = f"{au.name}.pdf"
            bins_edit, bins_no_edit = plot_au_distribution(
                df, au.name, title, fname, scale
            )
            print(
                f"Plotted {fname} with bins_edit: {bins_edit} and bins_no_edit: "
                f"{bins_no_edit}"
            )
            # mean and std
            print(
                f"Median AU {au.name} - Edited: {df[au.name].median():.4f}, No Edit: "
                f"{no_edit_df[au.name].median():.4f}"
            )
            print(
                f"Std AU {au.name} - Edited: {df[au.name].std():.4f}, No Edit: "
                f"{no_edit_df[au.name].std():.4f}"
            )
