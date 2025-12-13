import matplotlib.pyplot as plt
import pandas
import seaborn as sns

from rsp.constants import AU_SUBSET, FIGURES_DIR, RESULTS_DIR, SUPPORTED_AUS

AU_EDITING_DIR = RESULTS_DIR / "au_editing/"


# get the bar plot data for the original images (its in the) au_predictions.csv file
original_csv_path = AU_EDITING_DIR / "original_au_predictions.csv"
original_df = pandas.read_csv(original_csv_path)

# save the plot of the csv (its just the au values for
# each au in AU_SUBSET) so a bar plot
# should be made for each au in AU_SUBSET
# (the avg value for each au across all images)
fig, ax = plt.subplots()
sns.barplot(
    x=SUPPORTED_AUS,
    y=original_df[SUPPORTED_AUS].mean(),
    ax=ax,
    palette=["tab:blue" for _ in range(len(SUPPORTED_AUS))],
    hue=SUPPORTED_AUS,
    legend=False,
)
ax.set_title("Average AU Predictions")
ax.set_ylabel("Average AU Score")
ax.set_xlabel("Action Units (AUs)")
ax.set_xticks(range(len(SUPPORTED_AUS)))
ax.set_xticklabels(SUPPORTED_AUS, rotation=45)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "original_au_predictions.png", bbox_inches="tight", dpi=250)
plt.close(fig)

au_folders = [
    path
    for path in AU_EDITING_DIR.iterdir()
    if path.is_dir() and not path.name.startswith(".")
]

for au in au_folders:
    for result in au.glob("**/*.csv"):
        if result.parent.name != "simple":
            continue
        if au.name not in AU_SUBSET:
            continue

        au_df = pandas.read_csv(result)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x=SUPPORTED_AUS,
            y=au_df[SUPPORTED_AUS].mean(),
            ax=ax,
            palette=["tab:blue" for _ in range(len(SUPPORTED_AUS))],
            hue=SUPPORTED_AUS,
            legend=False,
        )
        ax.set_title(f"{au.name} Edited Images - Average AU Scores")
        ax.set_ylabel("Difference in Average AU Score")
        ax.set_xlabel("Action Units (AUs)")
        ax.set_xticks(range(len(SUPPORTED_AUS)))
        ax.set_xticklabels(SUPPORTED_AUS, rotation=45)
        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / f"{au.name}_au_predictions.png", bbox_inches="tight", dpi=250
        )
        plt.close()
        # now plot the difference from original
        diff = au_df[SUPPORTED_AUS].mean() - original_df[SUPPORTED_AUS].mean()
        fig, ax = plt.subplots(figsize=(10, 6))
        # red if negative green if positive
        sns.barplot(
            x=SUPPORTED_AUS,
            y=diff,
            ax=ax,
            palette=["tab:red" if x < 0 else "tab:blue" for x in diff],
            hue=SUPPORTED_AUS,
            legend=False,
        )
        # ax.patches[SUPPORTED_AUS.index(au.name)].set_linewidth(3)
        ax.patches[SUPPORTED_AUS.index(au.name)].set_color("darkorange")
        ax.set_title(f"{au.name} Edited Images - Difference in Average AU Scores")
        ax.set_ylabel("Difference in Average AU Score")
        ax.set_xlabel("Action Units (AUs)")
        ax.set_xticks(range(len(SUPPORTED_AUS)))
        # make tick for this au BOLD
        ax.set_xticklabels(SUPPORTED_AUS, rotation=45)
        ax.get_xticklabels()[SUPPORTED_AUS.index(au.name)].set_weight("bold")
        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / f"{au.name}_au_predictions_difference.png",
            bbox_inches="tight",
            dpi=250,
        )
        plt.close()
