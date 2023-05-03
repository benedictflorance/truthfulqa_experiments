import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

output_paths = ["../outputs/ada_output_labeled_features.csv", "../outputs/babbage_output_labeled_features.csv", "../outputs/curie_output_labeled_features.csv", "../outputs/davinci_output_labeled_features.csv"]

def generate_two_by_two_plot(variable, save_dir):
    sns.color_palette("Set2")
    sns.set(font_scale=1.7)

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6.5,8))
    axes[0,0].set_title('Ada')
    axes[0,1].set_title('Babbage')
    axes[1,0].set_title('Curie')
    axes[1,1].set_title('Davinci')

    df = pd.read_csv(output_paths[0])
    sns.kdeplot(df, ax=axes[0,0], x=variable, hue="truthfulness", fill=True, legend=False)

    df = pd.read_csv(output_paths[1])
    sns.kdeplot(df, ax=axes[0,1], x=variable, hue="truthfulness", fill=True, legend=False)

    df = pd.read_csv(output_paths[2])
    sns.kdeplot(df, ax=axes[1,0], x=variable, hue="truthfulness", fill=True, legend=False)

    df = pd.read_csv(output_paths[3])
    sns.kdeplot(df, ax=axes[1,1], x=variable, hue="truthfulness", fill=True, legend=False)
    plt.tight_layout()
    fig.savefig(f'{save_dir}/{variable}.png')

import lftk
searched_features = lftk.search_features(return_format = "list_key")
for feature in tqdm(searched_features):
    generate_two_by_two_plot(feature, "../figures")