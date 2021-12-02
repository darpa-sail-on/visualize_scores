import argparse
import copy
import itertools
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle as pkl
import seaborn as sns
from sklearn.manifold import TSNE
from tqdm import tqdm


sns.set_theme(style="whitegrid", font_scale=4.0)
#sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})


logger = logging.getLogger(__name__)


def cmd_options():
    parser = argparse.ArgumentParser("TSNE Visualization for features")
    parser.add_argument("--known-csv", required=True,
                        help="Path to csv file with known video names")
    parser.add_argument("--unknown-class-csv", required=True,
                        help="Path to csv file with unknown video names for class novelty")
    parser.add_argument("--unknown-spatial-csv", required=True,
                        help="Path to csv file with unknown video names for spatial novelty")
    parser.add_argument("--unknown-temporal-csv", required=True,
                        help="Path to csv file with unknown video names for temporal novelty")

    parser.add_argument("--class-features", required=True,
                        help="Path to pickle file with class novelty features")
    parser.add_argument("--spatial-features", required=True,
                        help="Path to pickle file with spatial novelty features")
    parser.add_argument("--temporal-features", required=True,
                        help="Path to pickle file with temporal novelty features")
    parser.add_argument("--plot-classwise", action="store_true", dest="plot_classwise",
                        help="Flag to plot class wise value in TSNE")
    parser.add_argument("--output-path",
                        help="Path to Visualization")
    parser.set_defaults(plot_classwise=False)
    args = parser.parse_args()
    return args


def read_csv(csv_path):
    return pd.read_csv(csv_path, delimiter=",", header=None,
                       names=["video_name", "class_idx"])


def read_features(feature_path):
    with open(feature_path, "rb") as f:
        features = pkl.load(f)
    return features


def filter_features(csv_df, features):
    filtered_feature_dict = {}
    for video_name in tqdm(csv_df["video_name"]):
        feat = np.mean(features["features_dict"][video_name].detach().cpu().numpy(),
                       axis=0)
        filtered_feature_dict[video_name] = feat
    return filtered_feature_dict


def create_tsne_subplot(known_embedded, unknown_embedded, axis, title, plot_classwise):
    plot_data = pd.concat([known_embedded, unknown_embedded], axis=0)
    sp = sns.scatterplot(data=plot_data, x=plot_data[0],
                         y=plot_data[1],
                         hue="class", style="class", ax=axis,
                         legend=not plot_classwise, s=70)
    sp.legend(bbox_to_anchor=(0.5, -0.05), loc="upper center", ncol=4)
    sp.set(xlabel=None, ylabel=None)
    axis.set_title(title)
    axis.grid()


def create_tsne_plot(known_feature_dict, unknown_class_dict,
                     unknown_spatial_dict, unknown_temporal_dict,
                     plot_classwise, output_path):
    tsne = TSNE()
    known_hue = list(map(lambda x: os.path.splitext(x)[0].split("_")[1], known_feature_dict.keys()))
    known_embedded = pd.DataFrame(tsne.fit_transform(np.array(list(known_feature_dict.values()))))
    unknown_class_embedded = pd.DataFrame(tsne.fit_transform(np.array(list(unknown_class_dict.values()))))
    unknown_spatial_embedded = pd.DataFrame(tsne.fit_transform(np.array(list(unknown_spatial_dict.values()))))
    unknown_temporal_embedded = pd.DataFrame(tsne.fit_transform(np.array(list(unknown_temporal_dict.values()))))
    # Add hue
    known_class_embedded = known_embedded.copy()
    known_class_embedded["class"] = known_hue
    known_embedded["class"] = "known"
    unknown_class_embedded["class"] = "unknown"
    unknown_spatial_embedded["class"] = "unknown"
    unknown_temporal_embedded["class"] = "unknown"

    #fig, axes = plt.subplots(1, 3, figsize=(10.0, 5.625))
    fig, axes = plt.subplots(1, 3, figsize=(6*5.625, 2*5.625))
    if plot_classwise:
        create_tsne_subplot(known_class_embedded, unknown_class_embedded,
                            axis=axes[0], title="Class Novelty",
                            plot_classwise=plot_classwise)
    else:
        create_tsne_subplot(known_embedded, unknown_class_embedded,
                            axis=axes[0], title="Class Novelty",
                            plot_classwise=False)
    create_tsne_subplot(known_embedded, unknown_spatial_embedded,
                        axis=axes[1], title="Spatial Novelty",
                        plot_classwise=False)
    create_tsne_subplot(known_embedded, unknown_temporal_embedded,
                        axis=axes[2], title="Temporal Novelty",
                        plot_classwise=False)
    fig.suptitle("TSNE Visualization for Class, Spatial and Temporal novelty")
    plt.tight_layout()
    plt.savefig(output_path)


def main(known_csv_path, unknown_class_csv_path, unknown_spatial_csv_path,
         unknown_temporal_csv_path, class_features_path, spatial_features_path,
         temporal_features_path, plot_classwise, output_path):
    known_csv = read_csv(known_csv_path)
    unknown_class_csv = read_csv(unknown_class_csv_path)
    unknown_spatial_csv = read_csv(unknown_spatial_csv_path)
    unknown_temporal_csv = read_csv(unknown_temporal_csv_path)
    class_features = read_features(class_features_path)
    spatial_features = read_features(spatial_features_path)
    temporal_features = read_features(temporal_features_path)
    known_feature_dict = filter_features(known_csv, class_features)
    unknown_class_dict = filter_features(unknown_class_csv, class_features)
    unknown_spatial_dict = filter_features(unknown_spatial_csv,
                                           spatial_features)
    unknown_temporal_dict = filter_features(unknown_temporal_csv,
                                            temporal_features)
    create_tsne_plot(known_feature_dict, unknown_class_dict, unknown_spatial_dict,
                     unknown_temporal_dict, plot_classwise, output_path)


if __name__ == "__main__":
    args = cmd_options()
    main(args.known_csv, args.unknown_class_csv, args.unknown_spatial_csv,
         args.unknown_temporal_csv, args.class_features, args.spatial_features,
         args.temporal_features, args.plot_classwise, args.output_path)
