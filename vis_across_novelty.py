import argparse
import copy
import itertools
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from tqdm import tqdm
#sns.set(font_scale=1.4)
sns.set_theme(style="whitegrid", font_scale=1.3)
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})


DICT_METRICS = ['m_num', 'm_num_stats', 'm_ndp',
               'm_ndp_pre', 'm_ndp_post', 'm_acc_round', 'm_num_stats',
               'm_acc', 'm_acc_failed', 'm_nrp']
SINGLE_VALUE_METRIC = [ 'm1', 'm2', 'm2.1', 'no_cdt']
logger = logging.getLogger(__name__)

def cmd_options():
    parser = argparse.ArgumentParser("Visualization across novelty levels")
    parser.add_argument("--result-root-path", required=True,
                        help="Path to root directory where results are stored")
    parser.add_argument("--algorithms", nargs="+",
                        help="Name of the algorithm that would be included in the plots")
    parser.add_argument("--novelty-levels", nargs="+",
                        help="Novelty levels used in plots")
    parser.add_argument("--timestamps", nargs="+",
                        help="Timestamps used in plots")
    parser.add_argument("--metrics", nargs="+",
                        help="Metrics used in plots")
    parser.add_argument("--row-subplot", default="metrics",
                        help="Row for subplots")
    parser.add_argument("--col-subplot", default="novelty_levels",
                        help="Column for subplots")
    parser.add_argument("--in-plot", default=("algorithms", "timestamps"),
                        nargs="+", help="Values combined in a plot")
    parser.add_argument("--output-path",
                        help="Filepath where visualizations should be stored")
    args = parser.parse_args()
    return args

def main(result_root_path, algorithms, novelty_levels, timestamps, metrics,
         row_subplot, col_subplot, in_plot, output_path):
    result_folders = itertools.product(novelty_levels, algorithms, timestamps)
    results = load_results(result_root_path, result_folders, row_subplot,
                           col_subplot, in_plot, metrics)
    # Plot results
    n_cols = len(_get_first_plot_key(col_subplot, metrics, novelty_levels, algorithms, timestamps))
    n_inplot = len(in_plot)
    #fig, axes = plt.subplots(2, n_cols, figsize=(2*10.0, 2*5.625))
    fig, axes = plt.subplots(2, n_cols, figsize=(1.5*10.0, 1.5*7.625))
    plot_m1_results(results['m1'], axes[0], n_cols, n_inplot, algorithms)
    results.pop("m1")
    plot_additional_results(results, axes[1], n_cols, n_inplot, algorithms)
    fig.suptitle("M1, M2, M2.1 and No Change Detected Across Class and Temporal Novelty")
    plt.tight_layout()
    plt.savefig(output_path)


def load_results(result_root_path, results_folders, row_subplot, col_subplot,
                 in_plot, metrics):
    plot_list = [row_subplot, col_subplot]
    plot_list.extend(in_plot)
    metric_idx = plot_list.index("metrics")
    plot_list.pop(metric_idx)
    results_dict = {}
    for novelty_level, algorithm, timestamp in tqdm(list(results_folders)):
        json_folder = os.path.join(result_root_path, novelty_level, algorithm,
                                   timestamp)
        json_files = os.listdir(json_folder)
        for json_file in tqdm(json_files, "json files"):
            json_result = json.load(open(os.path.join(json_folder, json_file), "r"))
            json_result = compute_program_metrics(json_result)
            for metric in metrics:
                if metric in results_dict:
                    current_result = results_dict[metric]
                else:
                    current_result = {}
                plot_dict = get_plot_dict({metric: json_result[metric]},
                                          copy.deepcopy(plot_list),
                                          metric,
                                          novelty_level,
                                          algorithm,
                                          timestamp,
                                          current_result)
                results_dict[metric] = plot_dict
    return results_dict


def compute_program_metrics(json_result):
    m_num_stats = json_result['m_num_stats']
    is_cdt = json_result['m_is_cdt_and_is_early']['Is CDT']
    is_early = json_result['m_is_cdt_and_is_early']['Is Early']
    gt_ind = int(m_num_stats['GT_indx'])
    p_ind = int(m_num_stats['P_indx_0.5'])
    if is_early:
        json_result['no_cdt'] = 0
        json_result['m2'] = 0
        json_result['m1'] = 1
        json_result['m2.1'] = 1
    elif is_cdt:
        json_result['no_cdt'] = 0
        m1 = p_ind - gt_ind
        json_result['m1'] = m1
        json_result['m2'] = 1
        json_result['m2.1'] = 0
    else:
        json_result['no_cdt'] = 1
        json_result['m1'] = p_ind - gt_ind
        json_result['m2'] = 0
        json_result['m2.1'] = 0
    return json_result


def _get_first_plot_key(plot_key, metric, novelty_level, algorithm, timestamp):
    if plot_key == "algorithms":
        result_key = algorithm
    elif plot_key == "novelty_levels":
        result_key = novelty_level
    elif plot_key == "timestamps":
        result_key = timestamp
    elif plot_key == "metrics":
        result_key = metric
    else:
        raise ValueError(f"Invalid value {plot_key} provided in plot_list")
    return result_key


def get_plot_dict(metric_dict, plot_list, metric, novelty_level, algorithm,
                  timestamp, result_dict):
    plot_key = plot_list.pop(0)
    result_key = _get_first_plot_key(plot_key, metric, novelty_level,
                                     algorithm, timestamp)
    if result_key in result_dict:
        if len(plot_list) < 1:
            result_dict[result_key] = aggregate_metric(metric_dict[metric],
                                                       result_dict[result_key],
                                                       metric)
        else:
            rec_results = get_plot_dict(metric_dict, plot_list, metric,
                                        novelty_level, algorithm, timestamp,
                                        result_dict[result_key])
            result_dict[result_key].update(rec_results)
    else:
        if len(plot_list) < 1:
            result_dict[result_key] = metric_dict[metric]
        else:
            rec_results = get_plot_dict(metric_dict, plot_list, metric,
                                        novelty_level, algorithm, timestamp,
                                        {})
            result_dict.update({result_key: rec_results})
    return result_dict


def aggregate_metric(metric_values, result_values, metric_name):
    if metric_name in SINGLE_VALUE_METRIC:
        if isinstance(result_values, list):
            result_values.extend([metric_values])
        else:
            result_values = [result_values]
            result_values.extend([metric_values])
    elif metric_name in DICT_METRICS:
        for metric_key, metric_value in metric_values.items():
            if isinstance(result_values[metric_key], list):
                result_values[metric_key].extend([metric_value])
            else:
                result_values[metric_key] = [result_values[metric_key]]
                result_values[metric_key].extend([metric_value])
    else:
        raise ValueError(f"Aggregating {metric_name} is not supported")
    return result_values


def create_inplot_df(results, n_inplots):
    assert n_inplots == 2, "Only a combination of algorithm with timestamp is supported"
    inplot_df = None
    for timestamp_name, timestamp_val in results.items():
        timestamp_df = None
        for algorithm_name, algorithm_val in timestamp_val.items():
            algorithm_df = pd.DataFrame(algorithm_val,
                                        columns=[algorithm_name])
            if timestamp_df is None:
                timestamp_df = algorithm_df
            else:
                timestamp_df = pd.concat([timestamp_df, algorithm_df],
                                         axis=1)
        timestamp_col = pd.DataFrame([timestamp_name]*len(timestamp_df),
                                     columns=["timestamp"])
        timestamp_df = pd.concat([timestamp_df, timestamp_col], axis=1)
        if inplot_df is None:
            inplot_df = timestamp_df
        else:
            inplot_df = pd.concat([inplot_df, timestamp_df], axis=0)
    return inplot_df


def plot_m1_results(results, axis, n_cols, n_inplots, algorithms):
    for col_index, col_name in enumerate(results.keys()):
        result_cols = results[col_name]
        is_outer_algo = any(list(map(lambda x: x in algorithms,
                                     result_cols.keys())))
        if is_outer_algo:
            results_df = result_cols
            results_df = pd.DataFrame(results_df).transpose()
            result_cols = results_df.to_dict()
        inplot_df = create_inplot_df(result_cols, n_inplots)
        subplot_axis = axis[col_index]
        inplot_melt = pd.melt(inplot_df,
                              id_vars="timestamp",
                              value_vars=algorithms,
                              var_name="Algorithms",
                              value_name="Number of videos")
        sns.boxplot(data=inplot_melt, x="timestamp", y="Number of videos",
                    hue="Algorithms", ax=subplot_axis)
        subplot_axis.set_title(f"M1 in {col_name}")
        subplot_axis.set(ylabel="Number of videos", xlabel="Red Button Push")
        subplot_axis.grid()


def plot_additional_results(results, axis, n_cols, n_inplots, algorithms, num_test=100):
    results_df = results
    results_df = pd.DataFrame(results_df).transpose()
    results = results_df.to_dict()
    for col_index, col_name in enumerate(results.keys()):
        result_cols = results[col_name]
        metric_df = None
        for row_name in result_cols.keys():
            result_rows = result_cols[row_name]
            is_outer_algo = any(list(map(lambda x: x in algorithms, result_rows.keys())))
            if is_outer_algo:
                result_df = result_rows
                result_df = pd.DataFrame(result_df).transpose()
                result_rows = result_df.to_dict()
            inplot_df = create_inplot_df(result_rows, n_inplots)
            inplot_df = inplot_df.reset_index(drop=True)
            percent_df = inplot_df.groupby("timestamp").sum()/float(100) * 100.0
            percent_df = percent_df.reset_index()
            percent_df["metric"] = row_name
            if metric_df is None:
                metric_df = percent_df
            else:
                metric_df = pd.concat([metric_df, percent_df])
        subplot_axis = axis[col_index]
        subplot_axis.set_title(f"M2, M2.1 and No Change Detected (no_cdt) in {col_name}")
        bottom_df = pd.DataFrame(np.unique(metric_df.timestamp), columns=["timestamp"])
        for algorithm in algorithms:
            bottom_df = pd.concat([bottom_df, pd.DataFrame(np.zeros(bottom_df.shape[0]),
                                                           columns=[algorithm])],
                                  axis=1)
        metrics = np.unique(metric_df.metric)
        algorithms_palette = {"gae_kl_nd": "Blues",
                             "fixed_x3d": "Oranges",
                             "adaptive_x3d": "Greens"}
        unique_timestamps = np.unique(metric_df.timestamp)
        bar_width = 0.2
        alpha = 0.9
        x_offsets = np.arange(-bar_width, bar_width+0.3, bar_width+0.05)
        for alg_idx, algorithm in enumerate(algorithms):
            algorithm_pallette = sns.color_palette(algorithms_palette[algorithm])
            current_df = metric_df[["timestamp", algorithm, "metric"]]
            x_val = np.arange(len(unique_timestamps)) + x_offsets[alg_idx]
            for ind, metric in enumerate(metrics):
                metric_pallette = tuple(list(algorithm_pallette[ind]) + [alpha])
                y_val = current_df[current_df.metric == metric][algorithm]
                bottom_val = bottom_df[algorithm]
                subplot_axis.bar(x_val, y_val, width=bar_width, \
                                 bottom=bottom_val, \
                                 label=f"{algorithm} ({metric})",
                                 color=metric_pallette,
                                 edgecolor="black")
                bottom_df[algorithm] += y_val
        subplot_axis.set(ylim=(0, 100), xlabel="Red Button Pushed",
                         ylabel="Percent",
                         xticks=np.arange(len(unique_timestamps)),
                         xticklabels=unique_timestamps)
        subplot_axis.grid()
        subplot_axis.legend()


if __name__ == "__main__":
    args = cmd_options()
    main(args.result_root_path, args.algorithms, args.novelty_levels,
         args.timestamps, args.metrics, args.row_subplot, args.col_subplot,
         args.in_plot, args.output_path)
