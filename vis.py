"""Quick visualization of results"""
import pandas as pd
import seaborn as sns
import glob
import json
import click
import sys
import os
import matplotlib.pyplot as plt
from pathlib import Path
import tqdm

def load_in_results(filepath):
    all_files = list(Path(filepath).rglob("*.json"))

    all_metrics = sorted(['m_num', 'm_num_stats', 'm_ndp', 
        'm_ndp_pre', 'm_ndp_post', 'm_acc_round','m_num_stats',
        'm_acc', 'm_acc_failed', 'm_is_cdt_and_is_early', 'm_nrp'])
    # TODO: this part really won't scale or generalilze well at all, rewrite so
    # it doesn't depend on algorithm/novelty/json_file.json structure
    results = {}#{m:{} for m in all_metrics}
    for f in tqdm.tqdm(all_files):
        splits = str(f).split('/')
        alg, novelty, _ = splits[-3:]
        with open(f) as json_file:
            js = json.load(json_file)
        
        metrics={}
        for metric_name, metric_dict in js.items():
            d = process_metric(metric_name, metric_dict)
            if d is not None:
                metrics.update(d)
        metrics.update(program_metrics(js))

        for metric, metric_val in metrics.items():
            if metric not in results:
                results[metric] = pd.DataFrame(columns=['x','y','alg','novelty'])
            for k,v in metric_val.items():
                results[metric].loc[len(results[metric])] = [
                    k,float(v), alg, novelty
                ]
            '''if alg not in results[metric]:
                results[metric][alg] = {}
            if novelty not in results[metric][alg]:
                results[metric][alg][novelty] = []
            results[metric][alg][novelty].append(metric_val)'''
    return results
    '''# check novelties
    novelties = None
    for metric, metric_dict in results.items():
        for v in metric_dict.values():
            kn = sorted(list(v.keys()))
            if novelties is None:
                novelties = kn
            else:
                assert kn == novelties
        
    df_results = {}
    for k,v in results.items():
        df_results[k] = pd.DataFrame(v)
    return df_results'''

def program_metrics(js):
    metric_agg = {}
    # calculate M1
    m_num_stats = js['m_num_stats']
    is_cdt = js['m_is_cdt_and_is_early']['Is CDT']
    is_early = js['m_is_cdt_and_is_early']['Is Early']
    if is_cdt and not is_early:
        gt_ind = int(m_num_stats['GT_indx'])
        p_ind = int(m_num_stats['P_indx_0.5'])
        '''m1_dict = {}
        for k,v in m_num_stats.items():
            if k == 'GT_indx':
                continue
            threshold = float(k.split('_')[-1])
            m1_dict[threshold] = v - gt_ind'''
        metric_agg['m1'] = {'Threshold 0.5' : p_ind - gt_ind}
    return metric_agg

def process_metric(metric_name, metric_dict):
    if metric_name == 'm_num':
        return {metric_name: metric_dict}
    elif metric_name == 'm_num_stats':
        return None # TODO: revisit this
    elif metric_name in ['m_ndp', 'm_ndp_pre', 'm_ndp_post', 'm_acc_failed']:
        metric_agg = {}
        for k,v in metric_dict.items():
            s = k.split('_')
            score = float(s[-1])
            t = '_'.join(s[:-1])
            key_name = metric_name + ':' + t
            if key_name not in metric_agg:
                metric_agg[key_name] = {score: v}
            else:
                metric_agg[key_name][score] = v
        return metric_agg
    elif metric_name in ['m_is_cdt_and_is_early']:
        return None # TODO: revisit this
    elif 'm_acc_round' in metric_name:
        return None
    elif metric_name == 'm_acc':
        metric_agg = {}
        asymptotic_dict = {k:v for k,v in metric_dict.items() if 'asymptotic'
                in k}
        other_dict = {k:v for k,v in metric_dict.items() if 'asymptotic'
                not in k}
        metric_agg[metric_name] = other_dict
        metric_agg[metric_name + ':asymptotic_top1'] = {}
        metric_agg[metric_name + ':asymptotic_top3'] = {}
        for k,v in asymptotic_dict.items():
            s = k.split('_')
            assert s[0] == 'asymptotic'
            assert len(s) == 3
            thresh = int(s[1])
            top_type = s[-1]
            metric_agg[metric_name + ':asymptotic_{}'.format(
                top_type)][thresh] = v
        return metric_agg
    elif metric_name == 'm_nrp':
        return {metric_name: metric_dict}
    else:
        raise ValueError("Unknown metric: {}".format(metric_name))

def check_keys(keys):
    # remove any m_acc_round keys since they can vary and are all captured
    keys = [x for x in keys if 'm_acc_round' not in x]
    expected_keys = sorted(['m_num', 'm_num_stats', 'm_ndp', 'm_ndp_pre', 'm_ndp_post', 
            'm_acc', 'm_acc_failed', 'm_is_cdt_and_is_early', 'm_nrp'])

    assert (sorted(keys) == expected_keys)

def data_to_df(all_results):
    '''
    Takes in dictionary of dictionaries, where key is the filename and 
    etalue is the actual scored json output.
    Returns a dictionary where the key is the metric, and the dataframe is the 
    scores aggregated across all the files
    '''
    all_dfs = {}
    for filename, result in all_results.items():
        check_keys(result.keys())
        for metric_name,metric_dict in result.items():
            separated_metrics = process_metric(metric_name, metric_dict)
            if separated_metrics is None:
                continue
            for k,v in separated_metrics.items():
                new_df = pd.DataFrame(v, index = [filename])
                if metric_name not in all_dfs:
                    all_dfs[k] = new_df
                else:
                    all_dfs[k] = pd.concat([all_dfs[metric_name],new_df])
    return {k:v.transpose() for k,v in all_dfs.items()}

def plot_data(metric_name, data, output_fpath):
    plt.figure(figsize=(20,10))
    if metric_name in ['m_nrp','m_acc', 'm1']:
        #sns_plot = sns.boxplot(x="index", y="value", 
        #    data=melted_data, label = label)
        sns.catplot(x='x',y='y',row='novelty', data=data, orient='v', kind='box', hue='alg')
    else:
        sns_plot = sns.lineplot(x='x',y='y',data=data, hue='alg',style='novelty')
        sns_plot.set_xlabel('Threshold')
        #sns_plot = sns.lineplot(x='index', y = 'value', 
        #    data =melted_data, label = label)
    
    '''melted_data = pd.melt(data.reset_index(), id_vars = ['index'])
    if metric_name == 'm_nrp':
        sns_plot = sns.barplot(x="index", y="value", data=melted_data)
        lower_lim = min(melted_data['value']) * 0.9
        upper_lim = max(melted_data['value']) * 1.1
        sns_plot.set(ylim=(lower_lim, upper_lim,))
    else:
        sns_plot = sns.lineplot(x='index', y = 'value', data =melted_data)
        sns_plot.set_xlabel('Threshold')'''
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize = 12)
    plt.title(metric_name, fontsize = 24)
    plt.tight_layout()
    plt.savefig(output_fpath)
    plt.close()

@click.command(help="Quick visualization script")
@click.option(
    "--result_path",
    "-r",
    "result_path",
    default=None,
    help="Filepath to a folder of json results (or a folder of folders" + 
    " of json results)",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option(
    "--output_path",
    "-o",
    "output_path",
    default=None,
    help="Filepath where visualizations should be stored",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
)
def main(result_path, output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    #result_filepaths = glob.glob(os.path.join(result_path,'*.json'))
    results_as_df = load_in_results(result_path)#result_filepaths)
    #results_as_df = data_to_df(results_as_dict)
    for k,v in results_as_df.items():
        plot_data(k,v, os.path.join(output_path, k + '.png')) 

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
