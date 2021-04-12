"""Aggregates results from the metrics_eval_best_weights.json in a parent folder"""

import argparse
import json
import os
working_path = os.path.abspath(os.path.dirname(__file__))

from tabulate import tabulate
import pandas as pd

import utils
import model.net as net

parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments',
                    help='Directory containing results of experiments')


def aggregate_metrics(parent_dir, metrics):
    """Aggregate the metrics of all experiments in folder `parent_dir`.

    Assumes that `parent_dir` contains multiple experiments, with their results stored in
    `parent_dir/subdir/metrics_dev.json`

    Args:
        parent_dir: (string) path to directory containing experiments results
        metrics: (dict) subdir -> {'accuracy': ..., ...}
    """
    
    # Get the metrics for the folder if it has results from an experiment
    metrics_file = os.path.join(parent_dir, 'metrics_test_best_weights.json')
    if os.path.isfile(metrics_file):
        # Get the number of params
        model_name = metrics_file.split('\\')[-2]
        params = utils.Params(os.path.join(parent_dir, 'params.json'))
        model = getattr(net, model_name)(params).cuda()
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        # for trainable params only
        # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # logfile - findout the training time
        df = pd.read_csv(os.path.join(parent_dir, '_train.log'), header=None)
        df[0] = pd.to_datetime(df[0])
        traintime = (df.iloc[-1, 0] - df.iloc[0, 0]).total_seconds()

        headname = parent_dir.split('\\')[-2] + ' - ' + parent_dir.split('\\')[-1]
        
        with open(metrics_file, 'r') as f:
            best_json = json.load(f)
            del(best_json['loss'])
            best_json['time(s)'] = traintime
            best_json['params'] = pytorch_total_params
            metrics[headname] = best_json

    # Check every subdirectory of parent_dir
    for subdir in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, subdir)):
            continue
        else:
            aggregate_metrics(os.path.join(parent_dir, subdir), metrics)


def metrics_to_table(metrics):
    # Get the headers from the first subdir. Assumes everything has the same metrics
    headers = metrics[list(metrics.keys())[0]].keys()
    table = [[subdir] + [values[h] for h in headers] for subdir, values in metrics.items()]
    res = tabulate(table, headers, tablefmt='pipe')

    return res

if __name__ == "__main__":
    args = parser.parse_args()
    # args.parent_dir = os.path.join(working_path, args.parent_dir)
    
    # Aggregate metrics from args.parent_dir directory
    metrics = dict()
    aggregate_metrics(args.parent_dir, metrics)
    table = metrics_to_table(metrics)

    # Display the table to terminal
    print(table)

    # Save results in parent_dir/results.md
    save_file = os.path.join(args.parent_dir, "results.md")
    with open(save_file, 'w') as f:
        f.write(table)

    # python synthesize_results.py --parent_dir experiments_4_4_4