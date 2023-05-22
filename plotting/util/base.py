# %%
import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import csv

from utils.env_utils import domain_to_epoch

rc_fonts = {
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.labelsize':8,
    'ytick.labelsize':8,
    "font.family": "times",
    "font.size": 10,
    'axes.titlesize':10,
    "legend.fontsize":10,
    "axes.spines.right": False,
    "axes.spines.top": False,
    # 'figure.figsize': (8, 3.5),
    # 'figure.figsize': (8.5, 11),
}
plt.rcParams.update(rc_fonts)
plt.rc('axes', unicode_minus=False)


def get_one_domain_one_run_res(path, key='exploration/Average Returns'):

    csv_path = osp.join(path, 'progress.csv'
    )
    if not os.path.exists(csv_path):
        text_path = osp.join(path, 'progress.txt'
                            )
        with open(text_path,'r') as input_file:
            first = True
            with open(csv_path,'w') as output_file:
                writer = csv.writer(output_file)
                for line in input_file:
                    if first:
                        line = line.replace('AverageTestEpRet', 'remote_evaluation/Average Returns')
                        line = line.replace('AverageEpRet', 'exploration/Average Returns')
                        first=False
                    row = line.strip().split('\t')
                    writer.writerow(row)

    values = []
    print('load file:',csv_path)
    with open(csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        # csv_file.read()

        col_names = next(reader)


        # Assume that the index of epoch is the last one
        # Not sure why the csv file is missing one col header
        # epoch_col_idx = col_names.index('Epoch')
        epoch_col_idx = col_names.index('Epoch')
        val_col_idx = col_names.index(key)

        for row in reader:

            epoch = int(row[epoch_col_idx])
            val = float(row[val_col_idx])
            #
            # if epoch == len(values):
            values.append(val)


    # Reshape the return value
    # to accomodate downstream api
    values = np.array(values)
    values = np.expand_dims(values, axis=-1)

    return values

def get_one_domain_one_run_res_by_multi_key(path, keys):

    csv_path = osp.join(path, 'progress.csv'
    )
    # values = []
    print('load file:',csv_path)
    data = np.genfromtxt(csv_path, delimiter=',',names=True,dtype=float)
    # col_name = [key.repalce('/','').replace(' ','_') for key in keys]

    res_dict = {}
    for key in keys:
        res_dict[key] = data[key.replace('/','').replace(' ','_')]
    return res_dict


def get_one_domain_all_run_res(algo_domian_path, key, classify=False):

        results = []
        seeds = os.listdir(algo_domian_path)
        for seed in seeds:
            algo_domian_seed_path = os.path.join(algo_domian_path,seed)
            res = get_one_domain_one_run_res(algo_domian_seed_path, key=key)
            results.append(res)

        min_rows = min([len(col) for col in results])
        results = [col[0:min_rows] for col in results]

        results = np.hstack(results)
        if classify:
            return results, seeds
        else:
            return results


def get_one_domain_all_run_res_by_multi_key(algo_domian_path, keys):
    results = {}
    for key in keys:
        results[key] = []

    seeds = os.listdir(algo_domian_path)
    for seed in seeds:
        algo_domian_seed_path = os.path.join(algo_domian_path, seed)
        res_dict = get_one_domain_one_run_res_by_multi_key(algo_domian_seed_path, keys=keys)
        for key in keys:
            results[key].append(res_dict[key].reshape(-1,1))

    for key in keys:

        min_row = min([len(value) for value in results[key]])
        values = [value[0:min_row] for value in results[key]]
        results[key] = np.hstack(values)

    return results


def smooth_results(results, smoothing_window=100):
    smoothed = np.zeros_like(results)

    for idx in range(len(smoothed)):

        if idx == 0:
            smoothed[idx] = results[idx]
            continue

        start_idx = max(0, idx - smoothing_window)

        smoothed[idx] = np.mean(results[start_idx:idx], axis=0)

    return smoothed


def get_tick_space(domain):
    #
    # if domain == 'Hopper':
    #     return 200
    #
    # if domain == 'humanoid':
    #     return 1000

    return 1000