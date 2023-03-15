# %%
import pandas as pd
import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import copy
import csv

from main import get_cmd_args, get_log_dir
from utils.env_utils import domain_to_epoch


plt.rcParams['font.size'] = '12'


def get_one_domain_one_run_res(algo, domain, seed):



    csv_path = osp.join('./data',
        algo, domain, "seed_"+str(seed), 'progress.csv'
    )

    values = []

    with open(csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file)

        col_names = next(reader)

        # Assume that the index of epoch is the last one
        # Not sure why the csv file is missing one col header
        # epoch_col_idx = col_names.index('Epoch')
        epoch_col_idx = -1
        val_col_idx = col_names.index('exploration/Average Returns')

        for row in reader:

            # If this equals Epoch, it means the header
            # was written to the csv file again
            # and we reset everything
            if row[epoch_col_idx] == 'Epoch':
                values = []
                continue

            epoch = int(row[epoch_col_idx])
            val = float(row[val_col_idx])

            # We need to check if the row contains the values
            # of the correct epoch
            # because after reloading from checkpoint,
            # we are writing the result to the same csv file
            if epoch == len(values):
                values.append(val)
            else:
                # print(
                    # f'Reloaded row found at epoch {len(values), epoch} found for', domain, seed, hyper_params)
                pass

    # Reshape the return value
    # to accomodate downstream api
    values = np.array(values)
    values = np.expand_dims(values, axis=-1)

    return values


def get_one_domain_all_run_res(algo, domain, seeds):

    results = []

    for seed in seeds:
        try:
            res = get_one_domain_one_run_res(algo, domain, seed)
            results.append(res)
        except:
            continue

    min_rows = min([len(col) for col in results])
    results = [col[0:min_rows] for col in results]

    results = np.hstack(results)

    return results


def smooth_results(results, smoothing_window=100):
    smoothed = np.zeros((results.shape[0], results.shape[1]))

    for idx in range(len(smoothed)):

        if idx == 0:
            smoothed[idx] = results[idx]
            continue

        start_idx = max(0, idx - smoothing_window)

        smoothed[idx] = np.mean(results[start_idx:idx], axis=0)

    return smoothed


def plot(values, label, color=[0, 0, 1, 1]):
    mean = np.mean(values, axis=1)
    std = np.std(values, axis=1)

    x_vals = np.arange(len(mean))

    blur = copy.deepcopy(color)
    blur[-1] = 0.1

    plt.plot(x_vals, mean, label=label, color=color)
    plt.fill_between(x_vals, mean - std, mean + std, color=blur)

    plt.legend()


# DOMAINS = ['humanoid', 'halfcheetah', 'hopper', 'ant', 'walker2d']
DOMAINS = ['halfcheetah']

seeds = [1,2]



# def sac_get_one_domain_one_run_res(path, domain, seed):
#
#     csv_path = osp.join(
#         path, domain, f'seed_{seed}', 'progress.csv'
#     )
#
#     result = pd.read_csv(csv_path, usecols=[
#         'exploration/Average Returns'])
#
#     return result.values
#tood
# num_trains_per_train_loop == 4000:


def get_tick_space(domain):

    if domain == 'Hopper':
        return 200

    if domain == 'humanoid':
        return 1000

    return 500
COLORS = [ "#ccb974", '#8172b2', '#c44e52','#55a868','#4c72b0']
for domain in DOMAINS:
    plt.clf()
    env = f'{domain}-v2'

    for algo in ("oac", "sac"):

        results = get_one_domain_all_run_res(algo, domain, seeds)
        results = smooth_results(results)

        mean = np.mean(results, axis=1)
        std = np.std(results, axis=1)

        x_vals = np.arange(len(mean))
        color = COLORS.pop(0)

        plt.plot(x_vals, mean, label=algo, color=color)
        plt.fill_between(x_vals, mean - std, mean + std, color=color, alpha=0.1)


        # if domain == 'humanoid' and FORMAL_FIG:
        #     mean = np.mean(results, axis=1)
        #     x_vals = np.arange(len(mean))
        #
        #     # This is the index where OAC has
        #     # the same performance as SAC with 10 million steps
        #     # Plus 200 so that we are not overstating our claim
        #     magic_idx = np.argmax(mean > 8000) + 300
        #
        #     plt.plot(8000 * np.ones(magic_idx), linestyle='--',
        #              color=[0, 0, 1, 1], linewidth=3, label='Soft Actor Critic 10 million steps performance')
        #     plt.vlines(x=magic_idx,
        #                ymin=0, ymax=8000, linestyle='--',
        #                color=[0, 0, 1, 1],)

        """
        Plot result
        """


    plt.title('performance compare on '+ env)

    plt.ylabel('Average Episode Return')

    xticks = np.arange(0, domain_to_epoch(
        domain) + 1, get_tick_space(domain))

    plt.xticks(xticks, xticks / 1000.0)

    plt.xlabel('Number of environment steps in millions')
    plt.legend()

    plt.show()
    # plot_path = './data/plot'
    # os.makedirs(plot_path, exist_ok=True)



# %%
