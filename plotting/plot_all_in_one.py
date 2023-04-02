# %%
import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import copy
import csv

from utils.env_utils import domain_to_epoch

rc_fonts = {
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.labelsize':8,
    'ytick.labelsize':8,
    "font.family": "times",
    "font.size": 8,
    'axes.titlesize':8,
    "legend.fontsize":8,
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

    values = []

    with open(csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        # csv_file.read()

        col_names = next(reader)

        # Assume that the index of epoch is the last one
        # Not sure why the csv file is missing one col header
        # epoch_col_idx = col_names.index('Epoch')
        epoch_col_idx = -1
        val_col_idx = col_names.index(key)

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


def get_one_domain_all_run_res(algo_domian_path, key='exploration/Average Returns'):

    results = []
    seeds = os.listdir(algo_domian_path)


    for seed in seeds:
        algo_domian_seed_path = os.path.join(algo_domian_path,seed)
        res = get_one_domain_one_run_res(algo_domian_seed_path, key=key)
        results.append(res)


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



DOMAINS = ['humanoid','ant', 'halfcheetah', 'walker2d', 'hopper', 'swimmer']
# DOMAINS = ['halfcheetah']

seeds = [1,2,3,4,5,6]



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
#note
# num_trains_per_train_loop == 4000:


def get_tick_space(domain):

    if domain == 'Hopper':
        return 200

    if domain == 'humanoid':
        return 1000

    return 500



algos_of_domain = {}
algo_domian_paths = {}
task = "sample_method"
paths = ["/home/chenxing/experiments/sample_method"]
for path in paths:
    algos = os.listdir(path)
    for algo in algos:
        domians = os.listdir(os.path.join(path,algo))
        for domian in domians:
            if domian not in algos_of_domain.keys():
                algos_of_domain[domian] = [algo]
            else:
                algos_of_domain[domian].append(algo)
            algo_domian_paths[algo + domian] = os.path.join(path,algo, domian)

algos_set = set()
for domian, algo in algos_of_domain.items():
    print(domian, ":",algo)
    algos_set=algos_set|set(algo)
for domian, algo in algo_domian_paths.items():
    print(algo, ":",domian)
algos_set = list(algos_set)
COLORS = ["#ccb974", '#8172b2', '#c44e52', '#55a868', '#4c72b0', '#0000FF']
fig, axs = plt.subplots(2,3)
axs = axs.flatten()
for domain, ax in zip(DOMAINS, axs):
    # plt.clf()
    env = f'{domain}-v2'

    for algo in algos_of_domain[domain]:
        algo_domian_path = algo_domian_paths[algo+domain]
        results = get_one_domain_all_run_res(algo_domian_path,)#key='trainer/Alpha'
        results = smooth_results(results)

        mean = np.mean(results, axis=1)
        std = np.std(results, axis=1)

        x_vals = np.arange(len(mean))
        color = COLORS[algos_set.index(algo)]

        ax.plot(x_vals, mean, label=algo, color=color)
        ax.fill_between(x_vals, mean - std, mean + std, color=color, alpha=0.1)

        """
        Plot result
        """

    ax.set_title(env)
    ax.set_ylabel('average reward')

    xticks = np.arange(0, domain_to_epoch(
        domain) + 1, get_tick_space(domain))

    ax.set_xticks(xticks, xticks / 1000.0)

    ax.set_xlabel('million steps')
    ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
    if domain=='halfcheetah':
        ax.legend()

plt.tight_layout()
# plt.show()
fig.savefig('./plotting/pdf/'+task+'.pdf', bbox_inches='tight', dpi=300, backend='pdf')
print('./plotting/pdf/'+task+'.pdf ','file saved!')
    # plot_path = './data/plot'
    # os.makedirs(plot_path, exist_ok=True)



# %%
