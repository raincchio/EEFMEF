from plotting.util.base import *
'''
Show the behavior difference between the explore strategy and current policy according to the reward diff for single algo
'''
# DOMAINS = ['humanoid','ant', 'halfcheetah', 'walker2d', 'hopper', 'swimmer']
domain = 'hopper'
test_algo = 'gac oac'

algos_of_domain = {}
algo_domian_paths = {}
task = "tmp"
paths = [
    "/home/chenxing/experiments/test",
         # "/home/chenxing/experiments/test",
]
rc_fonts = {
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.labelsize':12,
    'ytick.labelsize':12,
    "font.family": "times",
    "font.size": 12,
    'axes.titlesize':12,
    "legend.fontsize":12,
    "axes.spines.right": False,
    "axes.spines.top": False,
    # 'figure.figsize': (8, 3.5),
    # 'figure.figsize': (8.5, 11),
}
plt.rcParams.update(rc_fonts)
plt.rc('axes', unicode_minus=False)
for path in paths:
    algos = os.listdir(path)
    for algo in algos:
        if algo not in test_algo:
            continue
        domians = os.listdir(os.path.join(path,algo))
        for domian in domians:
            if domian not in algos_of_domain.keys():
                algos_of_domain[domian] = [algo]
            else:
                algos_of_domain[domian].append(algo)
            algo_domian_paths[algo + '_' + domian] = os.path.join(path, algo, domian)

algos_set = set()
for domian, algo in algos_of_domain.items():
    print(domian, ":",algo)
    algos_set=algos_set|set(algo)
for domian, algo in algo_domian_paths.items():
    print(algo, ":",domian)
algos_set = list(algos_set)

# pre defines
COLORS = ["#ccb974", '#8172b2', '#c44e52', '#55a868', '#4c72b0', '#0000FF']
keys = {1:'exploration/Average Returns',
        2:'remote_evaluation/Average Returns',
}

fig, axs = plt.subplots(1,2, figsize=(7.15,2.4))
env = f'{domain}-v2'

for algo, ax in zip(algos_of_domain[domain], axs):
    algo_domian_path = algo_domian_paths[algo + "_" + domain]

    results = get_one_domain_all_run_res_by_multi_key(algo_domian_path, keys=list(keys.values()))
    for idx, key in keys.items():
        average_value = smooth_results(results[key])
        mean = np.mean(average_value, axis=1)
        std = np.std(average_value, axis=1)
        x_vals = np.arange(len(mean))

        key_idct= {'exploration/Average Returns':"exploration",
                   'remote_evaluation/Average Returns':"evaluation"

        }

        ax.plot(x_vals, mean, label= key_idct[key], color=COLORS[idx])
        ax.fill_between(x_vals, mean - std, mean + std, color=COLORS[idx], alpha=0.1)


# Global Fig setup
        ax.set_title(algo+'-'+env)
        ax.set_ylabel("average return")

        xticks = np.arange(0, domain_to_epoch(
            domain) + 1, get_tick_space(domain))

        ax.set_xticks(xticks, xticks / 1000.0)

        ax.set_xlabel('million steps')
        ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
        ax.legend()


plt.tight_layout()
# plt.show()
fig.savefig('./'+task+'.pdf', bbox_inches='tight', dpi=300, backend='pdf')
print('./plotting/pdf/'+task+'.pdf ','plot finished!')