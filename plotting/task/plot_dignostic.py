from plotting.util.base import *

# DOMAINS = ['humanoid','ant', 'halfcheetah', 'walker2d', 'hopper', 'swimmer']
domain = 'halfcheetah'

algos_of_domain = {}
algo_domian_paths = {}
task = "tmp"
paths = [
    "/home/chenxing/experiments/EE",
         # "/home/chenxing/experiments/test",
]
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
algos_set = sorted(list(algos_set))
COLORS = ["#ccb974", '#8172b2', '#c44e52', '#55a868', '#4c72b0', '#0000FF']
keys = {
    1:'remote_evaluation/Average Returns',
    2:'exploration/Average Returns',
    3:'trainer/Q1 Predictions Min',
    4:'trainer/Q2 Predictions Min',
    5:"trainer/Policy Loss",
    6:'trainer/QF1 Loss',
    7:'trainer/QF2 Loss',
    8:'trainer/Log Pis Min',
    9:'trainer/Alpha',
}
fig, axs = plt.subplots(3,3)
axs = axs.flatten()
env = f'{domain}-v2'
# key = keys[1]
for key, ax in zip(keys.values(), axs):
    for algo in algos_of_domain[domain]:
        algo_domian_path = algo_domian_paths[algo+domain]

        try:
            results = get_one_domain_all_run_res(algo_domian_path, key=key)
        except:
            continue
        results = smooth_results(results)

        mean = np.mean(results, axis=1)
        std = np.std(results, axis=1)

        x_vals = np.arange(len(mean))
        color = COLORS[algos_set.index(algo)]

        ax.plot(x_vals, mean, label=algo, color=color)
        ax.fill_between(x_vals, mean - std, mean + std, color=color, alpha=0.1)

        ax.set_title(env)
        ax.set_ylabel(key.split('/')[1])
        ax.set_xlabel('million steps')
        ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')

# xticks = np.arange(0, domain_to_epoch(
#     domain) + 1, get_tick_space(domain))

# ax.set_xticks(xticks, xticks / 1000.0)

axs[5].legend()

plt.tight_layout()
# plt.show()
fig.savefig('./plotting/pdf/'+task+'.pdf', bbox_inches='tight', dpi=300, backend='pdf')
print('./plotting/pdf/'+task+'.pdf ','plot finished!')