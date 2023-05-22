from plotting.util.base import *

DOMAINS = ['humanoid','ant', 'halfcheetah', 'walker2d', 'hopper', 'swimmer']
# DOMAINS = ['halfcheetah']

algos_of_domain = {}
algo_domian_paths = {}
task = "td3"
paths = [
    "/home/chenxing/experiments/tt",
    # "/home/chenxing/experiments/test",
]
for path in paths:
    algos = os.listdir(path)
    for algo in algos:
        algo_path = os.path.join(path,algo)
        if not os.path.isdir(algo_path):
            continue
        domians = os.listdir(algo_path)
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
keys = {1:'trainer/Q1 Predictions Min', 2:'exploration/Average Returns', 3:"trainer/Policy Loss",
        4:'trainer/Alpha',5:'remote_evaluation/Average Returns'}
fig, axs = plt.subplots(2,3)
axs = axs.flatten()

key = keys[5]

for domain, ax in zip(DOMAINS, axs):
    # plt.clf()
    env = f'{domain}-v2'

    for algo in algos_of_domain[domain]:
        algo_domian_path = algo_domian_paths[algo+domain]
        results = get_one_domain_all_run_res(algo_domian_path, key=key)
        # try:
        #     results = get_one_domain_all_run_res(algo_domian_path, key=key)
        # # key = 'trainer/QF1 Loss'
        # except:
        #     print('except', algo_domian_path)
        #     continue
        # key='trainer/Alpha'
        # key = "trainer/Policy Loss"
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
    ax.set_ylabel(key)

    xticks = np.arange(0, domain_to_epoch(
        domain) + 1, get_tick_space(domain))

    ax.set_xticks(xticks, xticks / 1000.0)

    ax.set_xlabel('million steps')
    ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
    if domain=='halfcheetah':
        ax.legend()

plt.tight_layout()
# plt.show()
fig.savefig('/home/chenxing/experiments/pdf/'+task+'.pdf', bbox_inches='tight', dpi=300, backend='pdf')
# print('./plotting/pdf/'+task+'.pdf ','plot finished!')