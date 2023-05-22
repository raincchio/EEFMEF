from plotting.util.base import *

DOMAINS = ['humanoid','ant', 'halfcheetah', 'walker2d', 'hopper', 'swimmer']
# DOMAINS = ['humanoid','ant', 'halfcheetah', 'hopper',]

algos_of_domain = {}
algo_domian_paths = {}
task = "overall"
paths = [
    "/home/chenxing/experiments/overall",
    # "/home/chenxing/experiments/gac_exp/ablation_range",
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

algos_set = sorted(list(algos_set))
idd = algos_set.index('gac')
algos_set.pop(idd)
algos_set.insert(0,'gac')
print(algos_set)
COLORS = ['#c44e52',"#ccb974", '#8172b2',  '#55a868', '#4c72b0', '#0000FF']
keys = {1:'trainer/Q1 Predictions Min', 2:'exploration/Average Returns', 3:"trainer/Policy Loss",
        4:'trainer/Alpha',5:'remote_evaluation/Average Returns'}
# fig, axs = plt.subplots(1,4,figsize=(7.16,2.4))
fig, axs = plt.subplots(2,3,figsize=(7.16,4))
axs = axs.flatten()

key = keys[5]
idx=0
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

    ax.set_title('('+chr(97+idx)+') '+ domain.capitalize() +'-v2')
    idx +=1

    ax.set_ylabel('average return')

    xticks = np.arange(0, domain_to_epoch(
        domain) + 1, get_tick_space(domain))

    ax.set_xticks(xticks, xticks / 1000.0)
    if idx>3:
        ax.set_xlabel('million steps')
    ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
    if domain=='halfcheetah':
        handles, labels = ax.get_legend_handles_labels()
        wanted_labels = ['gac', 'oac', 'sac', 'td3']
        wanted_handles = [handles[labels.index(item)] for item in wanted_labels]
        ax.legend(wanted_handles,wanted_labels, edgecolor='None', facecolor='None')

plt.tight_layout()
# plt.show()
fig.savefig('./pdf/'+task+'.pdf', bbox_inches='tight', dpi=300, backend='pdf')
# print('./plotting/pdf/'+task+'.pdf ','plot finished!')