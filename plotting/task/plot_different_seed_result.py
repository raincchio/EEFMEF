from plotting.util.base import *
'''
Show the performence of different random seed
'''
# DOMAINS = ['humanoid','ant', 'halfcheetah', 'walker2d', 'hopper', 'swimmer']
DOMAINS = ['swimmer']
# domain = 'hopper'

algos_of_domain = {}
algo_domian_paths = {}
task = "tmp"
paths = [
    # "/home/chenxing/experiments/gac_U_Q_V",
         "/home/chenxing/experiments/ablation_bt_1_sr_7_ss_32",
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
keys = {1:'trainer/QF1 Loss', 2:'exploration/Average Returns', 3:"trainer/Policy Loss",
        4:'trainer/Alpha',5:'remote_evaluation/Average Returns'}

fig, axs = plt.subplots(2,3)
axs = axs.flatten()
for domain, ax in zip(DOMAINS, axs):
    env = f'{domain}-v2'
    key = keys[5]

    for algo in algos_of_domain[domain]:
        algo_domian_path = algo_domian_paths[algo+'_'+domain]

        results,seeds = get_one_domain_all_run_res(algo_domian_path, key=key, classify=True)
        results = smooth_results(results)

        for idx in range(len(seeds)):
            res = results[:,idx]
            x_vals = np.arange(len(res))
            color = COLORS[algos_set.index(algo)]

            ax.plot(x_vals, res, label=seeds[idx], color=COLORS[idx])
            # ax.fill_between(x_vals, mean - std, mean + std, color=color, alpha=0.1)

    ax.set_title(env)
    ax.set_ylabel(key)

    xticks = np.arange(0, domain_to_epoch(
        domain) + 1, get_tick_space(domain))

    ax.set_xticks(xticks, xticks / 1000.0)

    ax.set_xlabel('million steps')
    ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
    ax.legend()


plt.tight_layout()
plt.show()
# fig.savefig('./plotting/pdf/'+task+'.pdf', bbox_inches='tight', dpi=300, backend='pdf')
print('./plotting/pdf/'+task+'.pdf ','plot finished!')