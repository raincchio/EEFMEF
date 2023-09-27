import torch

import utils.pytorch_util as ptu
from torch.distributions import Uniform
import random


def get_random_exploration_action(ob_np, policy=None, qfs=None, sample_size=32, sample_range=1, beta=None):

    assert ob_np.ndim == 1

    ob = ptu.from_numpy(ob_np)

    # Ensure that ob is not batched
    assert len(list(ob.shape)) == 1

    _, pre_tanh_mu_T, _, _, std, _ = policy(ob)

    begin = pre_tanh_mu_T-sample_range*std
    end = pre_tanh_mu_T+sample_range*std
    actions = torch.tanh(Uniform(begin, end).sample([sample_size]))

    args =[ob.reshape(1,-1).expand(sample_size,len(ob)), actions]
    # args = list(torch.unsqueeze(i, dim=0) for i in (ob, actions[0]))

    Q1 = qfs[0](*args)
    Q2 = qfs[1](*args)
    Greedy_Q = Q1 if random.randint(0,10)>=5 else Q2

    Greedy_Q = Greedy_Q.squeeze()*beta
    wise_minus = Greedy_Q-Greedy_Q.max()
    log_sum = wise_minus.exp().sum().log()

    prob = (wise_minus - log_sum).exp()

    index_ac = prob.multinomial(1).item()

    max_q_ac = actions[index_ac]

    # mean_ac = torch.tanh(pre_tanh_mu_T)

    ac_np = ptu.get_numpy(max_q_ac)
    # print(max(max_q_ac), min(max_q_ac), max(mean_ac), min(mean_ac) )
    # print(max_q_ac- mean_ac)

    return ac_np, {}

