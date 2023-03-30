import torch

import utils.pytorch_util as ptu
from trainer.policies import TanhNormal
from torch.distributions import Uniform


def get_greedy_uniform_exploration_action(ob_np, policy=None, qfs=None):

    assert ob_np.ndim == 1

    ob = ptu.from_numpy(ob_np)

    # Ensure that ob is not batched
    assert len(list(ob.shape)) == 1

    _, pre_tanh_mu_T, _, _, std, _ = policy(ob)

    size = 32
    actions = torch. tanh(Uniform(pre_tanh_mu_T-2*std, pre_tanh_mu_T+2*std).sample(size))

    # actions = (torch.rand(size,policy.output_size)*2-1).to(ptu.device)
    # dist = TanhNormal(pre_tanh_mu_T, std)
    # actions = dist.sample_n([size])
    # Get the upper bound of the Q estimate

    args =[ob.reshape(1,-1).expand(size,len(ob)), actions]
    # args = list(torch.unsqueeze(i, dim=0) for i in (ob, actions[0]))
    Q1 = qfs[0](*args)
    Q2 = qfs[1](*args)

    Greedy_Q = torch.max(Q1,Q2).squeeze()
    wise_minus = Greedy_Q-Greedy_Q.max()
    log_sum = wise_minus.exp().sum().log()
    # input_tensor = Greedy_Q.exp() + 1e-8
    prob = (wise_minus - log_sum).exp()

    index_ac = prob.multinomial(1).item()

    max_q_ac = actions[index_ac]

    # mean_ac = torch.tanh(pre_tanh_mu_T)

    ac_np = ptu.get_numpy(max_q_ac)
    # print(max(max_q_ac), min(max_q_ac), max(mean_ac), min(mean_ac) )
    # print(max_q_ac- mean_ac)

    return ac_np, {}

