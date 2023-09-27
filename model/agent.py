from model.networks import FlattenMlp
from model.policies import TanhGaussianPolicy, MakeDeterministic

from algo.sac import SACTrainer
from algo.bacc import BACCTrainer
from algo.td3 import TD3Trainer
from algo.sac_no_min import SACNoMin
from algo.tripleQ import TripleQ
from algo.sac_loss import SAClossTrainer


def get_policy_producer(obs_dim, action_dim, hidden_sizes):
    def policy_producer(deterministic=False):
        policy = TanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
        )

        if deterministic:
            policy = MakeDeterministic(policy)

        return policy

    return policy_producer


def get_q_producer(obs_dim, action_dim, hidden_sizes):
    def q_producer():
        return FlattenMlp(input_size=obs_dim + action_dim,
                          output_size=1,
                          hidden_sizes=hidden_sizes, )

    return q_producer


def get_trainer(algo):

    return {'sac': SACTrainer,
            'td3': TD3Trainer,
            'oac': SACTrainer,
            'gac_random': SACTrainer,
            'sac_no_min': SACNoMin,
            'tripleq': TripleQ,
            'bacc': BACCTrainer,
            'sac_loss': SAClossTrainer
            }[algo]
