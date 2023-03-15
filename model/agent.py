from model.networks import FlattenMlp
from trainer.policies import TanhGaussianPolicy, MakeDeterministic


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