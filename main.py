from utils.default import variant, get_cmd_args, get_log_dir
import torch

import utils.pytorch_util as ptu
from replay_buffer import ReplayBuffer
from utils.env_utils import NormalizedBoxEnv, domain_to_epoch, env_producer

from launcher_util import set_up
from path_collector import MdpPathCollector, RemoteMdpPathCollector

from model.agent import get_policy_producer,get_q_producer

from trainer.trainer import SACTrainer

from rl_algorithm import BatchRLAlgorithm
from utils.pytorch_util import set_gpu_mode

import ray
import logging
# ray.init(
#     # If true, then output from all of the worker processes on all nodes will be directed to the driver.
#     # log_to_driver=True,
#     logging_level=logging.WARNING,
#
#     # The amount of memory (in bytes)
#     # object_store_memory=1073741824, # 1g
#     # redis_max_memory=1073741824 # 1g
# )


if __name__ == "__main__":

    args = get_cmd_args()

    variant['log_dir'] = get_log_dir(args)

    variant['seed'] = args.seed
    variant['domain'] = args.domain

    variant['algorithm_kwargs']['num_epochs'] = domain_to_epoch(args.domain)
    variant['algorithm_kwargs']['num_trains_per_train_loop'] = args.num_trains_per_train_loop
    variant['algorithm_kwargs']['num_expl_steps_per_train_loop'] = args.num_expl_steps_per_train_loop
    variant['algorithm_kwargs']['algo'] = args.algo

    variant['trainer_kwargs']['use_automatic_entropy_tuning'] = args.no_aet

    set_gpu_mode(args.use_gpu and torch.cuda.is_available(), args.seed)

    set_up(variant,
            seed=args.seed,
            snapshot_gap=100,# Save the params every snapshot_gap and override previously saved result
            snapshot_mode='last_every_gap',
            )

    domain = variant['domain']
    seed = variant['seed']
    expl_env = env_producer(domain, seed)
    obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size

    # Get producer function for policy and value functions

    M = variant['layer_size']

    q_producer = get_q_producer(obs_dim, action_dim, hidden_sizes=[M, M])
    # print(f'{bcolors.FAIL}pass2{bcolors.ENDC}')
    policy_producer = get_policy_producer(
        obs_dim, action_dim, hidden_sizes=[M, M])
    # Finished getting producer

    expl_path_collector = MdpPathCollector(
        expl_env,
    )

    replay_buffer = ReplayBuffer(
        variant['replay_buffer_size'],
        ob_space=expl_env.observation_space,
        action_space=expl_env.action_space
    )
    trainer = SACTrainer(
        policy_producer,
        q_producer,
        action_space=expl_env.action_space,
        **variant['trainer_kwargs']
    )

    algorithm = BatchRLAlgorithm(
        trainer=trainer,
        exploration_data_collector=expl_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )

    algorithm.to(ptu.device)

    algorithm.train()


