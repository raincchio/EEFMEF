import os.path as osp
import argparse
import torch as th
import numpy as np

variant = dict(
    algorithm="SAC",
    version="normal",
    layer_size=256,
    replay_buffer_size=int(1E6),
    algorithm_kwargs=dict(
        num_eval_steps_per_epoch=5000,
        num_trains_per_train_loop=None,
        num_expl_steps_per_train_loop=None,
        min_num_steps_before_training=10000,
        max_path_length=1000,
        batch_size=256,
    ),
    trainer_kwargs=dict(
        discount=0.99,
        soft_target_tau=5e-3,
        target_update_period=1,
        policy_lr=3E-4,
        qf_lr=3E-4,
        reward_scale=1,
        use_automatic_entropy_tuning=True,
    ),
    optimistic_exp={}
)


def get_cmd_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='oac')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--domain', type=str, default='invertedpendulum')
    parser.add_argument('--use_gpu', default=False, action='store_true')
    parser.add_argument('--base_log_dir', type=str, default='./data')

    # optimistic_exp_hyper_param
    parser.add_argument('--beta_UB', type=float, default=0.0)
    parser.add_argument('--delta', type=float, default=0.0)

    # Training param
    parser.add_argument('--num_expl_steps_per_train_loop',
                        type=int, default=1000)
    parser.add_argument('--num_trains_per_train_loop', type=int, default=1000)

    args = parser.parse_args()

    return args

def get_log_dir(args, should_include_base_log_dir=True, should_include_seed=True, should_include_domain=True):

    log_dir = args.algo
    #     # Algo kwargs portion
    #     f'num_expl_steps_per_train_loop_{args.num_expl_steps_per_train_loop}_num_trains_per_train_loop_{args.num_trains_per_train_loop}'
    #
    #     # optimistic exploration dependent portion
    #     f'beta_UB_{args.beta_UB}_delta_{args.delta}',
    # )

    # if args.beta_UB >1:
    #     log_dir="oac"

    if should_include_domain:
        log_dir = osp.join(log_dir, args.domain)

    if should_include_seed:
        log_dir = osp.join(log_dir, f'seed_{args.seed}')

    if should_include_base_log_dir:
        log_dir = osp.join(args.base_log_dir, log_dir)

    return log_dir