import abc
import time
from collections import OrderedDict

import numpy as np
import torch
torch.set_printoptions(profile='full')
from utils.logging import logger
import utils.eval_util as eval_util

import gtimer as gt
from component.replay_buffer import ReplayBuffer
from component.path_collector import MdpPathCollector, RemoteMdpPathCollector
from exploration.greedy import get_greedy_exploration_action
from utils.pytorch_util import state_dict_cpu, save_model
import ray


class Framework(metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_data_collector: MdpPathCollector,
            remote_eval_data_collector:RemoteMdpPathCollector,
            replay_buffer: ReplayBuffer,
            seed,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            sample_size=32,
            sample_range=1,
            beta=1,
            log_model=False,
            algo='sac',

    ):
        super().__init__()

        """
        The class state which should not mutate
        """
        self.seed=seed
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.algo=algo
        self.sample_size = sample_size
        self.sample_range = sample_range
        self.beta = beta
        self.log_model = log_model

        """
        The class mutable state
        """
        self._start_epoch = 0

        """
        This class sets up the main training loop, so it needs reference to other
        high level objects in the algorithm

        But these high level object maintains their own states
        and has their own responsibilities in saving and restoring their state for checkpointing
        """
        self.trainer = trainer

        self.expl_data_collector = exploration_data_collector
        self.remote_eval_data_collector = remote_eval_data_collector
        self.replay_buffer = replay_buffer

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        if self.log_model:
            save_model(self.trainer, seed=self.seed, epoch=0)
        self._train()


    def _train(self):

        # Fill the replay buffer to a minimum before training starts

        if self.min_num_steps_before_training > self.replay_buffer.num_steps_can_sample():
            # init_expl_paths = \
            expl_paths = self.expl_data_collector.collect_new_paths(
                self.trainer.policy,
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )

            self.replay_buffer.add_paths(expl_paths)
            self.expl_data_collector.end_epoch(-1)
        # exit()
        env = self.expl_data_collector._env
        obs = env.reset()
        for step in range(int(3e6)):

            action, agent_info = get_greedy_exploration_action(obs, policy=self.trainer.policy,
                                                               qfs=[self.trainer.qf1, self.trainer.qf2],
                                                               sample_size=self.sample_size,
                                                               sample_range=self.sample_range,
                                                               beta=(step/1000.0+1)*self.beta)

            next_obs, reward, terminal, env_info = env.step(action)

            if terminal:
                env.reset()
            obs = next_obs

            self.replay_buffer._add_path(obs[None], action[None],reward[None],next_obs[None],np.array([int(terminal)])[None],agent_info,env_info)


            train_data = self.replay_buffer.random_batch(
                self.batch_size)
            self.trainer.train(train_data)


            # Wait for eval to finish
            if step % 1000 == 0:
                pol_state_dict = state_dict_cpu(self.trainer.policy)
                remote_eval_obj_id = self.remote_eval_data_collector.async_collect_new_paths.remote(
                    self.max_path_length,
                    self.num_eval_steps_per_epoch,
                    discard_incomplete_paths=True,
                    deterministic_pol=True,
                    pol_state_dict=pol_state_dict)
                ray.get([remote_eval_obj_id])
                self._end_epoch(step)

    def _end_epoch(self, step):

        self._log_stats(step)

        self.expl_data_collector.end_epoch(step)
        ray.get([self.remote_eval_data_collector.end_epoch.remote(step)])

        self.replay_buffer.end_epoch(step)
        self.trainer.end_epoch(step)

        logger.record_tabular('Epoch', step)

        write_header = True if step == 0 else False
        logger.dump_tabular(with_prefix=False, with_timestamp=False,
                            write_header=write_header)


    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        """
        Replay Buffer
        """
        logger.record_dict(
            self.replay_buffer.get_diagnostics(),
            prefix='replay_buffer/'
        )

        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')

        """
        Exploration
        """
        # logger.record_dict(
        #     self.expl_data_collector.get_diagnostics(),
        #     prefix='exploration/'
        # )
        # expl_paths = self.expl_data_collector.get_epoch_paths()
        # logger.record_dict(
        #     eval_util.get_generic_path_information(expl_paths),
        #     prefix="exploration/",
        # )
        """
        Remote Evaluation
        """
        logger.record_dict(
            ray.get(self.remote_eval_data_collector.get_diagnostics.remote()),
            prefix='remote_evaluation/',
        )
        remote_eval_paths = ray.get(
            self.remote_eval_data_collector.get_epoch_paths.remote())
        logger.record_dict(
            eval_util.get_generic_path_information(remote_eval_paths),
            prefix="remote_evaluation/",
        )

        """
        Misc
        """

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)


def _get_epoch_timings():
    times_itrs = gt.get_times().stamps.itrs
    times = OrderedDict()
    epoch_time = 0
    for key in sorted(times_itrs):
        time = times_itrs[key][-1]
        epoch_time += time
        times['time/{} (s)'.format(key)] = time
    times['time/epoch (s)'] = epoch_time
    times['time/total (s)'] = gt.get_times().total
    return times
