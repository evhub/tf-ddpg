#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x70b5dfe1

# Compiled with Coconut version 1.4.0-post_dev30 [Ernest Scribbler]

# Coconut Header: -------------------------------------------------------------

from __future__ import print_function, absolute_import, unicode_literals, division
import sys as _coconut_sys, os.path as _coconut_os_path
_coconut_file_path = _coconut_os_path.dirname(_coconut_os_path.abspath(__file__))
_coconut_cached_module = _coconut_sys.modules.get(str("__coconut__"))
if _coconut_cached_module is not None and _coconut_os_path.dirname(_coconut_cached_module.__file__) != _coconut_file_path:
    del _coconut_sys.modules[str("__coconut__")]
_coconut_sys.path.insert(0, _coconut_file_path)
from __coconut__ import _coconut, _coconut_MatchError, _coconut_igetitem, _coconut_base_compose, _coconut_forward_compose, _coconut_back_compose, _coconut_forward_star_compose, _coconut_back_star_compose, _coconut_forward_dubstar_compose, _coconut_back_dubstar_compose, _coconut_pipe, _coconut_back_pipe, _coconut_star_pipe, _coconut_back_star_pipe, _coconut_dubstar_pipe, _coconut_back_dubstar_pipe, _coconut_bool_and, _coconut_bool_or, _coconut_none_coalesce, _coconut_minus, _coconut_map, _coconut_partial, _coconut_get_function_match_error, _coconut_addpattern, _coconut_sentinel, _coconut_assert
from __coconut__ import *
if _coconut_sys.version_info >= (3,):
    _coconut_sys.path.pop(0)

# Compiled Coconut: -----------------------------------------------------------

import gym
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ddpg.models import Actor
from ddpg.models import Critic
from ddpg.memory import ReplayMemory
from ddpg.util import run_with_sess
from ddpg.util import ornstein_uhlenbeck_noise


@run_with_sess
def train_with(sess, env, actor, critic, noise, num_episodes, batch_size, memory_size=100000, gamma=0.99):
    [obs_dim] = env.observation_space.shape
    [act_dim] = env.action_space.shape

    actor.update_target(sess)
    critic.update_target(sess)

    memory = ReplayMemory(memory_size)

    for i in tqdm(range(num_episodes)):

        obs = env.reset()
        done = False
        while not done:

            [action] = actor.predict(sess, np.asarray([obs])) + next(noise)
            next_obs, reward, done, info = env.step(action)

            memory.add(obs, action, reward, done, next_obs)

            if len(memory) >= batch_size:
                obs_batch, act_batch, r_batch, done_batch, next_obs_batch = memory.sample(batch_size)

                best_next_act_batch = actor.target_predict(sess, next_obs_batch)
                best_next_Q_batch = critic.target_predict(sess, next_obs_batch, best_next_act_batch)

                target_Q_values = np.asarray([r if done else r + gamma * Q for r, done, Q in zip(r_batch, done_batch, best_next_Q_batch)])
                target_Q_batch = np.reshape(target_Q_values, (batch_size, 1))

                predicted_Q_values = critic.train(sess, obs_batch, act_batch, target_Q_batch)

                best_acts = actor.predict(sess, obs_batch)
                [act_grads] = critic.get_act_grads(sess, obs_batch, best_acts)
                actor.train(sess, obs_batch, act_grads)

                actor.update_target(sess)
                critic.update_target(sess)

            obs = next_obs

    return actor, critic, memory


def train(env_id, num_episodes, batch_size, *args, **kwargs):
    env = gym.make(env_id)

    [obs_dim] = env.observation_space.shape
    [act_dim] = env.action_space.shape

    actor = Actor(obs_dim, act_dim, batch_size)
    critic = Critic(obs_dim, act_dim)
    noise = ornstein_uhlenbeck_noise(np.zeros(act_dim))

    return train_with(env, actor, critic, noise, num_episodes, batch_size, *args, **kwargs)


if __name__ == "__main__":
    train("Pendulum-v0", 100, 16)
