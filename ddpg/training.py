#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xb6aff5f7

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
from tqdm import tqdm

from ddpg.models import Actor
from ddpg.models import Critic
from ddpg.memory import ReplayMemory
from ddpg.util import run_with_sess
from ddpg.util import ornstein_uhlenbeck_noise


@run_with_sess
def train_with(sess, env, actor, critic, noise, num_episodes, batch_size, memory_size=100000, gamma=0.99, debug=False):
    """Train on an existing environment, actor, critic, and noise."""
    [obs_dim] = env.observation_space.shape
    [act_dim] = env.action_space.shape

    actor.update_target(sess)
    critic.update_target(sess)

    memory = ReplayMemory(memory_size)

    for episode_num in tqdm(range(num_episodes)):
        episode_rewards = []

        obs = env.reset()

        done = False
        while not done:

            [action] = actor.predict(sess, np.asarray([obs])) + next(noise)
            next_obs, reward, done, info = env.step(action)

            memory.add(obs, action, reward, done)

            if len(memory) >= batch_size:
                if debug and len(memory) == batch_size:
                    print("\nFinished accumulating memory, starting training.")

                obs_batch, act_batch, r_batch, done_batch, next_obs_batch = memory.sample(batch_size)

                best_next_act_batch = actor.target_predict(sess, next_obs_batch)
                best_next_Q_batch = critic.target_predict(sess, next_obs_batch, best_next_act_batch)

                target_Q_values = np.asarray([r if done else r + gamma * Q for r, done, Q in zip(r_batch, done_batch, best_next_Q_batch)])
                target_Q_batch = np.reshape(target_Q_values, (batch_size, 1))

                predicted_Q_values = critic.train(sess, obs_batch, act_batch, target_Q_batch)

                actor.train(sess, obs_batch)

                critic.update_target(sess)
                actor.update_target(sess)

            obs = next_obs
            episode_rewards.append(reward)

        if debug:
            sum_disc_r = 0
            for i, r in enumerate(episode_rewards):
                sum_disc_r += gamma**i * r
            print("\nR_{_coconut_format_0} = {_coconut_format_1} (over {_coconut_format_2} steps)".format(_coconut_format_0=(episode_num), _coconut_format_1=(sum_disc_r), _coconut_format_2=(len(episode_rewards))))

    return actor, critic, memory


def train(env_id, num_episodes, batch_size, *args, **kwargs):
    """Train a DDPG model on the given environment."""
    env = gym.make(env_id)

    [obs_dim] = env.observation_space.shape
    [act_dim] = env.action_space.shape

    critic = Critic(obs_dim, act_dim)
    actor = Actor(obs_dim, act_dim, critic)
    noise = ornstein_uhlenbeck_noise(np.zeros(act_dim))

    return train_with(env, actor, critic, noise, num_episodes, batch_size, *args, **kwargs)


if __name__ == "__main__":
    train("Pendulum-v0", num_episodes=100, batch_size=16, debug=True)
