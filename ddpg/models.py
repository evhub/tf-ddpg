#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x86c2bb5d

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

import tensorflow as tf

from ddpg.util import batch_input
from ddpg.util import get_params_defined_in
from ddpg.util import get_target_model_updater
from ddpg.util import run_sess_with_opt
from ddpg.networks import get_actor
from ddpg.networks import get_critic


def build_actor(obs_dim, act_dim):
    obs_input = batch_input(obs_dim)
    actor_params, actor = get_params_defined_in(_coconut.functools.partial(get_actor, obs_input, act_dim))
    return actor_params, [obs_input], actor


def build_critic(obs_dim, act_dim):
    obs_input = batch_input(obs_dim)
    act_input = batch_input(act_dim)
    critic_params, critic = get_params_defined_in(_coconut.functools.partial(get_critic, obs_input, act_input))
    return critic_params, (obs_input, act_input), critic


class Actor(_coconut.object):

    def __init__(self, obs_dim, act_dim, batch_size):
        self.params, (self.obs_input,), self.actor = build_actor(obs_dim, act_dim)
        self.target_params, (self.target_obs_input,), self.target_actor = build_actor(obs_dim, act_dim)
        self.target_updater = get_target_model_updater(self.target_params, self.params)
        self.act_grads_input = batch_input(act_dim)
        self.optimizer = ((tf.train.AdamOptimizer().apply_gradients)((_coconut_partial(zip, {1: self.params}, 2))(map(lambda x: x / batch_size, tf.gradients(self.actor, self.params, -self.act_grads_input)))))

    def train(self, sess, obs_batch, act_grads_batch):
        return run_sess_with_opt(sess, self.optimizer, [self.actor], feed_dict={self.obs_input: obs_batch, self.act_grads_input: act_grads_batch})

    def predict(self, sess, obs_batch):
        return sess.run(self.actor, feed_dict={self.obs_input: obs_batch})

    def target_predict(self, sess, obs_batch):
        return sess.run(self.target_actor, feed_dict={self.target_obs_input: obs_batch})

    def update_target(self, sess):
        sess.run(self.target_updater)


class Critic(_coconut.object):

    def __init__(self, obs_dim, act_dim):
        self.params, (self.obs_input, self.act_input), self.critic = build_critic(obs_dim, act_dim)
        self.target_params, (self.target_obs_input, self.target_act_input), self.target_critic = build_critic(obs_dim, act_dim)
        self.target_updater = get_target_model_updater(self.target_params, self.params)
        self.target_Q_input = batch_input(1)
        self.optimizer = tf.train.AdamOptimizer().minimize(tf.losses.mean_squared_error(self.target_Q_input, self.critic))
        self.act_grads = tf.gradients(self.critic, self.act_input)

    def train(self, sess, obs_batch, act_batch, target_Q_batch):
        return run_sess_with_opt(sess, self.optimizer, [self.critic], feed_dict={self.obs_input: obs_batch, self.act_input: act_batch, self.target_Q_input: target_Q_batch})

    def predict(self, sess, obs_batch, act_batch):
        return sess.run(self.critic, feed_dict={self.obs_input: obs_batch, self.act_input: act_batch})

    def target_predict(self, sess, obs_batch, act_batch):
        return sess.run(self.target_critic, feed_dict={self.target_obs_input: obs_batch, self.target_act_input: act_batch})

    def get_act_grads(self, sess, obs_batch, act_batch):
        return sess.run(self.act_grads, feed_dict={self.obs_input: obs_batch, self.act_input: act_batch})

    def update_target(self, sess):
        sess.run(self.target_updater)
