#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x396d1c39

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

from ddpg.constants import CRITIC_LEARNING_RATE
from ddpg.constants import ACTOR_LEARNING_RATE
from ddpg.util import batch_input
from ddpg.util import get_params_defined_in
from ddpg.util import get_target_model_updater
from ddpg.util import run_sess_with_opt
from ddpg.networks import get_actor
from ddpg.networks import get_critic


def get_actor_with_params(obs_input, act_dim):
    """Return (actor_params, actor_model)."""
    return get_params_defined_in(_coconut.functools.partial(get_actor, obs_input, act_dim))


def get_critic_with_params(obs_input, act_input):
    """Return (critic_params, critic_model)."""
    return get_params_defined_in(_coconut.functools.partial(get_critic, obs_input, act_input))


class Critic(_coconut.object):
    """A DDPG critic. Keeps track of both the current critic and the target critic."""

    def __init__(self, obs_dim, act_dim, lr=CRITIC_LEARNING_RATE):
# build critic model
        self.obs_input = batch_input(obs_dim)
        self.act_input = batch_input(act_dim)
        self.params, self.critic = get_critic_with_params(self.obs_input, self.act_input)
        self.target_params, self.target_critic = get_critic_with_params(self.obs_input, self.act_input)
        self.target_updater = get_target_model_updater(self.target_params, self.params)

# construct MSE loss
        self.target_Q_input = batch_input(1)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(tf.losses.mean_squared_error(self.target_Q_input, self.critic), var_list=self.params)

    def train(self, sess, obs_batch, act_batch, target_Q_batch):
        """Train the critic on the given observation, action, and target Q value batches."""
        return run_sess_with_opt(sess, self.optimizer, [self.critic], feed_dict={self.obs_input: obs_batch, self.act_input: act_batch, self.target_Q_input: target_Q_batch})

    def predict(self, sess, obs_batch, act_batch):
        """Get the predicted Q value for the given observation and action batches."""
        return sess.run(self.critic, feed_dict={self.obs_input: obs_batch, self.act_input: act_batch})

    def target_predict(self, sess, obs_batch, act_batch):
        """Same as predict but uses the target critic."""
        return sess.run(self.target_critic, feed_dict={self.obs_input: obs_batch, self.act_input: act_batch})

    def update_target(self, sess):
        """Update the target critic."""
        sess.run(self.target_updater)


class Actor(_coconut.object):
    """A DDPG actor. Keeps track of both the current actor and the target actor."""

    def __init__(self, obs_dim, act_dim, critic, lr=ACTOR_LEARNING_RATE):
# build actor model
        self.obs_input = batch_input(obs_dim)
        self.params, self.actor = get_actor_with_params(self.obs_input, act_dim)
        self.target_params, self.target_actor = get_actor_with_params(self.obs_input, act_dim)
        self.target_updater = get_target_model_updater(self.target_params, self.params)

# use critic to construct loss
        self.critic_params, self.self_critic = get_critic_with_params(self.obs_input, self.actor)
        self.critic_updater = get_target_model_updater(self.critic_params, critic.params, update_weight=1)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(-self.self_critic, var_list=self.params)

    def train(self, sess, obs_batch):
        """Train the actor on the given batch of observations."""
        return run_sess_with_opt(sess, self.optimizer, [self.actor], feed_dict={self.obs_input: obs_batch})

    def predict(self, sess, obs_batch):
        """Get the predicted best actions for the given observations."""
        return sess.run(self.actor, feed_dict={self.obs_input: obs_batch})

    def target_predict(self, sess, obs_batch):
        """Same as predict but uses the target actor."""
        return sess.run(self.target_actor, feed_dict={self.obs_input: obs_batch})

    def update_target(self, sess):
        """Update the target actor and self critic."""
        sess.run(self.target_updater)
        sess.run(self.critic_updater)
