#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x1f313196

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

import functools

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def dense_with_batch_norm(num_neurons, activation):
    """Behaves as layers.Dense but includes BatchNormalization."""
    return lambda prev_layer: ((layers.Activation(activation))((layers.BatchNormalization())((layers.Dense(num_neurons, activation=None))(prev_layer))))


def batch_input(input_dim):
    """Create a tensor for a batch of 1D inputs of the given length."""
    return tf.placeholder(tf.float32, [None, input_dim], name="batch_input_{_coconut_format_0}".format(_coconut_format_0=(input_dim)))


def get_params_defined_in(param_def_func):
    """Gets all the trainable variables defined in the given function."""
    num_old_params = len(tf.trainable_variables())
    result = param_def_func()
    new_params = tf.trainable_variables()[num_old_params:]
    return new_params, result


def get_target_model_updater(target_params, base_params, update_weight=0.001):
    """Gets an optimizer that can be used to soft update target_params with base_params."""
    return [target_param.assign((1 - update_weight) * target_param + update_weight * base_param) for target_param, base_param in zip(target_params, base_params)]


def ornstein_uhlenbeck_noise(mu, sigma=0.3, theta=0.15, dt=0.01):
    """Produces an iterator that outputs Ornstein-Uhlenbeck random noise."""
    x = np.zeros_like(mu)
    while True:
        x += theta * (mu - x) * dt + sigma * np.sqrt(dt) * np.random.randn(*mu.shape)
        yield x


def run_sess_with_opt(sess, opt, outputs=[], feed_dict={}):
    """Run a session including the given optimizer but excluding its output."""
    _coconut_match_to = sess.run([opt] + outputs, feed_dict=feed_dict)
    _coconut_match_check = False
    if (_coconut.isinstance(_coconut_match_to, _coconut.abc.Sequence)) and (_coconut.len(_coconut_match_to) >= 1):
        results = _coconut.list(_coconut_match_to[1:])
        _coconut_match_check = True
    if not _coconut_match_check:
        _coconut_match_val_repr = _coconut.repr(_coconut_match_to)
        _coconut_match_err = _coconut_MatchError("pattern-matching failed for " "'[_] + results = sess.run([opt] + outputs, feed_dict=feed_dict)'" " in " + (_coconut_match_val_repr if _coconut.len(_coconut_match_val_repr) <= 500 else _coconut_match_val_repr[:500] + "..."))
        _coconut_match_err.pattern = '[_] + results = sess.run([opt] + outputs, feed_dict=feed_dict)'
        _coconut_match_err.value = _coconut_match_to
        raise _coconut_match_err

    return results


def run_with_sess(base_func):
    """Wrap base_func by providing a sess and automatically initializing global variables."""
    @functools.wraps(base_func)
    def sess_func(*args, **kwargs):
        sess = tf.Session()
        try:
            sess.run(tf.global_variables_initializer())
            return base_func(sess, *args, **kwargs)
        finally:
            sess.close()
    return sess_func
