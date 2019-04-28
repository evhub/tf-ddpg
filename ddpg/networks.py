#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x35e91655

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

from tensorflow.keras import layers

from ddpg.util import dense_with_batch_norm


def proc_obs(obs_input):
    """Generate a model that processes an observation."""
    return ((dense_with_batch_norm(512, "relu"))((dense_with_batch_norm(512, "relu"))(obs_input)))


def proc_obs_and_act(obs_input, act_input):
    """Generate a model that processes an observation and an action."""
    return ((dense_with_batch_norm(512, "relu"))((dense_with_batch_norm(512, "relu"))(layers.Concatenate()([act_input, proc_obs(obs_input)]))))


def get_actor(obs_input, act_dim):
    """Generate a DDPG actor model."""
    return ((layers.Dense(act_dim, activation="tanh"))((proc_obs)(obs_input)))


def get_critic(obs_input, act_input):
    """Generate a DDPG critic model."""
    return ((layers.Dense(1, activation=None))(proc_obs_and_act(obs_input, act_input)))
