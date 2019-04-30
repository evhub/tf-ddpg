#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x2bd18f3c

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

# weight to use in soft updating target actor and critic models
SOFT_TARGET_UPDATE_WEIGHT = 0.001

# parameters for the Ornstein-Uhlenbeck action noise
ORNSTEIN_UHLENBECK_SIGMA = 0.3
ORNSTEIN_UHLENBECK_THETA = 0.15
ORNSTEIN_UHLENBECK_DT = 0.01

# parameters for the network's dense layers
DENSE_NEURONS = 128
DENSE_ACTIVATION = "relu"
DROPOUT_RATE = 0.1

# actions are assumed to be in the range [-ACTION_SCALE, ACTION_SCALE]
ACTION_SCALE = 2

# Adam learning rates for the actor and critic
ACTOR_LEARNING_RATE = 0.001
CRITIC_LEARNING_RATE = 0.001

# maximum size of the replay memory
MEMORY_SIZE = 100000

# reward discount rate
DISCOUNT_RATE = 0.99

# default training parameters
TRAINING_EPISODES = 100
BATCH_SIZE = 16
