#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x1b4674ea

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

import random
from collections import deque

import numpy as np


class ReplayMemory(_coconut.object):
    """A buffer for storing (obs, action, reward, done) tuples."""

    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = deque()

    def add(self, obs, action, reward, done):
        """Add a new observation to the replay memory."""
        self.memory.append((obs, action, reward, done))
        while len(self.memory) > self.memory_size:
            self.memory.popleft()

    def __len__(self):
# ignore the last elem if not done as it might not have a next_obs
        return len(self.memory) if self.memory[-1][-1] else max(0, len(self.memory) - 1)

    def sample(self, batch_size):
        """Sample obs_batch, action_batch, reward_batch, done_batch, next_obs_batch from the replay memory."""
        sampled_inds = random.sample(range(len(self)), batch_size)

# only iterate through the deque once, going in reverse order
#  so we can keep track of next_obs
        batch = []
        next_obs = None
        for i, (obs, action, reward, done) in reversed(enumerate(self.memory)):
            if done:
                next_obs = np.zeros_like(obs) * np.nan
            if i in sampled_inds:
                batch.append((obs, action, reward, done, next_obs))
            next_obs = obs

        return map(np.asarray, zip(*batch))
