#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x5bd7414a

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


class ReplayMemory(_coconut.object):

    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = deque()

    def add(self, obs, act, r, done, next_obs):
        self.memory.append((obs, act, r, done, next_obs))
        while len(self.memory) > self.memory_size:
            self.memory.popleft()

    def sample(self, batch_size):
        sampled_inds = random.sample(range(len(self.memory)), min(len(self.memory), batch_size))

# only iterate through the deque once
        batch = []
        for i, (obs, act, r, done, next_obs) in enumerate(self.memory):
            if i in sampled_inds:
                batch.append((obs, act, r, done, next_obs))

        return zip(*batch)

    def __len__(self):
        return len(self.memory)