#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The file used to implement the data store and replay

By xiaobo
Contact linxiaobo110@gmail.com
Created on Wed Jan 17 10:40:44 2018
"""

# Copyright (C)
#
# This file is part of QuadrotorFly
#
# GWpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GWpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GWpy.  If not, see <http://www.gnu.org/licenses/>.


from collections import deque
import random
import numpy as np

"""
********************************************************************************************************
**-------------------------------------------------------------------------------------------------------
**  Compiler   : python 3.6
**  Module Name: MemoryStore
**  Module Date: 2018-04-17
**  Module Auth: xiaobo
**  Version    : V0.1
**  Description: create the module
**-------------------------------------------------------------------------------------------------------
**  Reversion  : V0.2
**  Modified By: xiaobo
**  Date       : 2019-4-25
**  Content    : rewrite the module, add note
**  Notes      :
********************************************************************************************************/
**-------------------------------------------------------------------------------------------------------
**  Reversion  : V0.3
**  Modified By: xiaobo
**  Date       : 2019-5-20
**  Content    : modify the data record, compatible wit v0.2, store data each episode independently 
**  Notes      :
********************************************************************************************************/
"""


class ReplayBuffer(object):
    """ storing data in order replaying for train algorithm"""

    def __init__(self, buffer_size, random_seed=123):
        # size of minimize buffer is able to train
        self.buffer_size = buffer_size
        # counter for replay buffer
        self.count = 0
        # buffer, contain all data together
        self.buffer = deque()
        # used for random sampling
        random.seed(random_seed)
        # when count rise over the buffer_size, the train can begin
        self.isBufferFull = False
        # counter for episode
        self.episodeNum = 0
        # record the start position of each episode in buffer
        self.episodePos = deque()
        # record the sum rewards of steps for each episode
        self.episodeRewards = deque()

    def buffer_append(self, experience):
        """append data to buffer, should run each step after system update"""
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
            self.isBufferFull = False
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
            self.isBufferFull = True

    def episode_append(self, rewards):
        self.episodeNum += 1
        self.episodePos.append(self.count)
        self.episodeRewards.append(rewards)

    def size(self):
        return self.count

    def buffer_sample_batch(self, batch_size):
        """sample a batch of data with size of batch_size"""
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        return batch

    def clear(self):
        self.buffer.clear()
        self.count = 0
        self.episodeNum = 0
        self.episodePos.clear()
        self.episodeRewards.clear()


class DataRecord(object):
    """data record for show result"""
    def __init__(self, compatibility_mode=False):
        # new buffer, store data sepeartely with different episode, new in 0.3
        self.episodeList = list()
        self.bufferTemp = deque()
        self.compatibilityMode = compatibility_mode

        # counter for replay buffer
        self.count = 0

        # counter for episode
        self.episodeNum = 0

        # record the sum rewards of steps for each episode
        self.episodeRewards = deque()
        # record the average td error of steps for each episode
        self.episodeTdErr = deque()
        # record some sample of weights, once after episode
        self.episodeWeights = deque()
        # record some sample of weights, once each step, for observing the vary of weights
        self.weights = deque()

        if self.compatibilityMode:
            # buffer, contain all data together, discarded in 0.3
            self.buffer = deque()
            # record the start position of each episode in buffer, discarded in 0.3
            self.episodePos = deque()

    def buffer_append(self, experience, weights=0):
        """append data to buffer, should run each step after system update"""
        self.bufferTemp.append(experience)
        self.count += 1
        self.weights.append(weights)

        if self.compatibilityMode:
            self.buffer.append(experience)

    #        if self.count < self.buffer_size:
    #            self.buffer.append(experience)
    #            self.count += 1
    #            self.isBufferFull = False
    #        else:
    #            self.buffer.popleft()
    #            self.buffer.append(experience)

    def episode_append(self, rewards=0, td_err=0, weights=0):
        """append data to episode buffer, should run each episode after episode finish"""
        self.episodeNum += 1
        self.episodeRewards.append(rewards)
        self.episodeTdErr.append(td_err)
        self.episodeWeights.append(weights)

        self.episodeList.append(self.bufferTemp)
        self.bufferTemp = deque()
        if self.compatibilityMode:
            self.episodePos.append(self.count)

    def get_episode_buffer(self, index=-1):
        if index == -1:
            index = self.episodeNum - 1
        elif index > (self.episodeNum - 1):
            self.print_mess("Does not exist this episode!")
        else:
            index = index
            return

        buffer_temp = self.episodeList[index]
        data = list()
        item_len = len(buffer_temp[0])
        for ii in range(item_len):
            x = np.array([_[ii] for _ in buffer_temp])
            data.append(x)
        return data

    def size(self):
        return self.count

    def clear(self):
        self.count = 0
        self.episodeNum = 0
        self.episodeRewards.clear()
        self.bufferTemp.clear()
        self.episodeList.clear()
        if self.compatibilityMode:
            self.buffer.clear()
            self.episodePos.clear()

    @classmethod
    def print_mess(cls, mes=""):
        # implement with print or warning if the project exist
        print(mes)
