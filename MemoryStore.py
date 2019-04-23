#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:40:44 2018

@author: xiaobo
"""

from collections import deque
import random
import numpy as np

class ReplayBuffer(object):
    
    def __init__(self, buffer_size, random_seed = 123):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)
        self.isBufferFull = False
        self.eposideNum = 0
        self.eposidePos = deque()
        self.eposideRewards = deque()
    
        
    def bufferAppend(self, experience):
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
            self.isBufferFull = False
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
            self.isBufferFull = True
    
    def eposideAppend(self, rewards):
        self.eposideNum += 1
        self.eposidePos.append(self.count)
        self.eposideRewards.append(rewards)
            
    def size(self):
        return self.count
    
    def bufferSampleBatch(self, batch_size):
        batch = []
        if self.count <  batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)
        
        return batch
    
    def clear(self):
        self.buffer.clear()
        self.count = 0
        self.eposideCnt = 0
        self.eposidePos.clear()
        self.eposideRewards.clear()
        
class DataRecord(object):
    def __init__(self):
#        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
#        random.seed(random_seed)
#        self.isBufferFull = False
        self.eposideNum = 0
        self.eposidePos = deque()
        self.eposideRewards = deque()
        self.eposideTderr = deque()
        self.eposideWeights = deque()
        self.weights = deque()
        
    def bufferAppend(self, experience, weights = 0):
        self.buffer.append(experience)
        self.count += 1
        self.weights.append(weights)
#        if self.count < self.buffer_size:
#            self.buffer.append(experience)
#            self.count += 1
#            self.isBufferFull = False
#        else:
#            self.buffer.popleft()
#            self.buffer.append(experience)
#            self.isBufferFull = True
    
    def eposideAppend(self, rewards, tdErr=0, weights=0):
        self.eposideNum += 1
        self.eposidePos.append(self.count)
        self.eposideRewards.append(rewards)
        self.eposideTderr.append(tdErr)
        self.eposideWeights.append(weights)
            
    def size(self):
        return self.count
    
#    def bufferSampleBatch(self, batch_size):
#        batch = []
#        if self.count <  batch_size:
#            batch = random.sample(self.buffer, self.count)
#        else:
#            batch = random.sample(self.buffer, batch_size)
#        
#        return batch
    
    def clear(self):
        self.buffer.clear()
        self.count = 0
        self.eposideNum = 0
        self.eposidePos.clear()
        self.eposideRewards.clear()
        
    
        