#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implement the sensor details about Gps

By xiaobo
Contact linxiaobo110@gmail.com
Created on  六月 21 22:59 2019
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


import numpy as np
from QuadrotorFly.SensorBase import SensorBase, SensorType
import queue

"""
********************************************************************************************************
**-------------------------------------------------------------------------------------------------------
**  Compiler   : python 3.6
**  Module Name: SensorGps
**  Module Date: 2019/6/21
**  Module Auth: xiaobo
**  Version    : V0.1
**  Description: 'Replace the content between'
**-------------------------------------------------------------------------------------------------------
**  Reversion  :
**  Modified By:
**  Date       :
**  Content    :
**  Notes      :
********************************************************************************************************/
"""


class GpsPara(object):
    def __init__(self, max_update_frequency=10, start_delay=1, latency=0.2, name="gps",
                 accuracy_horizontal=2.5):
        """
        :param max_update_frequency: max-update-frequency supported, Hz
        :param start_delay: the sensor start after this time, s
        :param latency: the state captured is indeed the state before, s
        :param name: the name of sensor,
        :param accuracy_horizontal: the accuracy, m
        """
        self.minTs = 1 / max_update_frequency
        self.name = name
        self.startDelay = start_delay
        # important, the latency is assumed to be larger than minTs
        self.latency = latency
        self.accuracyHorizon = accuracy_horizontal


class SensorGps(SensorBase):
    def __init__(self, para=GpsPara()):
        SensorBase.__init__(self)
        self.sensorType = SensorType.gps
        self.para = para
        self._posMea = np.zeros(3)
        # the history data is used to implement the latency
        self._posHisReal = queue.Queue()

    def observe(self):
        """return the sensor data"""
        return self._isUpdated, self._posMea

    def update(self, real_state, ts):
        """Calculating the output data of sensor according to real state of vehicle,
            the difference between update and get_data is that this method will be called when system update,
            but the get_data is called when user need the sensor data.
            the real_state here should be a 12 degree vector,
            :param real_state:
            0       1       2       3       4       5
            p_x     p_y     p_z     v_x     v_y     v_z
            6       7       8       9       10      11
            roll    pitch   yaw     v_roll  v_pitch v_yaw
            :param ts: system tick now
        """
        # process the latency
        real_state_latency = np.zeros(3)
        if ts < self.para.latency:
            self._posHisReal.put(real_state)
        else:
            self._posHisReal.put(real_state)
            real_state_latency = self._posHisReal.get()

        # process the star_time
        if ts < self.para.startDelay:
            self._posMea = np.zeros(3)
            self._isUpdated = False
        else:
            # process the update period
            if (ts - self._lastTick) >= self.para.minTs:
                self._isUpdated = True
                self._lastTick = ts
            else:
                self._isUpdated = False

            if self._isUpdated:
                noise_gps = (1 * np.random.random(3) - 0.5) * self.para.accuracyHorizon
                self._posMea = real_state_latency[0:3] + noise_gps
            else:
                # keep old
                pass

        return self.observe()

    def reset(self, real_state):
        """reset the sensor"""
        self._lastTick = 0
        self._posMea = np.zeros(3)
        if not self._posHisReal.empty():
            self._posHisReal.queue.clear()

    def get_name(self):
        """get the name of sensor, format: type:model-no"""
        return self.para.name


if __name__ == '__main__':
    " used for testing this module"
    testFlag = 1
    if testFlag == 1:
        s1 = SensorGps()
        t1 = np.arange(0, 5, 0.01)
        nums = len(t1)
        vel = np.sin(t1)
        pos = np.zeros([nums, 3])
        posMea = np.zeros([nums, 3])
        flagArr = np.zeros([nums, 3])
        for ii in range(nums):
            if ii > 0:
                pos[ii] = pos[ii - 1] + vel[ii]

        for ii in range(nums):
            flagArr[ii], posMea[ii] = s1.update(np.hstack([pos[ii], np.zeros(9)]), t1[ii])

        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.plot(t1, pos, '-b', label='real')
        plt.plot(t1, posMea, '-g', label='measure')
        plt.plot(t1, flagArr * 100, '-r', label='update flag')
        plt.legend()
        plt.show()
