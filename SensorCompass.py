#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implement the sensor details about compass

By xiaobo
Contact linxiaobo110@gmail.com
Created on  六月 23 11:24 2019
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
from SensorBase import SensorBase, SensorType
import CommonFunctions as Cf

"""
********************************************************************************************************
**-------------------------------------------------------------------------------------------------------
**  Compiler   : python 3.6
**  Module Name: SensorCompass
**  Module Date: 2019/6/23
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


class CompassPara(object):
    def __init__(self, max_update_frequency=50, start_delay=0, latency=0, name="compass",
                 accuracy=0.5):
        """
        :param max_update_frequency: max-update-frequency supported, Hz
        :param start_delay: the sensor start after this time, s
        :param latency: the state captured is indeed the state before, s
        :param name: the name of sensor,
        :param accuracy: the accuracy, uT
        """
        self.minTs = 1 / max_update_frequency
        self.startDelay = start_delay
        self.latency = latency
        self.name = name
        self.accuracy = accuracy

        # the world-frame used in QuadrotorFly is East-North-Sky
        # varying magnetic filed or fixed one
        self.refFlagFixed = True
        self.refField = np.array([9.805, 34.252, -93.438])


class SensorCompass(SensorBase):

    def __init__(self, para=CompassPara()):
        """
        :param para:
        """
        SensorBase.__init__(self)
        self.para = para
        self.sensorType = SensorType.compass
        self.magMea = np.zeros(3)

    def observe(self):
        """return the sensor data"""
        return self._isUpdated, self.magMea

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

        # process the update period
        if (ts - self._lastTick) >= self.para.minTs:
            self._isUpdated = True
            self._lastTick = ts
        else:
            self._isUpdated = False

        if self._isUpdated:
            # Magnetic sensor
            mag_world = self.para.refField
            rot_matrix = Cf.get_rotation_inv_matrix(real_state[6:9])
            acc_body = np.dot(rot_matrix, mag_world)
            noise_mag = (1 * np.random.random(3) - 0.5) * np.sqrt(self.para.accuracy)
            self.magMea = acc_body + noise_mag
        else:
            # keep old
            pass

        return self.observe()

    def reset(self, real_state):
        """reset the sensor"""
        self._lastTick = 0

    def get_name(self):
        """get the name of sensor, format: type:model-no"""
        return self.para.name


if __name__ == '__main__':
    " used for testing this module"
    D2R = Cf.D2R
    testFlag = 1
    if testFlag == 1:
        from QuadrotorFly import QuadrotorFlyModel as Qfm
        q1 = Qfm.QuadModel(Qfm.QuadParas(), Qfm.QuadSimOpt(init_mode=Qfm.SimInitType.fixed,
                                                           init_att=np.array([5, 6, 8])))
        s1 = SensorCompass()
        t = np.arange(0, 10, 0.01)
        ii_len = len(t)
        stateArr = np.zeros([ii_len, 12])
        meaArr = np.zeros([ii_len, 3])
        for ii in range(ii_len):
            state = q1.observe()
            action, oil = q1.get_controller_pid(state)
            q1.step(action)

            flag, meaArr[ii] = s1.update(state, q1.ts)
            stateArr[ii] = state

        estArr = np.zeros(ii_len)
        for i, x in enumerate(meaArr):
            temp = Cf.get_rotation_matrix(stateArr[i, 6:9])
            ref_temp = np.dot(temp, np.array([0, 0, s1.para.refField[2]]))
            # estArr[i] = np.arctan2(temp[0], temp[1])
            mea_temp = meaArr[i, :]
            # mea_temp[0] -= s1.para.refField[2] * np.sin(stateArr[i, 7])
            # mea_temp[1] -= s1.para.refField[2] * np.sin(stateArr[i, 6])
            print(mea_temp, ref_temp)
            mag_body1 = mea_temp[1] * np.cos(stateArr[i, 6]) - mea_temp[2] * np.sin(stateArr[i, 6])
            mag_body2 = (mea_temp[0] * np.cos(stateArr[i, 7]) +
                         mea_temp[1] * np.sin(stateArr[i, 6]) * np.sin(stateArr[i, 7]) +
                         mea_temp[2] * np.cos(stateArr[i, 6]) * np.sin(stateArr[i, 7]))
            if (mag_body1 != 0) and (mag_body2 != 0):
                estArr[i] = np.arctan2(-mag_body1, mag_body2) + 90 * D2R - 16 * D2R

        print((estArr[100] - stateArr[100, 9]) / D2R)
        import matplotlib.pyplot as plt
        plt.figure(3)
        plt.plot(t, stateArr[:, 8] / D2R, '-b', label='real')
        plt.plot(t, estArr / D2R, '-g', label='mea')
        plt.show()
        # plt.plot(t, flagArr * 100, '-r', label='update flag')
