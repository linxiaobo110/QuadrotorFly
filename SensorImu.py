#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implement the sensor details about imu

By xiaobo
Contact linxiaobo110@gmail.com
Created on  六月 21 10:33 2019
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
# import QuadrotorFly.SensorBase as SensorBase
from QuadrotorFly.SensorBase import SensorBase, SensorType
import QuadrotorFly.CommonFunctions as Cf

"""
********************************************************************************************************
**-------------------------------------------------------------------------------------------------------
**  Compiler   : python 3.6
**  Module Name: SensorImu
**  Module Date: 2019/6/21
**  Module Auth: xiaobo
**  Version    : V0.1
**  Description: this module referenced the source of sensor part in AirSim, data sheet of mpu6050
**-------------------------------------------------------------------------------------------------------
**  Reversion  :
**  Modified By:
**  Date       :
**  Content    :
**  Notes      :
********************************************************************************************************/
"""

D2R = Cf.D2R
g = 9.8


class ImuPara(object):
    def __init__(self, gyro_zro_tolerance_init=5, gyro_zro_var=30, gyro_noise_sd=0.01, min_time_sample=0.01,
                 acc_zgo_tolerance=60, acc_zg_var_temp=1.5, acc_noise_sd=300, name='imu'
                 ):
        """
        zro is zero rate output, sd is spectral density, zgo is zero g output
        :param gyro_zro_tolerance_init: the zero-bias of gyro, \deg/s
        :param gyro_zro_var: the noise variation in normal temperature, \deg/s
        :param gyro_noise_sd: the rate noise spectral density, \deg/s/\sqrt(Hz)
        :param acc_zgo_tolerance: the zeros of acc, mg
        :param acc_zg_var_temp: the noise variation vs temperature (-40~85), mg/(\degC)
        :param acc_noise_sd: the rate noise spectral density, mg\sqrt(Hz)
        :param name: the name of sensor
        :param min_time_sample: min sample time
        """
        # transfer the unit for general define
        # transfer to \rad/s
        self.gyroZroToleranceInit = gyro_zro_tolerance_init * D2R
        self.gyroZroVar = gyro_zro_var * (D2R**2)
        self.gyroNoiseSd = gyro_noise_sd  # i do not understand it, in fact
        # transfer to m/(s^2)
        self.accZroToleranceInit = acc_zgo_tolerance / 1000 * g
        std_temp = 1 / 1000 * g * 60  # assumed the temperature is 20 (\degC) here
        self.accZroVar = acc_zg_var_temp * (std_temp**2)
        self.accNoiseSd = acc_noise_sd  # i do not understand it, in fact
        self.name = name
        self.minTs = min_time_sample


mpu6050 = ImuPara(5, 30, 0.01)


class SensorImu(SensorBase):

    def __init__(self, imu_para=mpu6050):
        """
        :param imu_para:
        """
        SensorBase.__init__(self)
        self.para = imu_para
        self.sensorType = SensorType.imu
        self.angularMea = np.zeros(3)
        self.gyroBias = (1 * np.random.random(3) - 0.5) * self.para.gyroZroToleranceInit
        self.accMea = np.zeros(3)
        self.accBias = (1 * np.random.random(3) - 0.5) * self.para.accZroToleranceInit

    def observe(self):
        """return the sensor data"""
        return self._isUpdated, np.hstack([self.accMea, self.angularMea])

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
            # gyro
            noise_gyro = (1 * np.random.random(3) - 0.5) * np.sqrt(self.para.gyroZroVar)
            self.angularMea = real_state[9:12] + noise_gyro + self.gyroBias

            # accelerator
            acc_world = real_state[0:3] + np.array([0, 0, -g])
            rot_matrix = Cf.get_rotation_inv_matrix(real_state[6:9])
            acc_body = np.dot(rot_matrix, acc_world)
            noise_acc = (1 * np.random.random(3) - 0.5) * np.sqrt(self.para.gyroZroVar)
            self.accMea = acc_body + noise_acc + self.accBias
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
    testFlag = 2
    if testFlag == 1:
        s1 = SensorImu()
        flag1, v1 = s1.update(np.random.random(12), 0.1)
        flag2, v2 = s1.update(np.random.random(12), 0.105)
        print(flag1, "val", v1, flag2, "val", v2)

    elif testFlag == 2:
        from QuadrotorFly import QuadrotorFlyModel as Qfm
        q1 = Qfm.QuadModel(Qfm.QuadParas(), Qfm.QuadSimOpt())
        s1 = SensorImu()
        t = np.arange(0, 10, 0.01)
        ii_len = len(t)
        stateArr = np.zeros([ii_len, 12])
        meaArr = np.zeros([ii_len, 6])
        for ii in range(ii_len):
            state = q1.observe()
            action, oil = q1.get_controller_pid(state)
            q1.step(action)

            flag, meaArr[ii] = s1.update(state, q1.ts)
            stateArr[ii] = state

        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.plot(t, stateArr[:, 9:12], '-b', label='real')
        plt.plot(t, meaArr[:, 3:6], '-g', label='measure')
        plt.show()
        plt.figure(2)
        plt.plot(t, stateArr[:, 3:6], '-b', label='real')
        plt.plot(t, meaArr[:, 0:3], '-g', label='measure')
        plt.show()
        # plt.plot(t, flagArr * 100, '-r', label='update flag')
