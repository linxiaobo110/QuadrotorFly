#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module is design a state estimator for quadrotor according the data from sensors

By xiaobo
Contact linxiaobo110@gmail.com
Created on  六月 24 19:17 2019
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
import QuadrotorFlyModel as Qfm
import abc
import CommonFunctions as Cf

"""
********************************************************************************************************
**-------------------------------------------------------------------------------------------------------
**  Compiler   : python 3.6
**  Module Name: StateEstimator
**  Module Date: 2019/6/24
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


class StateEstimatorBase(object, metaclass=abc.ABCMeta):
    def __init__(self):
        super(StateEstimatorBase, self).__init__()
        self._name = "State estimator"

    def update(self, sensor_data, ts):
        """Calculating the output data of sensor according to real state of vehicle,
            the difference between update and get_data is that this method will be called when system update,
            but the get_data is called when user need the sensor data.
            :param sensor_data real system state from vehicle
            :param ts: the system tick
        """
        pass

    def reset(self, state_init):
        pass

    @property
    def name(self):
        return self._name


class KalmanFilterSimple(StateEstimatorBase):
    def __init__(self):
        StateEstimatorBase.__init__(self)
        self.state = np.zeros(12)
        self.imuTickPre = 0
        self.gpsTickPre = 0
        self.magTickPre = 0
        self.gyroBias = np.zeros(3)
        self.accBias = np.zeros(3)
        self.magRef = np.zeros(3)

    def reset(self, state_init):
        self.state = state_init

    def update(self, sensor_data, ts):
        # print(sensor_data)
        # print(sensor_data['imu'])
        # print()
        data_imu = sensor_data['imu'][1]
        data_gps = sensor_data['gps'][1]
        data_mag = sensor_data['compass'][1]

        angle_mea = np.zeros(3)
        # pos_pct = np.zeros(3)
        # angle_pct = np.zeros(3)
        # pos_mea = np.zeros(3)

        if sensor_data['imu'][0]:
            period_temp = ts - self.imuTickPre
            self.imuTickPre = ts

            # attitude estimation
            angle_pct = self.state[6:9] + (data_imu[3:6] - self.gyroBias) * period_temp

            angle_acc = np.zeros(3)
            mea_acc_temp = data_imu[0:3] - self.accBias
            acc_sum1 = np.sqrt(np.square(mea_acc_temp[1]) + np.square(mea_acc_temp[2]))
            acc_sum2 = np.sqrt(np.square(mea_acc_temp[0]) + np.square(mea_acc_temp[2]))
            angle_acc[1] = np.arctan2(mea_acc_temp[0], acc_sum1)
            angle_acc[0] = -np.arctan2(mea_acc_temp[1], acc_sum2)
            angle_mea[0:2] = angle_pct[0:2] + 0.1 * (angle_acc[0:2] - angle_pct[0:2])
            self.state[6:8] = angle_mea[0:2]
            self.state[9:12] = data_imu[3:6]
            self.state[8] = angle_pct[2]

            # velocity estimation
            rot_matrix = Cf.get_rotation_inv_matrix(self.state[6:9])
            acc_ext = np.dot(rot_matrix, mea_acc_temp) + np.array([0, 0, 9.8])
            # print(acc_ext, mea_acc_temp)
            acc_ext[2] = mea_acc_temp[2] / np.cos(self.state[6]) / np.cos(self.state[7]) + 9.8
            self.state[3:5] = self.state[3:5] + acc_ext[0:2] * period_temp * 0.5
            self.state[0:2] = self.state[0:2] + self.state[3:5] * period_temp * 0.5
            self.state[5] = self.state[5] + acc_ext[2] * period_temp * 5
            self.state[2] = self.state[2] + self.state[5] * period_temp

        if sensor_data['compass'][0]:
            roll_cos = np.cos(self.state[6])
            roll_sin = np.sin(self.state[6])
            pitch_cos = np.cos(self.state[7])
            pitch_sin = np.sin(self.state[7])
            mag_body1 = data_mag[1] * roll_cos - data_mag[2] * roll_sin
            mag_body2 = (data_mag[0] * pitch_cos +
                         data_mag[1] * roll_sin * pitch_sin +
                         data_mag[2] * roll_cos * pitch_sin)
            if (mag_body1 != 0) and (mag_body2 != 0):
                angle_mag = np.arctan2(-mag_body1, mag_body2) + 90 * D2R - 16 * D2R
                angle_mea[2] = self.state[8] + 0.1 * (angle_mag - self.state[8])
                self.state[8] = angle_mea[2]
                # self.state[8] = angle_pct[2]

        if sensor_data['gps'][0]:
            period_temp = ts - self.gpsTickPre
            self.gpsTickPre = ts

            # position
            pos_pct = self.state[0:3]  # + self.state[3:6] * period_temp # the predict has been done with acc
            pos_mea = pos_pct + 0.2 * (data_gps - pos_pct)

            # velocity
            vel_gps = (data_gps - self.state[0:3]) / period_temp
            vel_mea12 = self.state[3:5] + (pos_mea[0:2] - pos_pct[0:2]) * 0.1 + (vel_gps[0:2] - self.state[3:5]) * 0.00
            vel_mea3 = self.state[5] + (pos_mea[2] - pos_pct[2]) * 0.1 + (vel_gps[2] - self.state[5]) * 0.003

            self.state[0:3] = pos_mea[0:3]
            self.state[3:6] = np.hstack([vel_mea12, vel_mea3])

        return self.state


if __name__ == '__main__':
    " used for testing this module"
    D2R = Cf.D2R
    testFlag = 1
    if testFlag == 1:
        # from QuadrotorFly import QuadrotorFlyModel as Qfm
        q1 = Qfm.QuadModel(Qfm.QuadParas(), Qfm.QuadSimOpt(init_mode=Qfm.SimInitType.fixed, enable_sensor_sys=True,
                                                           init_pos=np.array([5, -4, 0]), init_att=np.array([0, 0, 5])))
        # init the estimator
        s1 = KalmanFilterSimple()
        # set the init state of estimator
        s1.reset(q1.state)
        # simulation period
        t = np.arange(0, 30, 0.01)
        ii_len = len(t)
        stateRealArr = np.zeros([ii_len, 12])
        stateEstArr = np.zeros([ii_len, 12])
        meaArr = np.zeros([ii_len, 3])

        # set the bias
        s1.gyroBias = q1.imu0.gyroBias
        s1.accBias = q1.imu0.accBias
        s1.magRef = q1.mag0.para.refField
        print(s1.gyroBias, s1.accBias)

        for ii in range(ii_len):
            # wait for start
            if ii < 100:
                sensor_data1 = q1.observe()
                _, oil = q1.get_controller_pid(q1.state)
                action = np.ones(4) * oil
                q1.step(action)
                stateEstArr[ii] = s1.update(sensor_data1, q1.ts)
                stateRealArr[ii] = q1.state
            else:
                sensor_data1 = q1.observe()
                action, oil = q1.get_controller_pid(s1.state, np.array([0, 0, 4, 0]))
                q1.step(action)
                stateEstArr[ii] = s1.update(sensor_data1, q1.ts)
                stateRealArr[ii] = q1.state
        import matplotlib.pyplot as plt
        plt.figure(1)
        ylabelList = ['roll', 'pitch', 'yaw', 'rate_roll', 'rate_pit', 'rate_yaw']
        for ii in range(6):
            plt.subplot(6, 1, ii + 1)
            plt.plot(t, stateRealArr[:, 6 + ii] / D2R, '-b', label='real')
            plt.plot(t, stateEstArr[:, 6 + ii] / D2R, '-g', label='est')
            plt.legend()
            plt.ylabel(ylabelList[ii])
        # plt.show()

        ylabelList = ['p_x', 'p_y', 'p_z', 'vel_x', 'vel_y', 'vel_z']
        plt.figure(2)
        for ii in range(6):
            plt.subplot(6, 1, ii + 1)
            plt.plot(t, stateRealArr[:, ii], '-b', label='real')
            plt.plot(t, stateEstArr[:, ii], '-g', label='est')
            plt.legend()
            plt.ylabel(ylabelList[ii])
        plt.show()

