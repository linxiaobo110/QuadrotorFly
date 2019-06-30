#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""a more completed test example with sensor system for QuadrotorFly

By xiaobo
Contact linxiaobo110@gmail.com
Created on  六月 29 21:26 2019
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
import QuadrotorFlyGui as Qfg
import MemoryStore
import matplotlib.pyplot as plt
import StateEstimator

"""
********************************************************************************************************
**-------------------------------------------------------------------------------------------------------
**  Compiler   : python 3.6
**  Module Name: Test_fullWithSensor
**  Module Date: 2019/6/29
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

# translate degree to rad
D2R = Qfm.D2R
# set the simulation para
simPara = Qfm.QuadSimOpt(
        # initial mode（random or fixed, default is random）；init para for attitude（bound in random mode，
        # direct value in fixed mode）；init papa for position (like init_att)
        init_mode=Qfm.SimInitType.rand, init_att=np.array([5., 5., 5.]), init_pos=np.array([1., 1., 1.]),
        # max position(unit is m), max velocity(unit is m/s), max attitude(unit is degree), max angular(unit is deg/s)
        max_position=8, max_velocity=8, max_attitude=180, max_angular=200,
        # system noise (internal noisy, default is 0)
        sysnoise_bound_pos=0, sysnoise_bound_att=0,
        # mode for actuator(motor) (with dynamic or not), enable the sensor system (default is false)
        actuator_mode=Qfm.ActuatorMode.dynamic, enable_sensor_sys=True,
        )
# set the para for quadrotor
uavPara = Qfm.QuadParas(
        # gravity accelerate；sample period；frame type (plus or x)
        g=9.8, tim_sample=0.01, structure_type=Qfm.StructureType.quad_plus,
        # length of arm(m)；mass(kg)；the moment of inertia around xyz（kg·m^2）
        uav_l=0.45, uav_m=1.5, uav_ixx=1.75e-2, uav_iyy=1.75e-2, uav_izz=3.18e-2,
        # para from motor speed to thrust (N/(rad/s)), papa from motor speed to torque (N ·m/(rad/s)), MoI of motor
        rotor_ct=1.11e-5, rotor_cm=1.49e-7, rotor_i=9.9e-5,
        # proportional parameter from u to motor speed (deg/s), bias parameters from u to motor (deg/s), response time
        rotor_cr=646, rotor_wb=166, rotor_t=1.36e-2
        )

# all above is the default parameters expect the enable_sensor_sys
# crete the UAV with set parameters
# quad1 = Qfm.QuadModel(uavPara, simPara)
quad1 = Qfm.QuadModel(Qfm.QuadParas(), Qfm.QuadSimOpt(init_mode=Qfm.SimInitType.fixed, enable_sensor_sys=True,
                                                      init_pos=np.array([5, -3, 0]), init_att=np.array([0, 0, 5])))
# create the estimator
filter1 = StateEstimator.KalmanFilterSimple()

# create the gui with created UAV
gui = Qfg.QuadrotorFlyGui([quad1])

# crate the record, used to record the data between UAV fly
record = MemoryStore.DataRecord()

# reset the system
quad1.reset_states()
record.clear()
filter1.reset(quad1.state)

# set the real bias for filter directly, it is not real in fact, just a simplify
filter1.gyroBias = quad1.imu0.gyroBias
filter1.accBias = quad1.imu0.accBias
print(filter1.gyroBias, filter1.accBias)

t = np.arange(0, 30, 0.01)
ii_len = len(t)

# simulation process
for ii in range(ii_len):
    # wait for sensor start
    if ii < 100:
        # get the sensor data
        sensor_data1 = quad1.observe()
        # the control value is calculated with real state before the sensor start completely
        _, oil = quad1.get_controller_pid(quad1.state)
        # just use the oil signal as control value
        action = np.ones(4) * oil
        # execute the dynamic of quadrotor
        quad1.step(action)
        # feed the filter with sensor data which will will return the estimated state
        state_est = filter1.update(sensor_data1, quad1.ts)
        # store the real state and the estimated state
        record.buffer_append((quad1.state, state_est))
        # stateEstArr[ii] = filter1.update(sensor_data1, quad1.ts)
        # stateRealArr[ii] = quad1.state
    else:
        # get the sensor data
        sensor_data1 = quad1.observe()
        # calculate the system state based on the state estimated by kalman filter, not the real state
        action, oil = quad1.get_controller_pid(filter1.state, np.array([0, 0, 2, 0]))
        # execute the dynamic of quadrotor
        quad1.step(action)
        # feed the filter with sensor data which will will return the estimated state
        state_est = filter1.update(sensor_data1, quad1.ts)
        # print(state_est)
        # store the real state and the estimated state
        record.buffer_append((quad1.state, state_est.copy()))
        # stateEstArr[ii] = filter1.update(sensor_data1, quad1.ts)
        # stateRealArr[ii] = q1.state
    # gui.render()

record.episode_append()
# output the result
# 1. get data form record
data = record.get_episode_buffer()
# 1.1 the real state
bs_r = data[0]
# 1.2 the estimated state
bs_e = data[1]
# 2. generate the time sequence
t = range(0, record.count)
# 3. draw figure
# 3.1 draw the attitude and angular
fig1 = plt.figure(2)
yLabelList = ['roll', 'pitch', 'yaw', 'rate_roll', 'rate_pit', 'rate_yaw']
for ii in range(6):
    plt.subplot(6, 1, ii + 1)
    plt.plot(t, bs_r[:, 6 + ii] / D2R, '-b', label='real')
    plt.plot(t, bs_e[:, 6 + ii] / D2R, '-g', label='est')
    plt.legend()
    plt.ylabel(yLabelList[ii])

# 3.2 draw the position and velocity
yLabelList = ['p_x', 'p_y', 'p_z', 'vel_x', 'vel_y', 'vel_z']
plt.figure(3)
for ii in range(6):
    plt.subplot(6, 1, ii + 1)
    plt.plot(t, bs_r[:, ii], '-b', label='real')
    plt.plot(t, bs_e[:, ii], '-g', label='est')
    plt.legend()
    plt.ylabel(yLabelList[ii])
plt.show()
print("Simulation finish!")
