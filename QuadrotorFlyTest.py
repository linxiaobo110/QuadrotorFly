#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This file is used for testing the QuadrotorFly

By xiaobo
Contact linxiaobo110@gmail.com
Created on  五月 06 17:13 2019
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
from enum import Enum
import enum
import StateEstimator
import CamDown
import time
import cv2

"""
********************************************************************************************************
**-------------------------------------------------------------------------------------------------------
**  Compiler   : python 3.6
**  Module Name: QuadrotorFlyTest
**  Module Date: 2019/5/6
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


class TestPara(Enum):
    Test_Module_Dynamic = enum.auto()
    Test_Module_Dynamic_Sensor = enum.auto()
    Test_Module_Dynamic_CamDown = enum.auto()


D2R = Qfm.D2R
testFlag = TestPara.Test_Module_Dynamic_Sensor

if testFlag == TestPara.Test_Module_Dynamic:
    print("QuadrotorFly Dynamic Test: ")
    # define the quadrotor parameters
    uavPara = Qfm.QuadParas()
    # define the simulation parameters
    simPara = Qfm.QuadSimOpt(init_mode=Qfm.SimInitType.rand,
                             init_att=np.array([10., 10., 0]), init_pos=np.array([0, 3, 0]))
    # define the data capture
    record = MemoryStore.DataRecord()
    record.clear()
    # define the first uav
    quad1 = Qfm.QuadModel(uavPara, simPara)
    # define the second uav
    quad2 = Qfm.QuadModel(uavPara, simPara)
    # gui init
    gui = Qfg.QuadrotorFlyGui([quad1, quad2])

    # simulation begin
    for i in range(1000):
        # set the reference
        ref = np.array([0., 0., 1., 0.])

        # update the first uav
        stateTemp = quad1.observe()
        action2, oil = quad1.get_controller_pid(stateTemp, ref)
        quad1.step(action2)
        # update the second uav
        action2, oil2 = quad2.get_controller_pid(quad2.observe(), ref)
        quad2.step(action2)

        # gui render
        gui.render()

        # store data
        record.buffer_append((stateTemp, action2))

    # Data_recorder 0.3+ store episode data with independent deque
    record.episode_append()

    # draw result
    data = record.get_episode_buffer()
    bs = data[0]
    ba = data[1]
    t = range(0, record.count)
    fig1 = plt.figure(2)
    plt.clf()
    # draw position
    plt.subplot(3, 1, 1)
    plt.plot(t, bs[t, 6] / D2R, label='roll')
    plt.plot(t, bs[t, 7] / D2R, label='pitch')
    plt.plot(t, bs[t, 8] / D2R, label='yaw')
    plt.ylabel('Attitude $(\circ)$', fontsize=15)
    plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
    # draw position
    plt.subplot(3, 1, 2)
    plt.plot(t, bs[t, 0], label='x')
    plt.plot(t, bs[t, 1], label='y')
    plt.ylabel('Position (m)', fontsize=15)
    plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
    # draw altitude
    plt.subplot(3, 1, 3)
    plt.plot(t, bs[t, 2], label='z')
    plt.ylabel('Altitude (m)', fontsize=15)
    plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
    plt.show()

elif testFlag == TestPara.Test_Module_Dynamic_Sensor:
    # from QuadrotorFly import QuadrotorFlyModel as Qfm
    q1 = Qfm.QuadModel(Qfm.QuadParas(), Qfm.QuadSimOpt(init_mode=Qfm.SimInitType.fixed, enable_sensor_sys=True,
                                                       init_pos=np.array([5, -4, 0]), init_att=np.array([0, 0, 5])))
    # init the estimator
    s1 = StateEstimator.KalmanFilterSimple()
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
            action, oil = q1.get_controller_pid(s1.state, np.array([0, 0, 3, 0]))
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
elif testFlag == TestPara.Test_Module_Dynamic_CamDown:
    import matplotlib.pyplot as plt
    from QuadrotorFlyModel import QuadModel, QuadSimOpt, QuadParas, StructureType, SimInitType
    D2R = np.pi / 180
    video_write_flag = True

    print("PID  controller test: ")
    uavPara = QuadParas(structure_type=StructureType.quad_x)
    simPara = QuadSimOpt(init_mode=SimInitType.fixed, enable_sensor_sys=False,
                         init_att=np.array([5., -5., 0]), init_pos=np.array([5, -5, 0]))
    quad1 = QuadModel(uavPara, simPara)
    record = MemoryStore.DataRecord()
    record.clear()
    step_cnt = 0

    # init the camera
    cam1 = CamDown.CamDown(render_mode=CamDown.CamDownPara.Render_Mode_Gpu)
    cam1.load_ground_img()
    print('Load img completed!')
    if video_write_flag:
        v_format = cv2.VideoWriter_fourcc(*'MJPG')
        out1 = cv2.VideoWriter('Data/img/test.avi', v_format, 1 / quad1.uavPara.ts, (cam1.imgVertical, cam1.imgHorizon))
    for i in range(1000):
        if i == 0:
            time_start = time.time()
        ref = np.array([0., 0., 3., 0.])
        stateTemp = quad1.observe()
        # get image
        pos_0 = quad1.position * 1000
        att_0 = quad1.attitude
        img1 = cam1.get_img_by_state(pos_0, att_0)
        # file_name = 'Data/img/test_' + str(i) + '.jpg'
        # cv2.imwrite(file_name, img1)
        if video_write_flag:
            out1.write(img1)

        action2, oil = quad1.get_controller_pid(stateTemp, ref)
        print('action: ', action2)
        action2 = np.clip(action2, 0.1, 0.9)
        quad1.step(action2)
        record.buffer_append((stateTemp, action2))
        step_cnt = step_cnt + 1
    time_end = time.time()
    print('time cost:', str(time_end - time_start))
    record.episode_append()
    if video_write_flag:
        out1.release()

    print('Quadrotor structure type', quad1.uavPara.structureType)
    # quad1.reset_states()
    print('Quadrotor get reward:', quad1.get_reward())
    data = record.get_episode_buffer()
    bs = data[0]
    ba = data[1]
    t = range(0, record.count)
    # mpl.style.use('seaborn')
    fig1 = plt.figure(1)
    plt.clf()
    plt.subplot(3, 1, 1)
    plt.plot(t, bs[t, 6] / D2R, label='roll')
    plt.plot(t, bs[t, 7] / D2R, label='pitch')
    plt.plot(t, bs[t, 8] / D2R, label='yaw')
    plt.ylabel('Attitude $(\circ)$', fontsize=15)
    plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
    plt.subplot(3, 1, 2)
    plt.plot(t, bs[t, 0], label='x')
    plt.plot(t, bs[t, 1], label='y')
    plt.ylabel('Position (m)', fontsize=15)
    plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
    plt.subplot(3, 1, 3)
    plt.plot(t, bs[t, 2], label='z')
    plt.ylabel('Altitude (m)', fontsize=15)
    plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
    plt.show()
