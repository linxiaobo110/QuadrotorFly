#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""a more completed test example for QuadrotorFly with camdown module

By xiaobo
Contact linxiaobo110@gmail.com
Created on  十二月 07 13:03 2019
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
import CamDown
import os
import cv2
import time

"""
********************************************************************************************************
**-------------------------------------------------------------------------------------------------------
**  Compiler   : python 3.6
**  Module Name: Test_fullWithCamdown
**  Module Date: 2019/12/7
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
        actuator_mode=Qfm.ActuatorMode.dynamic, enable_sensor_sys=False,
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
quad1 = Qfm.QuadModel(uav_para=uavPara, sim_para=simPara)

# create the UAV camdown object
cam1 = CamDown.CamDown(
    # the resolution and depth (RGB is 3, gray is 1) of the the sensor
    img_horizon=400, img_vertical=400, img_depth=3,
    # the physical size of the active area of the sensor (unit is mm), the focal of the lens (unit is mm)
    sensor_horizon=4., sensor_vertical=4., cam_focal=2.36,
    # the path of the ground image, used in MEM render mode, doesn't need in other mode
    ground_img_path='../Data/groundImgWood.jpg',
    # the path of the mapping image, used in other render mode except MEM
    small_ground_img_path='../Data/groundImgSmall.jpg',
    # the path of the mapping image(in the center), used in other render mode except MEM
    small_land_img_path='../Data/landingMark.jpg',
    # render mode, could be Render_Mode_Gpu, Render_Mode_Cpu, Render_Mode_Mem, GPU mode is recommended
    render_mode=CamDown.CamDownPara.Render_Mode_Gpu)
# load ground image
cam1.load_ground_img()

# create path for video store
if not os.path.exists('../Data/img'):
    os.mkdir('../Data/img')
v_format = cv2.VideoWriter_fourcc(*'MJPG')
out1 = cv2.VideoWriter('../Data/img/test.avi', v_format, 1 / quad1.uavPara.ts, (cam1.imgVertical, cam1.imgHorizon))

# create the gui with created UAV
gui = Qfg.QuadrotorFlyGui([quad1])

# crate the record, used to record the data between UAV fly
record = MemoryStore.DataRecord()

# reset the system
quad1.reset_states()
record.clear()

t = np.arange(0, 15, 0.01)
ii_len = len(t)

print('Camera module test begin:')
time_start = time.time()
# simulation process
for ii in range(ii_len):
    # get the state of the quadrotor
    stateTemp = quad1.observe()
    # calculate image according the state
    pos_0 = quad1.position * 1000
    att_0 = quad1.attitude
    img1 = cam1.get_img_by_state(pos_0, att_0)

    # calculate the control value
    action, oil = quad1.get_controller_pid(stateTemp, np.array([0, 0, 3, 0]))
    # execute the dynamic of quadrotor
    quad1.step(action)
    print('action:', action)

    # store the real state and the estimated state
    record.buffer_append((stateTemp, action))
    # write the video steam
    out1.write(img1)
    gui.render()
out1.release()
# gui.render()
time_end = time.time()
print('time cost:', str(time_end - time_start))
record.episode_append()

# output the result
# 1. get data form record
data = record.get_episode_buffer()
# 1.1 get the state
bs = data[0]
# 1.2 get the action
ba = data[1]

# 2. plot result
t = range(0, record.count)
fig1 = plt.figure(2)
plt.clf()
# 2.1 figure attitude
plt.subplot(3, 1, 1)
plt.plot(t, bs[t, 6] / D2R, label='roll')
plt.plot(t, bs[t, 7] / D2R, label='pitch')
plt.plot(t, bs[t, 8] / D2R, label='yaw')
plt.ylabel('Attitude $(\circ)$', fontsize=15)
plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
# 2.2 figure position
plt.subplot(3, 1, 2)
plt.plot(t, bs[t, 0], label='x')
plt.plot(t, bs[t, 1], label='y')
plt.ylabel('Position (m)', fontsize=15)
plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
#  2.3 figure position
plt.subplot(3, 1, 3)
plt.plot(t, bs[t, 2], label='z')
plt.ylabel('Altitude (m)', fontsize=15)
plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
plt.show()
print("Simulation finish!")
print('Saved video is under the ../Data/img/test.mp4')
