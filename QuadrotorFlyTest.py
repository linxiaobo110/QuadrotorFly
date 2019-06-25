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

D2R = Qfm.D2R

print("QuadrotorFly Test: ")
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
