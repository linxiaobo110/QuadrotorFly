#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The simplest test example

By xiaobo
Contact linxiaobo110@gmail.com
Created on  五月 06 17:34 2019
"""

# Copyright (C)
#
# This file is part of quadrotorfly
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

"""
********************************************************************************************************
**-------------------------------------------------------------------------------------------------------
**  Compiler   : python 3.6
**  Module Name: Test_simple
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


uavPara = Qfm.QuadParas()
# define the simulation parameters
simPara = Qfm.QuadSimOpt()
# define the first uav
quad1 = Qfm.QuadModel(uavPara, simPara)

# simulation begin
print("Simplest simulation begin!")
for i in range(1000):
    # set target，i.e., position in x,y,z, and yaw
    ref = np.array([0., 0., 1., 0.])
    # acquire the state of quadrotor，
    #   they are 12 degrees' vector，position in xyz，velocity in xyz,
    #   roll pitch yaw, and angular in roll pitch yaw
    stateTemp = quad1.observe()
    # calculate the control value; this is a pid controller example
    action2, oil = quad1.get_controller_pid(stateTemp, ref)
    # update the uav
    quad1.step(action2)
    
print("Simulation finish!")
