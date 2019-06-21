#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""'abstract class for sensors, define the general call interface'

By xiaobo
Contact linxiaobo110@gmail.com
Created on  六月 20 22:35 2019
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
import enum
from enum import Enum
import abc

"""
********************************************************************************************************
**-------------------------------------------------------------------------------------------------------
**  Compiler   : python 3.6
**  Module Name: SensorBase
**  Module Date: 2019/6/20
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


class SensorType(Enum):
    """Define the sensor types"""
    none = enum.auto()
    imu = enum.auto()
    compass = enum.auto()
    gps = enum.auto()


class SensorBase(object, metaclass=abc.ABCMeta):
    """Define the abstract sensor_base class"""
    sensorType = SensorType.none

    def __init__(self):
        super(SensorBase, self).__init__()

    def get_data(self):
        """return the sensor data"""
        pass

    def update(self, real_state, ts):
        """Calculating the output data of sensor according to real state of vehicle,
            the difference between update and get_data is that this method will be called when system update,
            but the get_data is called when user need the sensor data.
            :param real_state: real system state from vehicle
            :param ts: sample time
        """
        pass

    def reset(self, real_state):
        """reset the sensor"""
        pass

    def get_name(self):
        """get the name of sensor, format: type:model-no"""
        pass
