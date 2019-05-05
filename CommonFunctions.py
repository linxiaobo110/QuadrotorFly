#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This file implement many common methods and constant

By xiaobo
Contact linxiaobo110@gmail.com
Created on  五月 05 11:03 2019
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
import warnings

"""
********************************************************************************************************
**-------------------------------------------------------------------------------------------------------
**  Compiler   : python 3.6
**  Module Name: CommonFunctions
**  Module Date: 2019/5/5
**  Module Auth: xiaobo
**  Version    : V0.1
**  Description: create the file
**-------------------------------------------------------------------------------------------------------
**  Reversion  :
**  Modified By:
**  Date       :
**  Content    :
**  Notes      :
********************************************************************************************************/
"""


class QuadrotorFlyError(Exception):
    """General exception of QuadrotorFly"""
    def __init__(self, error_info):
        super().__init__(self)
        self.errorInfo = error_info
        warnings.warn("QuadrotorFly Error:" + self.errorInfo, DeprecationWarning)

    def __str__(self):
        return "QuadrotorFly Error:" + self.errorInfo


def get_rotation_matrix(att):
    cos_att = np.cos(att)
    sin_att = np.sin(att)

    rotation_x = np.array([[1, 0, 0], [0, cos_att[0], -sin_att[0]], [0, sin_att[0], cos_att[0]]])
    rotation_y = np.array([[cos_att[1], 0, sin_att[1]], [0, 1, 0], [-sin_att[1], 0, cos_att[1]]])
    rotation_z = np.array([[cos_att[2], -sin_att[2], 0], [sin_att[2], cos_att[2], 0], [0, 0, 1]])
    rotation_matrix = np.dot(rotation_z, np.dot(rotation_y, rotation_x))

    return rotation_matrix


if __name__ == '__main__':
    try:
        raise QuadrotorFlyError('Quadrotor Exception Test')
    except QuadrotorFlyError as e:
        print(e)
