#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This file implement the GUI for QuadrotorFly
This module refer the 'quadcopter simulator' by abhijitmajumdar

By xiaobo
Contact linxiaobo110@gmail.com
Created on  Apr 29 20:53 2019
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
import QuadrotorFly.QuadrotorFlyModel as Qfm
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3d

"""
********************************************************************************************************
**-------------------------------------------------------------------------------------------------------
**  Compiler   : python 3.6
**  Module Name: QuadrotorFlyGui
**  Module Date: 2019/4/29
**  Module Auth: xiaobo
**  Version    : V0.1
**  Description: 
**-------------------------------------------------------------------------------------------------------
**  Reversion  :
**  Modified By:
**  Date       :
**  Content    :
**  Notes      :
********************************************************************************************************/
"""


class QuadrotorFlyGuiEnv(object):
    def __init__(self, bound_x=3., bound_y=3., bound_z=5.):
        """Define the environment of quadrotor simulation
        :param bound_x:
        :param bound_y:
        :param bound_z:
        """
        self.fig = plt.figure()
        self.boundX = bound_x * 1.
        self.boundY = bound_y * 1.
        self.boundZ = bound_z * 1.
        self.ax = Axes3d.Axes3D(self.fig)
        self.ax.set_xlim3d([-self.boundX, self.boundX])
        self.ax.set_ylim3d([-self.boundY, self.boundY])
        self.ax.set_zlim3d([0, self.boundZ])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('QuadrotorFly Simulation')


def get_rotation_matrix(att):
    cos_att = np.cos(att)
    sin_att = np.sin(att)

    rotation_x = np.array([[1, 0, 0], [0, cos_att[0], -sin_att[0]], [0, sin_att[0], cos_att[0]]])
    rotation_y = np.array([[cos_att[1], 0, sin_att[1]], [0, 1, 0], [-sin_att[1], 0, cos_att[1]]])
    rotation_z = np.array([[cos_att[2], -sin_att[2], 0], [sin_att[2], cos_att[2], 0], [0, 0, 1]])
    rotation_matrix = np.dot(rotation_z, np.dot(rotation_y, rotation_x))

    return rotation_matrix


class QuadrotorFlyGuiUav(object):
    """Draw quadrotor class"""
    def __init__(self, quad: Qfm.QuadModel, ax: Axes3d.Axes3D):
        self.quad = quad
        self.ax = ax
        # init the gui for quadrotor uav
        if quad.uavPara.structureType == Qfm.StructureType.quad_plus:
            self.barX, = self.ax.plot([], [], [], color='red', linewidth=3, antialiased=False)
            self.barY, = self.ax.plot([], [], [], color='black', linewidth=3, antialiased=False)
            self.hub, = self.ax.plot([], [], [], marker='o', color='green', markersize=6, antialiased=False)
        elif quad.uavPara.structureType == Qfm.StructureType.quad_x:
            self.hub, = self.ax.plot([], [], [], marker='o', color='green', markersize=6, antialiased=False)
            self.frontBar1, = self.ax.plot([], [], [], color='red', linewidth=3, antialiased=False)
            self.frontBar2, = self.ax.plot([], [], [], color='red', linewidth=3, antialiased=False)
            self.backBar1, = self.ax.plot([], [], [], color='black', linewidth=3, antialiased=False)
            self.backBar2, = self.ax.plot([], [], [], color='black', linewidth=3, antialiased=False)

    def render(self):
        uav_l = self.quad.uavPara.uavL
        position = self.quad.position
        if self.quad.uavPara.structureType == Qfm.StructureType.quad_plus:
            attitude = self.quad.attitude
            rot_matrix = get_rotation_matrix(attitude)
            points = np.array([[-uav_l, 0, 0], [uav_l, 0, 0], [0, -uav_l, 0], [0, uav_l, 0], [0, 0, 0], [0, 0, 0]]).T
            points_rotation = np.dot(rot_matrix, points)
            points_rotation[0, :] += position[0]
            points_rotation[1, :] += position[1]
            points_rotation[2, :] += position[2]
            self.barX.set_data(points_rotation[0, 0:2], points_rotation[1, 0:2])
            self.barX.set_3d_properties(points_rotation[2, 0:2])
            self.barY.set_data(points_rotation[0, 2:4], points_rotation[1, 2:4])
            self.barY.set_3d_properties(points_rotation[2, 2:4])
            self.hub.set_data(points_rotation[0, 5], points_rotation[1, 5])
            self.hub.set_3d_properties(points_rotation[2, 5])
        elif self.quad.uavPara.structureType == Qfm.StructureType.quad_x:
            attitude = self.quad.attitude
            rot_matrix = get_rotation_matrix(attitude)
            pos_rotor = uav_l * np.sqrt(0.5)
            # this points is the position of rotor in the body frame; the [0, 0, 0] is the center of UAV;
            #  and the sequence is front_left, front_right, back_left, back_right.
            points = np.array([[pos_rotor, pos_rotor, 0], [0, 0, 0], [pos_rotor, -pos_rotor, 0], [0, 0, 0],
                              [-pos_rotor, pos_rotor, 0], [0, 0, 0], [-pos_rotor, -pos_rotor, 0], [0, 0, 0]]).T
            # trans axi from body-frame to world-frame
            points_rotation = np.dot(rot_matrix, points)
            points_rotation[0, :] += position[0]
            points_rotation[1, :] += position[1]
            points_rotation[2, :] += position[2]
            self.frontBar1.set_data(points_rotation[0, 0:2], points_rotation[1, 0:2])
            self.frontBar2.set_data(points_rotation[0, 2:4], points_rotation[1, 2:4])
            self.backBar1.set_data(points_rotation[0, 4:6], points_rotation[1, 4:6])
            self.backBar2.set_data(points_rotation[0, 6:8], points_rotation[1, 6:8])
            self.frontBar1.set_3d_properties(points_rotation[2, 0:2])
            self.frontBar2.set_3d_properties(points_rotation[2, 2:4])
            self.backBar1.set_3d_properties(points_rotation[2, 4:6])
            self.backBar2.set_3d_properties(points_rotation[2, 6:8])
            self.hub.set_data(position[0], position[1])
            self.hub.set_3d_properties(position[2])


class QuadrotorFlyGui(object):
    """ Gui manage class"""
    def __init__(self, quad: Qfm.QuadModel):
        self.quad = quad
        self.env = QuadrotorFlyGuiEnv()
        self.ax = self.env.ax
        self.quadGui = QuadrotorFlyGuiUav(self.quad, self.ax)

    def render(self):
        self.quadGui.render()
        plt.pause(0.000000000000001)


if __name__ == '__main__':
    import MemoryStore
    " used for testing this module"
    D2R = Qfm.D2R
    testFlag = 1
    if testFlag == 1:
        # import matplotlib as mpl
        print("PID  controller test: ")
        uavPara = Qfm.QuadParas(structure_type=Qfm.StructureType.quad_x)
        simPara = Qfm.QuadSimOpt(init_mode=Qfm.SimInitType.fixed,
                                 init_att=np.array([10., 10., 0]), init_pos=np.array([0, 3, 0]))
        quad1 = Qfm.QuadModel(uavPara, simPara)
        record = MemoryStore.ReplayBuffer(10000, 1)
        record.clear()

        # gui init
        gui = QuadrotorFlyGui(quad1)

        # simulation begin
        step_cnt = 0
        for i in range(1000):
            ref = np.array([0., 0., 1., 0.])
            stateTemp = quad1.observe()
            action2, oil = quad1.get_controller_pid(stateTemp, ref)
            print('action: ', action2)
            action2 = np.clip(action2, 0.1, 0.9)
            quad1.step(action2)
            gui.render()
            record.buffer_append((stateTemp, action2))
            step_cnt = stateTemp + 1

        print('Quadrotor structure type', quad1.uavPara.structureType)
        # quad1.reset_states()
        print('Quadrotor get reward:', quad1.get_reward())
        bs = np.array([_[0] for _ in record.buffer])
        ba = np.array([_[1] for _ in record.buffer])
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
