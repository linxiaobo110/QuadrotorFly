# -*- coding: utf-8 -*-
"""The file used to describe the dynamic of quadrotor UAV

The module include dynamic of quadrotor, actuator,
"""

# Author: xiaobo

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

"""
********************************************************************************************************
**-------------------------------------------------------------------------------------------------------
**  Compiler   : python 3.6
**  Module Name: QuadrotorFlyModel
**  Module Date: 2019-04-19
**  Module Auth: xiaobo
**  Version    : V0.1
**  Description: create the module
**-------------------------------------------------------------------------------------------------------
**  Reversion  :
**  Modified By:
**  Date       :
**  Content    :
**  Notes      :
********************************************************************************************************/
"""


def rk4(func, x0, u, h):
    """Runge-Kutta 4 order update function
    :param func: system dynamic
    :param x0: system state
    :param u: control input
    :param h: time of sample
    :return: state of next time
    """
    k1 = func(x0, u)
    k2 = func(x0 + h * k1 / 2, u)
    k3 = func(x0 + h * k2 / 2, u)
    k4 = func(x0 + h * k3, u)
    x1 = x0 + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x1


class QuadParas(object):
    """Define the parameters of quadrotor model

    """

# some constant
    D2R = np.pi / 180.

    def __init__(self, g=9.81, rotor_num=4, tim_sample=0.01,
                 uav_l=0.450, uav_m=1.50, uav_ixx=1.75e-2, uav_iyy=1.75e-2, uav_izz=3.18e-2,
                 rotor_ct=1.11e-5, rotor_cm=1.49e-7, rotor_cr=646, rotor_wb=166, rotor_i=9.90e-5, rotor_t=1.36e-2):
        """init the quadrotor paramteres
        These parameters are able to be estimation in web(https://flyeval.com/) if you do not have a real UAV.
        common parameters:
            -g          : N/kg,      acceleration gravity
            -rotor-num  : int,       number of rotors, e.g. 4, 6, 8
            -tim_sample : s,         sample time of system
        uav:
            -uav_l      : m,        distance from center of mass to center of rotor
            -uav_m      : kg,       the mass of quadrotor
            -uav_ixx    : kg.m^2    central principal moments of inertia of UAV in x
            -uav_iyy    : kg.m^2    central principal moments of inertia of UAV in y
            -uav_izz    : kg.m^2    central principal moments of inertia of UAV in z
        rotor (assume that four rotors are the same):
            -rotor_ct   : N/(rad/s)^2,      lump parameter thrust coefficient, which translate rate of rotor to thrust
            -rotor_cm   : N.m/(rad/s)^2,    lump parameter torque coefficient, like ct, usd in yaw
            -rotor_cr   : rad/s,            scale para which translate oil to rate of motor
            -rotor_wb   : rad/s,            bias para which translate oil to rate of motor
            -rotor_i    : kg.m^2,           inertia of moment of rotor(including motor and propeller)
            -rotor_t    : s,                time para of dynamic response of motor
        """
        self.g = g
        self.numOfRotors = rotor_num
        self.ts = tim_sample
        self.uavL = uav_l
        self.uavM = uav_m
        self.uavInertia = np.array([uav_ixx, uav_iyy, uav_izz])
        self.rotorCt = rotor_ct
        self.rotorCm = rotor_cm
        self.rotorCr = rotor_cr
        self.rotorWb = rotor_wb
        self.rotorInertia = rotor_i
        self.rotorTimScale = 1 / rotor_t


class SimInitType(Enum):
    rand = enum.auto()
    fixed = enum.auto()


class ActuatorMode(Enum):
    simple = enum.auto()
    dynamic = enum.auto()
    disturbance = enum.auto()
    dynamic_voltage = enum.auto()
    disturbance_voltage = enum.auto()


class QuadSimOpt(object):
    """contain the parameters for guiding the simulation process
    """

    def __init__(self, init_mode=SimInitType.rand, init_att=np.array([5, 5, 5]), init_pos=np.array([1, 1, 1]),
                 actuator_mode=ActuatorMode.simple):
        self.initMode = init_mode
        self.initAtt = init_att
        self.initPos = init_pos
        self.actuatorMode = actuator_mode


class QuadActuator(object):
    """Dynamic of  actuator including motor and propeller
    """

    def __init__(self, quad_para: QuadParas, mode: ActuatorMode):
        """Parameters is maintain together
        :param quad_para:   parameters of quadrotor,maintain together
        :param mode:        'simple', without dynamic of motor; 'dynamic' with dynamic;
        """
        self.para = quad_para
        self.motorPara_scale = self.para.rotorTimScale * self.para.rotorCr
        self.motorPara_bias = self.para.rotorTimScale * self.para.rotorWb
        self.mode = mode

        # states of actuator
        self.outThrust = np.zeros([self.para.numOfRotors])
        self.outTorque = np.zeros([self.para.numOfRotors])
        # rate of rotor
        self.rotorRate = np.zeros([self.para.numOfRotors])

    def dynamic_actuator(self, rotor_rate, u):
        """dynamic of motor and propeller
        input: rotorRate, u
        output: rotorRateDot,
        """

        rate_dot = self.motorPara_scale * u + self.motorPara_bias - self.para.rotorTimScale * rotor_rate
        return rate_dot

    def reset(self):
        """reset all state"""

        self.outThrust = np.zeros([self.para.numOfRotors])
        self.outTorque = np.zeros([self.para.numOfRotors])
        # rate of rotor
        self.rotorRate = np.zeros([self.para.numOfRotors])

    def step(self, u: 'int > 0'):
        """calculate the next state based on current state and u
        :param u:
        :return:
        """

        if u > 1:
            u = 1

        if self.mode == ActuatorMode.simple:
            # without dynamic of motor
            self.rotorRate = self.para.rotorCr * u + self.para.rotorWb
        elif self.mode == ActuatorMode.dynamic:
            # with dynamic of motor
            self.rotorRate = rk4(self.dynamic_actuator, self.rotorRate, u, self.para.ts)
        else:
            self.rotorRate = 0

        self.outThrust = self.para.rotorCt * np.square(self.rotorRate)
        self.outTorque = self.para.rotorCm * np.square(self.rotorRate)


class QuadDynamic(object):
    """module interface, main class including basic dynamic of quad
    """

    def __init__(self, uav_para: QuadParas, sim_para: QuadSimOpt):
        """init a quadrotor
        :param uav_para:    parameters of quadrotor,maintain together
        :param actuator_mode:        'simple', without dynamic of motor; 'dynamic' with dynamic;
        """
        self.uavPara = uav_para
        self.simPara = sim_para
        self.actuator = QuadActuator(self.uavPara, sim_para.actuatorMode)

        # states of quadrotor
        #   -position, m
        self.pos = np.array([0, 0, 0])
        #   -velocity, m/s
        self.velocity = np.array([0, 0, 0])
        #   -attitude, rad
        self.attitude = np.array([0, 0, 0])
        #   -angular, rad/s
        self.angular = np.array([0, 0, 0])

    def generate_init_att(self):
        """used to generate a init attitude according to simPara"""
        angle = self.simPara.initAtt * QuadParas.D2R
        if self.simPara.initMode == SimInitType.rand:
            phi = (1 * np.random.random() - 0.5) * angle[0]
            theta = (1 * np.random.random() - 0.5) * angle[1]
            psi = (1 * np.random.random() - 0.5) * angle[2]
        else:
            phi = angle[0]
            theta = angle[1]
            psi = angle[2]
        return np.array([phi, theta, psi])

    def generate_init_pos(self):
        pos = self.init_pos
        if self.init_mode == 'rand':
            x = (1 * np.random.random() - 0.5) * pos[0]
            y = (1 * np.random.random() - 0.5) * pos[1]
            z = (1 * np.random.random() - 0.5) * pos[2]
        else:
            x = pos[0]
            y = pos[1]
            z = pos[2]

        return np.array([x, y, z])


if __name__ == '__main__':
    " used for test each module"
    # test for actuator
    qp = QuadParas()
    ac0 = QuadActuator(qp)
    print("QuadActuator Test")
    print("dynamic result0:", ac0.rotorRate)
    result1 = ac0.dynamic_actuator(ac0.rotorRate, np.array([0.2, 0.4, 0.6, 0.8]))
    print("dynamic result1:", result1)
    result2 = ac0.dynamic_actuator(np.array([400, 800, 1200, 1600]), np.array([0.2, 0.4, 0.6, 0.8]))
    print("dynamic result2:", result2)
    ac0.reset()
    ac0.step(np.array([0.2, 0.4, 0.6, 0.8]))
    print("dynamic result3:", ac0.rotorRate, ac0.outTorque, ac0.outThrust)
    print("QuadActuator Test Completed! ---------------------------------------------------------------")

