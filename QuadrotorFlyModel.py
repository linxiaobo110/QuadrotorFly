#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The file used to describe the dynamic of quadrotor UAV

By xiaobo
Contact linxiaobo110@gmail.com
Created on Fri Apr 19 10:40:44 2019
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
import MemoryStore

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

# definition of key constant
D2R = np.pi / 180
state_dim = 12
action_dim = 4
state_bound = np.array([10, 10, 10, 5, 5, 5, 80 * D2R, 80 * D2R, 180 * D2R, 100 * D2R, 100 * D2R, 100 * D2R])
action_bound = np.array([1, 1, 1, 1])


def rk4(func, x0, action, h):
    """Runge Kutta 4 order update function
    :param func: system dynamic
    :param x0: system state
    :param action: control input
    :param h: time of sample
    :return: state of next time
    """
    k1 = func(x0, action)
    k2 = func(x0 + h * k1 / 2, action)
    k3 = func(x0 + h * k2 / 2, action)
    k4 = func(x0 + h * k3, action)
    # print('rk4 debug: ', k1, k2, k3, k4)
    x1 = x0 + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x1


class StructureType(Enum):
    quad_x = enum.auto()
    quad_plus = enum.auto()


class QuadParas(object):
    """Define the parameters of quadrotor model

    """

    def __init__(self, g=9.81, rotor_num=4, tim_sample=0.01, structure_type=StructureType.quad_plus,
                 uav_l=0.450, uav_m=1.50, uav_ixx=1.75e-2, uav_iyy=1.75e-2, uav_izz=3.18e-2,
                 rotor_ct=1.11e-5, rotor_cm=1.49e-7, rotor_cr=646, rotor_wb=166, rotor_i=9.90e-5, rotor_t=1.36e-2):
        """init the quadrotor parameters
        These parameters are able to be estimation in web(https://flyeval.com/) if you do not have a real UAV.
        common parameters:
            -g          : N/kg,      acceleration gravity
            -rotor-num  : int,       number of rotors, e.g. 4, 6, 8
            -tim_sample : s,         sample time of system
            -structure_type:         quad_x, quad_plus
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
        self.structureType = structure_type
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
                 max_position=10, max_velocity=10, max_attitude=180, max_angular=200,
                 sysnoise_bound_pos=0, sysnoise_bound_att=0,
                 actuator_mode=ActuatorMode.simple):
        """ init the parameters for simulation process, focus on conditions during an episode
        :param init_mode:
        :param init_att:
        :param init_pos:
        :param sysnoise_bound_pos:
        :param sysnoise_bound_att:
        :param actuator_mode:
        """
        self.initMode = init_mode
        self.initAtt = init_att
        self.initPos = init_pos
        self.actuatorMode = actuator_mode
        self.sysNoisePos = sysnoise_bound_pos
        self.sysNoiseAtt = sysnoise_bound_att
        self.maxPosition = max_position
        self.maxVelocity = max_velocity
        self.maxAttitude = max_attitude * D2R
        self.maxAngular = max_angular * D2R


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

    def dynamic_actuator(self, rotor_rate, action):
        """dynamic of motor and propeller
        input: rotorRate, u
        output: rotorRateDot,
        """

        rate_dot = self.motorPara_scale * action + self.motorPara_bias - self.para.rotorTimScale * rotor_rate
        return rate_dot

    def reset(self):
        """reset all state"""

        self.outThrust = np.zeros([self.para.numOfRotors])
        self.outTorque = np.zeros([self.para.numOfRotors])
        # rate of rotor
        self.rotorRate = np.zeros([self.para.numOfRotors])

    def step(self, action: 'int > 0'):
        """calculate the next state based on current state and u
        :param action:
        :return:
        """
        action = np.clip(action, 0, 1)
        # if u > 1:
        #     u = 1

        if self.mode == ActuatorMode.simple:
            # without dynamic of motor
            self.rotorRate = self.para.rotorCr * action + self.para.rotorWb
        elif self.mode == ActuatorMode.dynamic:
            # with dynamic of motor
            self.rotorRate = rk4(self.dynamic_actuator, self.rotorRate, action, self.para.ts)
        else:
            self.rotorRate = 0

        self.outThrust = self.para.rotorCt * np.square(self.rotorRate)
        self.outTorque = self.para.rotorCm * np.square(self.rotorRate)
        return self.outThrust, self.outTorque


class QuadModel(object):
    """module interface, main class including basic dynamic of quad
    """

    def __init__(self, uav_para: QuadParas, sim_para: QuadSimOpt):
        """init a quadrotor
        :param uav_para:    parameters of quadrotor,maintain together
        :param sim_para:    'simple', without dynamic of motor; 'dynamic' with dynamic;
        """
        self.uavPara = uav_para
        self.simPara = sim_para
        self.actuator = QuadActuator(self.uavPara, sim_para.actuatorMode)

        # states of quadrotor
        #   -position, m
        self.position = np.array([0, 0, 0])
        #   -velocity, m/s
        self.velocity = np.array([0, 0, 0])
        #   -attitude, rad
        self.attitude = np.array([0, 0, 0])
        #   -angular, rad/s
        self.angular = np.array([0, 0, 0])

        # initial the states
        self.reset_states()

    def generate_init_att(self):
        """used to generate a init attitude according to simPara"""
        angle = self.simPara.initAtt * D2R
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
        """used to generate a init position according to simPara"""
        pos = self.simPara.initPos
        if self.simPara.initMode == SimInitType.rand:
            x = (1 * np.random.random() - 0.5) * pos[0]
            y = (1 * np.random.random() - 0.5) * pos[1]
            z = (1 * np.random.random() - 0.5) * pos[2]
        else:
            x = pos[0]
            y = pos[1]
            z = pos[2]
        return np.array([x, y, z])

    def reset_states(self, att='none', pos='none'):
        self.actuator.reset()
        if isinstance(att, str):
            self.attitude = self.generate_init_att()
        else:
            self.attitude = att

        if isinstance(pos, str):
            self.position = self.generate_init_pos()
        else:
            self.position = pos

        self.velocity = np.array([0, 0, 0])
        self.angular = np.array([0, 0, 0])

    def dynamic_basic(self, state, action):
        """ calculate /dot(state) = f(state) + u(state)
        This function will be executed many times during simulation, so high performance is necessary.
        :param state:
            0       1       2       3       4       5
            p_x     p_y     p_z     v_x     v_y     v_z
            6       7       8       9       10      11
            roll    pitch   yaw     v_roll  v_pitch v_yaw
        :param action: u1(sum of thrust), u2(torque for roll), u3(pitch), u4(yaw)
        :return: derivatives of state inclfrom bokeh.plotting import figure
        """
        # variable used repeatedly
        att_cos = np.cos(state[6:9])
        att_sin = np.sin(state[6:9])
        noise_pos = self.simPara.sysNoisePos * np.random.random(3)
        noise_att = self.simPara.sysNoiseAtt * np.random.random(3)

        dot_state = np.zeros([12])
        # dynamic of position cycle
        dot_state[0:3] = state[3:6]
        # we need not to calculate the whole rotation matrix because just care last column
        dot_state[3:6] = action[0] / self.uavPara.uavM * np.array([
            att_cos[2] * att_sin[1] * att_cos[0] + att_sin[2] * att_sin[0],
            att_sin[2] * att_sin[1] * att_cos[0] - att_cos[2] * att_sin[0],
            att_cos[0] * att_cos[1]
        ]) - np.array([0, 0, self.uavPara.g]) + noise_pos

        # dynamic of attitude cycle
        dot_state[6:9] = state[9:12]
        # Coriolis force on UAV from motor, this is affected by the direction of rotation.
        #   Pay attention, it needs to be modify when the model of uav varies.
        #   The signals of this equation should be same with toque for yaw
        rotor_rate_sum = (self.actuator.rotorRate[3] + self.actuator.rotorRate[2]
                          - self.actuator.rotorRate[1] - self.actuator.rotorRate[0])

        para = self.uavPara
        dot_state[9:12] = np.array([
            state[10] * state[11] * (para.uavInertia[1] - para.uavInertia[2]) / para.uavInertia[0]
            - para.rotorInertia / para.uavInertia[0] * state[10] * rotor_rate_sum
            + para.uavL * action[1] / para.uavInertia[0],
            state[9] * state[11] * (para.uavInertia[2] - para.uavInertia[0]) / para.uavInertia[1]
            + para.rotorInertia / para.uavInertia[1] * state[9] * rotor_rate_sum
            + para.uavL * action[2] / para.uavInertia[1],
            state[9] * state[10] * (para.uavInertia[0] - para.uavInertia[1]) / para.uavInertia[2]
            + action[3] / para.uavInertia[2]
        ]) + noise_att

        ''' Just used for test
        temp1 = state[10] * state[11] * (para.uavInertia[1] - para.uavInertia[2]) / para.uavInertia[0]
        temp2 = - para.rotorInertia / para.uavInertia[0] * state[10] * rotor_rate_sum
        temp3 = + para.uavL * action[1] / para.uavInertia[0]
        print('dyanmic Test', temp1, temp2, temp3, action)
       '''

        return dot_state

    def observe(self):
        """out put the system state, with sensor system or without sensor system"""
        return np.hstack([self.position, self.velocity, self.attitude, self.angular])

    def is_finished(self):
        if (np.max(np.abs(self.position)) < self.simPara.maxPosition)\
                and (np.max(np.abs(self.velocity) < self.simPara.maxVelocity))\
                and (np.max(np.abs(self.attitude) < self.simPara.maxAttitude))\
                and (np.max(np.abs(self.angular) < self.simPara.maxAngular)):
            return False
        else:
            return True

    def get_reward(self):
        reward = np.sum(np.square(self.position)) / 8 + np.sum(np.square(self.velocity)) / 20 \
                 + np.sum(np.square(self.attitude)) / 3 + np.sum(np.square(self.angular)) / 10
        return reward

    def rotor_distribute_dynamic(self, thrusts, torque):
        """ calculate torque according to the distribution of rotors
        :param thrusts:
        :param torque:
        :return:
        """
        ''' The structure of quadrotor, left is '+' and the right is 'x'
        The 'x' 'y' in middle defines the positive direction X Y axi in body-frame, which is a right hand frame.
        The numbers inside the rotors indicate the index of the motors;
        The signals show the direction of rotation, positive is ccw while the negative is cw.
        ---------------------------------------------------------------------------------------------------
                        ******                                                                            
                      **  3   **                                          ****                 ****     
                     **   -    **                                       **    **             **    **   
                      **      **                                      **   3    **         **    1   ** 
                       **    **                                       **   -    **         **    +   **  
                          **                                            **    **             **    **   
            ****          **          ****                                ****   **   **   **  ****     
         **      **     **  **      **    **            x(+)                       ***  ***            
        **   2    **  **      **  **   1    **       y(+)  y(-)                   **      **              
        **   +    **  **      **  **   +    **          x(-)                      **      **              
         **     **      ******      **    **                                      * **  ** *           
            ****          **          ****                                ****  **          ** ****    
                          **                                            **    **             **    **  
                        **  **                                        **   2    **         **    4   **
                      **      **                                      **   +    **         **    -   **
                     **   4    **                                       **    **             **    **  
                      **  -   **                                          ****                 ****    
                        ******     
        ---------------------------------------------------------------------------------------------------                                                                          
        '''
        forces = np.zeros(4)
        if self.uavPara.structureType == StructureType.quad_plus:
            forces[0] = np.sum(thrusts)
            forces[1] = thrusts[1] - thrusts[0]
            forces[2] = thrusts[3] - thrusts[2]
            forces[3] = torque[3] + torque[2] - torque[1] - torque[0]
        elif self.uavPara.structureType == StructureType.quad_x:
            forces[0] = np.sum(thrusts)
            forces[1] = -thrusts[0] + thrusts[1] + thrusts[2] - thrusts[3]
            forces[2] = -thrusts[0] + thrusts[1] - thrusts[2] + thrusts[3]
            forces[3] = -torque[0] - torque[1] + torque[2] + torque[3]
        else:
            forces = np.zeros(4)

        return forces

    def step(self, action: 'int > 0'):

        # 1.1 Actuator model, calculate the thrust and torque
        thrusts, toques = self.actuator.step(action)

        # 1.2 Calculate the force distribute according to 'x' type or '+' type, assum '+' type
        forces = self.rotor_distribute_dynamic(thrusts, toques)

        # 1.3 Basic model, calculate the basic model, the u need to be given directly in test-mode for Matlab
        state_temp = np.hstack([self.position, self.velocity, self.attitude, self.angular])
        state_next = rk4(self.dynamic_basic, state_temp, forces, self.uavPara.ts)
        [self.position, self.velocity, self.attitude, self.angular] = np.split(state_next, 4)

        # 2. Calculate Sensor sensor model
        ob = self.observe()

        # 3. Check whether finish (failed or completed)
        finish_flag = self.is_finished()

        # 4. Calculate a reference reward
        reward = self.get_reward()

        return ob, reward, finish_flag

    def get_controller_pid(self, state, ref_state):
        """ pid controller
        :param state: system state, 12
        :param ref_state: reference value for x, y, z, yaw
        :return: control value for four motors
        """

        # position-velocity cycle, velocity cycle is regard as kd
        kp_pos = np.array([0.2, 0.2, 0.8])
        kp_vel = np.array([0.2, 0.2, 0.5])
        err_pos = ref_state[0:3] - np.array([state[0], state[1], state[2]])
        ref_vel = err_pos * kp_pos
        err_vel = ref_vel - np.array([state[3], state[4], state[5]])
        ref_angle = kp_vel * err_vel

        # attitude-angular cycle, angular cycle is regard as kd
        kp_angle = np.array([0.5, 0.5, 0.3])
        kp_angular = np.array([0.2, 0.2, 0.2])
        # ref_angle = np.zeros(3)
        err_angle = np.array([-ref_angle[1], ref_angle[0], ref_state[3]]) - np.array([state[6], state[7], state[8]])
        ref_rate = err_angle * kp_angle
        err_rate = ref_rate - [state[9], state[10], state[11]]
        con_rate = err_rate * kp_angular

        # the control value in z direction needs to be modify considering gravity
        err_altitude = (ref_state[2] - state[2]) * 1
        con_altitude = (err_altitude - state[5]) * 1
        oil_altitude = 0.5 + con_altitude
        if oil_altitude > 0.75:
            oil_altitude = 0.75

        action_motor = np.zeros(4)
        if self.uavPara.structureType == StructureType.quad_plus:
            action_motor[0] = oil_altitude - con_rate[0] - con_rate[2]
            action_motor[1] = oil_altitude + con_rate[0] - con_rate[2]
            action_motor[2] = oil_altitude - con_rate[1] + con_rate[2]
            action_motor[3] = oil_altitude + con_rate[1] + con_rate[2]
        elif self.uavPara.structureType == StructureType.quad_x:
            action_motor[0] = oil_altitude - con_rate[2] - con_rate[1] - con_rate[0]
            action_motor[1] = oil_altitude - con_rate[2] + con_rate[1] + con_rate[0]
            action_motor[2] = oil_altitude + con_rate[2] - con_rate[1] + con_rate[0]
            action_motor[3] = oil_altitude + con_rate[2] + con_rate[1] - con_rate[0]
        else:
            action_motor = np.zeros(4)

        action_pid = action_motor
        return action_pid, oil_altitude


if __name__ == '__main__':
    " used for testing this module"
    testFlag = 3

    if testFlag == 1:
        # test for actuator
        qp = QuadParas()
        ac0 = QuadActuator(qp, ActuatorMode.simple)
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
    elif testFlag == 2:
        print("Basic model test: ")
        uavPara = QuadParas()
        simPara = QuadSimOpt(init_mode=SimInitType.fixed, actuator_mode=ActuatorMode.dynamic,
                             init_att=np.array([0.2, 0.2, 0.2]), init_pos=np.array([0, 0, 0]))
        quad1 = QuadModel(uavPara, simPara)
        u = np.array([100., 20., 20., 20.])
        stateTemp = np.array([1., 2., 3., 0.2, 0.3, 0.4, 0.3, 0.4, 0.5, 0.4, 0.5, 0.6])
        result1 = quad1.dynamic_basic(stateTemp, np.array([100., 20., 20., 20.]))
        print("result1 ", result1)
        [quad1.pos, quad1.velocity, quad1.attitude, quad1.angular] = np.split(stateTemp, 4)
        result2 = quad1.step(u)
        print("result2 ", result2, quad1.pos, quad1.velocity, quad1.attitude, quad1.angular)
    elif testFlag == 3:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        print("PID  controller test: ")
        uavPara = QuadParas(structure_type=StructureType.quad_x)
        simPara = QuadSimOpt(init_mode=SimInitType.fixed,
                             init_att=np.array([10., -10., 5]), init_pos=np.array([5, -5, 0]))
        quad1 = QuadModel(uavPara, simPara)
        record = MemoryStore.ReplayBuffer(10000, 1)
        record.clear()
        step_cnt = 0
        for i in range(1000):
            ref = np.array([0., 0., 1., 0.])
            stateTemp = quad1.observe()
            action2, oil = quad1.get_controller_pid(stateTemp, ref)
            print('action: ', action2)
            action2 = np.clip(action2, 0.1, 0.9)
            quad1.step(action2)
            record.buffer_append((stateTemp, action2))
            step_cnt = step_cnt + 1

        print('Quadrotor structure type', quad1.uavPara.structureType)
        # quad1.reset_states()
        print('Quadrotor get reward:', quad1.get_reward())
        bs = np.array([_[0] for _ in record.buffer])
        ba = np.array([_[1] for _ in record.buffer])
        t = range(0, record.count)
        mpl.style.use('seaborn')
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
