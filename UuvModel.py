#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   UuvModel.py
@Time    :   Mon Aug 17 2020 10:15:02
@Author  :   xiaobo
@Contact :   linxiaobo110@gmail.com
'''

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

# here put the import lib
import numpy as np
import MemoryStore
'''
********************************************************************************************************
**-------------------------------------------------------------------------------------------------------
**  Compiler   : python 3.6
**  Module Name: UuvModel.py
**  Module Date: 2020-08-17
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
'''

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
    # print(test)
    # print('rk4 debug: ', trans(k1), trans(k2), trans(k3), trans(k4))
    x1 = x0 + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x1

def trans(xx):
    s2 = np.zeros(12)
    s2[0:3] = xx[3:6]
    s2[3:6] = xx[9:12]
    s2[6:9] = xx[6:9]
    s2[9:12] = xx[0:3]
    return s2


class UuvParas(object):
    """Define the parameters of quadrotor model

    """

    def __init__(self, g=9.8, tim_sample=0.01,
                uuv_b=14671, uuv_m=1840, uuv_dis_c=np.array([0.02, -0.005, 0.008]), uuv_j=np.array([289.3, 6771.8, 6771.8]), uuv_lu=1019.2, uuv_s=0.224, uuv_l=7.738,
                uuv_cx=0.141, uuv_mx_beta=0.00152, uuv_mx_dr=-0.000319, uuv_mx_dd=-0.0812, uuv_mx_wx=-0.0044, uuv_mx_wy=0.0008, uuv_mx_xp=0,
                uuv_cy_alfa=2.32, uuv_cy_de=0.51, uuv_cy_wz=1.17, uuv_my_beta=0.69, uuv_my_dr=-0.11, uuv_my_wx=0, uuv_my_wy=-0.61,
                uuv_cz_beta=-2.32, uuv_cz_dr=-0.2, uuv_cz_wy=-1.17, uuv_mz_alpha=0.69, uuv_mz_de=-0.28, uuv_mz_wz=-0.61,
                uuv_add_m_k11=0.0222, uuv_add_m_k22=1.1096, uuv_add_m_K44=0.1406, uuv_add_m_k55=3.8129, uuv_add_m_k26=-0.363):
        """init the quadrotor parameters
        These parameters are able to be estimation in web(https://flyeval.com/) if you do not have a real UAV.
        common parameters:
            -g          : N/kg,      acceleration gravity
            -tim_sample : s,         sample time of system
        uuv:
            -uuv_b      : N,        floating force of uuv
            -uuv_m      : kg,       the mass of quadrotor
            -uuv_dis_c  : m,        distance from center of mass to center of floting
            -uuv_j      : kg.m^2    the central principal moments of inertia of UAV in x,y,z
            -uuv_lu     : kg/(m^3)  the density of water
            -uuv_s      : m^2       the max area of the uuv in direction x 
            -uuv_l      : m         the length of the uuv
            -uuv_cx     :           the factor between drag-force and uuv_s
            -uuv_mx_beta:           the factor between roll-force and beta
            -uuv_mx_dr:             the factor between roll-force and delta_r
            -uuv_mx_dd:             the factor between roll-force and delta_d
            -uuv_mx_wx:             the factor between roll-force and omega_x
            -uuv_mx_wy:             the factor between roll-force and omega_y
            -uuv_mx_xp:             
            -uuv_cy_alfa:           the factor between up-force and alpha
            -uuv_cy_de:             the factor between up-force and delta_e
            -uuv_cy_wz:             the factor between up-force and omega_z
            -uuv_my_beta:           the factor between yaw-force and beta
            -uuv_my_dr:             the factor between yaw-force and delta_r
            -uuv_my_wx:            
            -uuv_my_wy:             the factor between yaw-force and omega_y
            -uuv_cz_beta:           the factor between right-force and beta
            -uuv_cz_dr:             the factor between right-force and delta_r
            -uuv_cz_wy:             the factor between right-force and omega_y
            -uuv_mz_alpha:          the factor between pitch-force and alpha
            -uuv_mz_de:             the factor between pitch-force and delta_e
            -uuv_mz_wz:             the factor between pitch-force and omega_z
            -uuv_add_m_k11:         the factor between added mass in x
            -uuv_add_m_k22:         the factor between added mass in z
            -uuv_add_m_K44:         the factor between added mass in roll
            -uuv_add_m_k55:         the factor between added mass in pitch
            -uuv_add_m_k26:         the factor between added mass in 
        """
        self.ts = tim_sample
        self.g = g

        self.B = uuv_b
        self.G = uuv_m * g # Gravity of uuv
        self.M = uuv_m
        self.Dis_c = uuv_dis_c
        self.J = uuv_j
        self.Lu = uuv_lu
        self.S = uuv_s
        self.L = uuv_l
        self.Cx = uuv_cx
        self.MxBeta = uuv_mx_beta
        self.MxDr = uuv_mx_dr
        self.MxDd = uuv_mx_dd
        self.MxWx = uuv_mx_wx
        self.MxWy = uuv_mx_wy
        self.MxP = 0
        self.CyAlpha = uuv_cy_alfa
        self.CyDe = uuv_cy_de
        self.CyWz = uuv_cy_wz
        self.MyBeta = uuv_my_beta
        self.MyDr = uuv_my_dr
        self.MyWy = uuv_my_wy
        self.CzBeta = uuv_cz_beta
        self.CzDr = uuv_cz_dr
        self.CzWy = uuv_cz_wy
        self.MzAlpha = uuv_mz_alpha
        self.MzDe = uuv_mz_de
        self.MzWz = uuv_mz_wz
        self.AddMk11 = uuv_add_m_k11
        self.AddMk22 = uuv_add_m_k22
        self.AddMk33 = uuv_add_m_k22
        self.AddMk44 = uuv_add_m_K44
        self.AddMk55 = uuv_add_m_k55
        self.AddMk66 = uuv_add_m_k55
        self.AddMk26 = uuv_add_m_k26
        self.AddMk35 = -uuv_add_m_k26

        V = self.B / (self.Lu * self.g)
        self.AddML11 = self.AddMk11 * self.Lu * V
        self.AddML22 = self.AddMk22 * self.Lu * V
        self.AddML33 = self.AddMk33 * self.Lu * V
        self.AddML44 = self.AddMk44 * self.Lu * np.power(V, 5./3)
        self.AddML55 = self.AddMk55 * self.Lu * np.power(V, 5./3)
        self.AddML66 = self.AddMk66 * self.Lu * np.power(V, 5./3)
        self.AddML26 = self.AddMk26 * self.Lu * np.power(V, 4./3)
        self.AddML35 = self.AddMk35 * self.Lu * np.power(V, 4./3)

class UuvModel(object):
    def __init__(self, uuv_para: UuvParas):
        """init a uuv"""

        self.agtPara = uuv_para # the para of agent
        # states of uuv
        #   -position, m
        self.position = np.array([0, 0, 0])
        #   -velocity, m/s
        self.velocity = np.array([0, 0, 0])
        #   -attitude, rad
        self.attitude = np.array([0, 0, 0])
        #   -angular, rad/s
        self.angular = np.array([0, 0, 0])
        # accelerate, m/(s^2)
        self.acc = np.zeros(3)
        # time control, s
        self.__ts = 0

    def dynamic_basic(self, state, action, debug=False):
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
        dot_state = np.zeros([12])
        if debug:
            test = dict()
        p = self.agtPara
        a = np.array([
            [p.M + p.AddML11,   0,                              0,                              0,                      p.M * p.Dis_c[2],               -p.M * p.Dis_c[1]],
            [0,                 p.M + p.AddML22,                0,                              -p.M * p.Dis_c[2],      0,                              p.M * p.Dis_c[0] + p.AddML26],
            [0,                 0,                              p.M + p.AddML33,                p.M * p.Dis_c[1],       p.AddML35 - p.M * p.Dis_c[0],   0],
            [0,                 -p.M * p.Dis_c[2],              p.M * p.Dis_c[1],               p.J[0] + p.AddML44,     0,                              0],
            [p.M * p.Dis_c[2],  0,                              p.AddML35 - p.M * p.Dis_c[0],   0,                      p.J[1] + p.AddML55,             0],
            [-p.M * p.Dis_c[1], p.M * p.Dis_c[0] + p.AddML26,   0,                              0,                      0,                              p.J[2] + p.AddML66]   
        ])
        de = action[0]
        dr = action[1]
        dd = action[2]
        tl = action[3]
        # the sequence here is different from the book "yu lei hang xing xue"
        vx = state[3]
        vy = state[4]
        vz = state[5] 
        wx = state[9]
        wy = state[10]
        wz = state[11]
        pesi = state[8]
        sita = state[7]
        fai = state[6]
        v2 = vx * vx + vy * vy + vz * vz
        # speed
        v = np.sqrt(v2)
        # attack angle
        alfa = np.arctan(-vy / vx)
        vxy = np.sqrt(vx * vx + vy * vy)
        # side angle
        beta = np.arctan(vz / vxy)

        salfa = np.sin(alfa)
        calfa = np.cos(alfa)
        sbeta = np.sin(beta)
        cbeta = np.cos(beta)
        spesi = np.sin(pesi)
        cpesi = np.cos(pesi)
        ssita = np.sin(sita)
        csita = np.cos(sita)
        sfai = np.sin(fai)
        cfai = np.cos(fai)

        cvb = np.array([
            [calfa * cbeta,     salfa,         -calfa * sbeta],
            [-salfa * cbeta,    calfa,         salfa * sbeta],
            [sbeta,             0,             cbeta]
        ])

        ceb = np.array([
            [csita * cpesi,                         ssita,              -csita * spesi],
            [-ssita * cpesi * cfai + spesi * sfai,  csita * cfai,       ssita * spesi * cfai + cpesi * sfai],
            [ssita * cpesi * sfai + spesi * cfai,   -csita * sfai,      -ssita * spesi * sfai + cpesi * cfai]
        ])
        if debug:
            test['cvb'] = cvb
            test['ceb'] = ceb
            # print('dynamic test 1: -----------------------')
            # print('cvb is ', cvb)
            # print('ceb is ', ceb)
        fs_v = np.array([-p.Cx * 0.5 * p.Lu * v2 * p.S,
                         p.CyAlpha * 0.5 * p.Lu * v2 * p.S * alfa + p.CyWz * 0.5 * p.Lu * p.S * p.L * v * wz + p.CyDe * 0.5 * p.Lu * v2 * p.S * de,
                         p.CzBeta * 0.5 * p.Lu * v2 * p.S * beta + p.CzWy * 0.5 * p.Lu * p.S * p.L * v * wy + p.CzDr * 0.5 * p.Lu * v2 * p.S * dr
        ])
        # force
        fs = cvb.dot(fs_v)

        # torque
        ms = np.array([
            p.MxBeta * 0.5 * p.Lu * p.S * p.L * v2 * beta + p.MxWx * 0.5 * p.Lu * p.S * (np.square(p.L)) * wx * v
             + p.MxDr * 0.5 * p.Lu * p.S * p.L * v2 * dr + p.MxDd  * 0.5 * p.Lu * p.S * p.L * v2 * dd + p.MxP * v2,
            p.MyBeta * 0.5 * p.Lu * p.S * p.L * v2 * beta + p.MyWy * 0.5 * p.Lu * p.S * (np.square(p.L)) * wy * v + p.MyDr * 0.5 * p.Lu * p.S * p.L * v2 * dr,
            p.MzAlpha * 0.5 * p.Lu * p.S * p.L * v2 * alfa + p.MzWz * 0.5 * p.Lu * p.S * (np.square(p.L)) * wz * v + p.MzDe * 0.5 * p.Lu * p.S * p.L * v2 * de
        ])
        # floating force
        fg = ceb.dot(np.array([0, p.B - p.G, 0]))
        if debug:
            test['fs'] = fs
            test['ms'] = ms
            test['fg'] = fg
        
        mg_m = np.array([ 
            [0,             -p.Dis_c[2],    p.Dis_c[1]],
            [p.Dis_c[2],    0,              -p.Dis_c[0]],
            [-p.Dis_c[1],   p.Dis_c[0],     0]
        ])
        # gravity
        mg = mg_m.dot(ceb.dot(np.array([0, -p.G, 0])))
        ft = np.array([
            [-p.M * (vz * wy - vy * wz + p.Dis_c[1] * wx * wy + p.Dis_c[2] * wx * wz - p.Dis_c[0] * (np.square(wy) + np.square(wz)))],
            [-p.M * (vx * wz - vz * wx + p.Dis_c[2] * wy * wz + p.Dis_c[0] * wy * wx - p.Dis_c[1] * (np.square(wz) + np.square(wx)))],
            [-p.M * (vy * wx - vx * wy + p.Dis_c[0] * wz * wx + p.Dis_c[1] * wz * wy - p.Dis_c[2] * (np.square(wx) + np.square(wy)))]
        ])
        mt = np.array([
            [-p.M * (p.Dis_c[1] * (vy * wx - vx * wy) + p.Dis_c[2] * (vz * wx - vx * wz)) - (p.J[2] - p.J[1]) * wy * wz],
            [-p.M * (p.Dis_c[2] * (vz * wy - vy * wz) + p.Dis_c[0] * (vx * wy - vy * wx)) - (p.J[0] - p.J[2]) * wz * wx],
            [-p.M * (p.Dis_c[0] * (vx * wz - vz * wx) + p.Dis_c[1] * (vy * wz - vz * wy)) - (p.J[1] - p.J[0]) * wx * wy]
        ])
        if debug:
            test['mg'] = mg
            test['ft'] = ft
            test['mt'] = mt
            # print('dynamic test 3: -----------------------')
            # print('mg is ', mg)
            # print('ft is ', ft)
            # print('mt is ', mt)

        ftl = np.array([tl, 0, 0])
        f = np.hstack([fs.reshape([3]) + fg.reshape([3]) + ft.reshape([3]) + ftl.reshape([3]), ms.reshape([3]) + mg.reshape([3]) + mt.reshape([3])])
        df_speed = np.linalg.inv(a).dot(f)
        angle_rot = np.array([
            [0,     cfai / csita,   -sfai / csita],
            [0,     sfai,           cfai],
            [1,     -cfai * np.tan(sita),   sfai * np.tan(sita)]
        ])
        df_2 = angle_rot.dot(np.array([wx, wy, wz]))
        df_3 = ceb.dot([vx, vy, vz])
        if debug:
            test['f'] = f
            test['a'] = a
            test['df_speed'] = df_speed
            test['df_angle'] = df_2
        
        dot_state[0:3] = df_3
        dot_state[3:6] = df_speed[0:3]
        dot_state[6:9] = df_2
        dot_state[9:12] = df_speed[3:6]
        if debug:
            return dot_state, test
        else:
            return dot_state

    def observe(self):
        """out put the system state, with sensor system or without sensor system"""
        # if self.simPara.enableSensorSys:
        #     sensor_data = dict()
        #     for index, sensor in enumerate(self.sensorList):
        #         if isinstance(sensor, SensorBase.SensorBase):
        #             # name = str(index) + '-' + sensor.get_name()
        #             name = sensor.get_name()
        #             sensor_data.update({name: sensor.observe()})
        #     return sensor_data
        # else:
        return np.hstack([self.position, self.velocity, self.attitude, self.angular])

    def get_reward(self):
        reward = np.sum(np.square(self.position)) / 8 + np.sum(np.square(self.velocity)) / 20 \
                 + np.sum(np.square(self.attitude)) / 3 + np.sum(np.square(self.angular)) / 10
        return reward
    
    def step(self, action):

        self.__ts += self.agtPara.ts
        # # 1.1 Actuator model, calculate the thrust and torque
        # thrusts, toques = self.actuator.step(action)

        # # 1.2 Calculate the force distribute according to 'x' type or '+' type, assum '+' type
        # forces = self.rotor_distribute_dynamic(thrusts, toques)

        # 1.3 Basic model, calculate the basic model, the u need to be given directly in test-mode for Matlab
        state_temp = np.hstack([self.position, self.velocity, self.attitude, self.angular])
        state_next = rk4(self.dynamic_basic, state_temp, action, self.agtPara.ts)
        [self.position, self.velocity, self.attitude, self.angular] = np.split(state_next, 4)
        # calculate the accelerate
        state_dot = self.dynamic_basic(state_temp, action)
        self.acc = state_dot[3:6]

        # 2. Calculate Sensor sensor model
        # if self.simPara.enableSensorSys:
        #     for index, sensor in enumerate(self.sensorList):
        #         if isinstance(sensor, SensorBase.SensorBase):
        #             sensor.update(np.hstack([state_next, self.acc]), self.__ts)
        ob = self.observe()

        # 3. Check whether finish (failed or completed)
        finish_flag = False#self.is_finished()

        # 4. Calculate a reference reward
        reward = self.get_reward()

        return ob, reward, finish_flag

if __name__ == '__main__':
    " used for testing this module"
    testFlag = 3

    if testFlag == 1:
        # test case 1
        # 使用简单的原始数据测试
        para = UuvParas()
        agent = UuvModel(para)
        xi = np.array([0, 0, 0, 10, 0, 0, 
                        0, 0, 0, 0, 0, 0])
        a0 = np.array([ 0, 0, 0, 10672])
        dot_state, test = agent.dynamic_basic(xi, a0 ,debug=True)
        ## sec 1  ##############################################################
        r_cvb = np.diag(np.ones(3))
        r_ceb = np.diag(np.ones(3))
        ## sec 2  ##############################################################
        r_fs = np.array([-1.6095,         0,         0]) * 1000
        r_ms = np.array([0, 0, 0])
        r_fg = np.array([0,       -3361,          0])
        ## sec 3  ##############################################################
        r_mg = np.array([144.2560,       0, -360.6400])
        r_ft = np.array([0, 0, 0])
        r_mt = np.array([0, 0, 0])
        ## sec 4  ##############################################################
        r_f = 1.0e+04 * np.array([1.2282, -0.3361, 0, 0.0144, 0, -0.0361])
        r_a = 1.0e+04 * np.array([
            [0.1873,         0,         0,         0,    0.0015,    0.0009],
            [     0,    0.3501,         0,   -0.0015,         0,   -0.0581],
            [     0,         0,    0.3501,   -0.0009,    0.0581,         0],
            [     0,   -0.0015,   -0.0009,    0.0561,         0,         0],
            [0.0015,         0,    0.0581,         0,    1.4148,         0],
            [0.0009,   -0.0581,         0,         0,         0,    1.4148]
        ]),
        r_df_speed = np.array([6.5567,   -0.9706,    0.0018,    0.2316,   -0.0069,   -0.0696])
        r_df_angle = np.array([0, 0, 0])
        print('sec 1: ------------------------------------------------------------')
        print('cvb is ', test['cvb'], '. should be ', r_cvb)
        print('ceb is ', test['ceb'], '. should be ', r_ceb)
        print('sec 2: ------------------------------------------------------------')
        print('fs is ', test['fs'], '. should be ', r_fs)
        print('ms is ', test['ms'], '. should be ', r_ms)
        print('fg is ', test['fg'], '. should be ', r_fg)
        print('sec 3: ------------------------------------------------------------')
        print('mg is ', test['mg'], '. should be ', r_mg)
        print('ft is ', test['ft'], '. should be ', r_ft)
        print('mt is ', test['mt'], '. should be ', r_mt)
        print('sec 4 -------------------------------------------------------------')
        print('f is ', test['f'], '. should be ', r_f)
        print('a is ', test['a'], '. should be ', r_a)
        print('df_speed is ', test['df_speed'], '. should be ', r_df_speed)
        print('df_angle is ', test['df_angle'], '. should be ', r_df_angle)
        print('result is ', trans(dot_state))

    elif testFlag == 2:
        # import matplotlib.pyplot as plt
        para = UuvParas()
        agent = UuvModel(para)
        s2 = np.zeros(12)
        agent.velocity[0] = 10
        xi = np.array([0, 0, 0, 10, 0, 0, 
                        0, 0, 0, 0, 0, 0])
        a0 = np.array([ 0, 0, 0, 10672])
        # record = MemoryStore.DataRecord()
        # record.clear()
        step_cnt = 0
        action2 = np.zeros(4)
        action2[3] = 10672
        print('action: ', action2)
        agent.step(action2)
        s1 = agent.observe()
        s2[0:3] = s1[3:6]
        s2[3:6] = s1[9:12]
        s2[6:9] = s1[6:9]
        s2[9:12] = s1[3:6]
        print('1', s2)
    elif testFlag == 3:
        import matplotlib.pyplot as plt
        para = UuvParas()
        agent = UuvModel(para)
        agent.velocity[0] = 10
        xi = np.array([0, 0, 0, 10, 0, 0, 
                        0, 0, 0, 0, 0, 0])
        a0 = np.array([ 0, 0, 0, 10672])
        record = MemoryStore.DataRecord()
        record.clear()
        step_cnt = 0
        for i in range(2000):
            ref = np.array([0., 0., 1., 0.])
            stateTemp = agent.observe()
            
            print(stateTemp)
            action2 = np.zeros(4)
            action2[3] = 10672
            # action2, oil = quad1.get_controller_pid(stateTemp, ref)
            print('action: ', action2)
            # action2 = np.clip(action2, 0.1, 0.9)
            agent.step(action2)
            record.buffer_append((stateTemp, action2))
            step_cnt = step_cnt + 1
        record.episode_append()


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
        plt.plot(t, bs[t, 4], label='y')
        plt.plot(t, bs[t, 5], label='z')
        plt.ylabel('Position (m)', fontsize=15)
        plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
        plt.subplot(3, 1, 3)
        plt.plot(t, bs[t, 3], label='x')
        plt.ylabel('Forward (m)', fontsize=15)
        plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
        plt.show()