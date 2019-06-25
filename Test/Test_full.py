#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""a more completed test example for QuadrotorFly

By xiaobo
Contact linxiaobo110@gmail.com
Created on  五月 06 18:41 2019
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
import QuadrotorFlyGui as Qfg
import MemoryStore
import matplotlib.pyplot as plt


"""
********************************************************************************************************
**-------------------------------------------------------------------------------------------------------
**  Compiler   : python 3.6
**  Module Name: Test_full
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
# 角度到弧度的转换
D2R = Qfm.D2R
# 仿真参数设置
simPara = Qfm.QuadSimOpt(
        # 初值重置方式（随机或者固定）；姿态初值参数（随机就是上限，固定就是设定值）；位置初值参数（同上）
        init_mode=Qfm.SimInitType.rand, init_att=np.array([5., 5., 5.]), init_pos=np.array([1., 1., 1.]),
        # 仿真运行的最大位置，最大速度，最大姿态角（角度，不是弧度注意），最大角速度（角度每秒）
        max_position=8, max_velocity=8, max_attitude=180, max_angular=200,
        # 系统噪声，分别在位置环和速度环
        sysnoise_bound_pos=0, sysnoise_bound_att=0,
        #执行器模式，简单模式没有电机动力学，
        actuator_mode=Qfm.ActuatorMode.dynamic
        )
# 无人机参数设置，可以为不同的无人机设置不同的参数
uavPara = Qfm.QuadParas(
        # 重力加速度；采样时间；机型plus或者x
        g=9.8, tim_sample=0.01, structure_type=Qfm.StructureType.quad_plus,
        # 无人机臂长（米）；质量（千克）；飞机绕三个轴的转动惯量（千克·平方米）
        uav_l=0.45, uav_m=1.5, uav_ixx=1.75e-2, uav_iyy=1.75e-2, uav_izz=3.18e-2,
        # 螺旋桨推力系数（牛每平方（弧度每秒）），螺旋桨扭力系数（牛·米每平方（弧度每秒）），旋翼转动惯量，
        rotor_ct=1.11e-5, rotor_cm=1.49e-7, rotor_i=9.9e-5,
        # 电机转速比例参数（度每秒），电机转速偏置参数（度每秒），电机相应时间（秒）
        rotor_cr=646, rotor_wb=166, rotor_t=1.36e-2
        )
# 使用参数新建无人机
quad1 = Qfm.QuadModel(uavPara, simPara)

# 初始化GUI，并添加无人机
gui = Qfg.QuadrotorFlyGui([quad1])

# 初始化记录器，用于记录飞行数据
record = MemoryStore.DataRecord()

# 重置系统
# 当需要跑多次重复实验时注意重置系统，
quad1.reset_states()
record.clear()

# 仿真过程
print("Simplest simulation begin!")
for i in range(1000):
    # 模型更新
    # 设置目标
    ref = np.array([0., 0., 1., 0.])
    # 获取无人机的状态，共12维的向量
    #   分别是，xyz位置，xyz速度，roll pitch yaw姿态角，roll pitch yaw 角速度
    stateTemp = quad1.observe()
    # 通过无人机状态计算控制量，这也是自行设计控制器应修改的地方
    #    这是一个内置的控制器，返回值是给各个电机的控制量即油门的大小
    action2, oil = quad1.get_controller_pid(stateTemp, ref)
    # 使用控制量更新无人机，控制量是4维向量，范围0-1
    quad1.step(action2)
    
    # 记录数据
    record.buffer_append((stateTemp, action2))
    
    # 渲染GUI
    gui.render()

record.episode_append()
# 输出结果
## 获取记录中的状态
data = record.get_episode_buffer()
bs = data[0]
## 获取记录中的控制量
ba = data[1]
## 生成时间序列
t = range(0, record.count)
## 画图
fig1 = plt.figure(2)
plt.clf()
## 姿态图
plt.subplot(3, 1, 1)
plt.plot(t, bs[t, 6] / D2R, label='roll')
plt.plot(t, bs[t, 7] / D2R, label='pitch')
plt.plot(t, bs[t, 8] / D2R, label='yaw')
plt.ylabel('Attitude $(\circ)$', fontsize=15)
plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
## 位置图(xy)
plt.subplot(3, 1, 2)
plt.plot(t, bs[t, 0], label='x')
plt.plot(t, bs[t, 1], label='y')
plt.ylabel('Position (m)', fontsize=15)
plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
## 高度图
plt.subplot(3, 1, 3)
plt.plot(t, bs[t, 2], label='z')
plt.ylabel('Altitude (m)', fontsize=15)
plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
plt.show()
print("Simulation finish!")
