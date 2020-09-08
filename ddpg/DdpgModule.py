#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   DdpgModule.py
@Time    :   Thu Sep 03 2020 10:36:47
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
# %%
import tensorflow as tf
import numpy as np
import sys
sys.path.append('..')
import MemoryStore as MemoryStore

'''
********************************************************************************************************
**-------------------------------------------------------------------------------------------------------
**  Compiler   : python 3.6
**  Module Name: DdpgModule.py
**  Module Date: 2020-09-03
**  Module Auth: xiaobo
**  Version    : V0.1
**  Description: rewrite the ddpg with tensorflow 2.0
**-------------------------------------------------------------------------------------------------------
**  Reversion  :
**  Modified By:
**  Date       :
**  Content    :
**  Notes      :
********************************************************************************************************/
'''
# %%


def denormalize(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean


def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / (stats.std + 1e-8)


debug = 0


def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init


def mlp(name='test', para_hidden=[100], para_activation=[tf.keras.activations.tanh, None]):
    def network_fun(input_shape):
        # check the para
        if len(para_hidden) != (len(para_activation)-1):
            debug_print("num_activation should equal num_hidden + 1")
            return 
        debug_print('Create the bone of ' + name + ". input_shape is {}".format(input_shape))
        x_input = tf.keras.Input(shape=input_shape)
        h = x_input
        for i in range(len(para_hidden)):
            h = tf.keras.layers.Dense(units=para_hidden[i], 
                                      kernel_initializer=ortho_init(np.sqrt(2)),
                                      name=name + '_fc{}'.format(i),
                                      activation=para_activation[i]
                                      )(h)
        # y_out = tf.keras.layers.Dense(units=output_shape, 
        #                               #kernel_initializer=ortho),
        #                               name=name + '_output',
        #                               activation=para_activation[len(para_hidden)]
        # )(h)
        network = tf.keras.Model(inputs=x_input, outputs=h)
        return network
    return network_fun


def debug_print(s):
    if debug == 1:
        print(s)

# # %%
# # test code
# debug = 1
# m1 = mlp()(3, 1)
# x1 = np.array([3,4,5])
# m1.predict(np.expand_dims(x1, axis=0))


class Model(tf.keras.Model):
    def __init__(self, name, para_hidden, para_activation):
        super(Model,self).__init__(name=name)
        self.paraHidden = para_hidden
        self.paraActivation = para_activation

    def perturbable_vars(self):
        return [var for var in self.trainable_variables if 'layer_normalization' not in var.name]


class Actor(Model):
    """Actor network"""
    def __init__(self, dim_s=12, dim_a=4, name='actor', para_hidden=[100], para_activation=[tf.keras.activations.tanh, tf.keras.activations.tanh]):
        super().__init__(name=name, para_hidden=para_hidden, para_activation=para_activation)
        self.dim_s = dim_s
        self.dim_a = dim_a
        self.network_builder = mlp(para_hidden=para_hidden, para_activation=para_activation)(dim_s)
        self.output_layer = tf.keras.layers.Dense(units=dim_a,
                                                  kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
                                                  name=name + '_output',
                                                  activation=para_activation[len(para_hidden)]
                                                  )
        _ = self.output_layer(self.network_builder.outputs[0])

    @tf.function
    def call(self, obs):
        return self.output_layer(self.network_builder(obs))

# # %%
# # test code
# a1 = Actor()
# x1 = np.random.rand(12)
# x1_d = np.expand_dims(x1, axis=0)
# a1(x1_d)  


class Critic(Model):
    """Critic network"""
    def __init__(self, dim_s=12, dim_a=4, name='critic', para_hidden=[300], para_activation=[tf.keras.activations.relu, None]):
        super().__init__(name=name, para_hidden=para_hidden, para_activation=para_activation)
        self.dim_s = dim_s
        self.dim_a = dim_a
        # self.layer_norm
        self.network_builder = mlp(para_hidden=para_hidden, para_activation=para_activation)(dim_s + dim_a)
        self.output_layer = tf.keras.layers.Dense(units=1, 
                                      kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
                                      name=name + '_output',
                                      activation=para_activation[len(para_hidden)]
                                      )
        _ = self.output_layer(self.network_builder.outputs[0])

    @tf.function
    def call(self, obs, actions):
        x = tf.concat([obs, actions], axis=-1)
        
        return self.output_layer(self.network_builder(x)) 

# # %%
# # test code
# c1 = Critic()
# x1 = np.random.rand(12)
# x1 = np.array(x1, dtype=np.float32)
# a1 = np.random.rand(4)
# a1 = np.array(a1, dtype=np.float32)
# x1_d = np.expand_dims(x1, axis=0)
# a1_d = np.expand_dims(a1, axis=0)
# print(x1_d, a1_d)
# c1(x1_d, a1_d)  
# %%


class DdpgPara(object):
    def __init__(self, ddpg_gamma=0.9, ddpg_tau=0.001, weight_noise=None, action_noise=None, batch_size=32, replay_buffer_size=100000,
                 dim_s=12, dim_a=4, range_s=[-10, 10], range_a=[-1, 1], range_return=[-np.inf, np.inf],
                critic_hidden=[100], critic_activation=[tf.keras.activations.relu, None], critic_lr=1e-3,
                actor_hidden=[50], actor_activation=[tf.keras.activations.tanh, tf.keras.activations.tanh], actor_lr=1e-4):
        self.ddpgGamma = ddpg_gamma
        self.ddpgTau = ddpg_tau
        self.weightNoise = weight_noise
        self.actionNoise = action_noise
        self.batchSize = batch_size
        self.repalyBufferSize = replay_buffer_size
        self.dim_s = dim_s
        self.dim_a = dim_a
        self.range_s = range_s
        self.range_a = range_a
        self.range_return = range_return
        self.criticHidden = critic_hidden
        self.criticActivation = critic_activation
        self.criticLr = critic_lr
        self.actorHidden = actor_hidden
        self.actorActivation = actor_activation
        self.actorLr = actor_lr


class DdpgModle(object):
    
    def __init__(self, para:DdpgPara):
        self.para = para
        self.critic = Critic(dim_s=para.dim_s, 
                             dim_a=para.dim_a,
                             name='Critic',
                             para_hidden=para.criticHidden,
                             para_activation=para.criticActivation
                             )
        self.actor = Actor(dim_s=para.dim_s, 
                           dim_a=para.dim_a,
                           name='Actor',
                           para_hidden=para.actorHidden,
                           para_activation=para.actorActivation
                           )
        self.target_critic = Critic(dim_s=para.dim_s, 
                                    dim_a=para.dim_a,
                                    name='Critic_target',
                                    para_hidden=para.criticHidden,
                                    para_activation=para.criticActivation
                                    )
        self.target_actor = Actor(dim_s=para.dim_s, 
                                  dim_a=para.dim_a,
                                  name='Actor_target',
                                  para_hidden=para.actorHidden,
                                  para_activation=para.actorActivation
                                  )
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=para.criticLr)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=para.actorLr)

        # for var, target_var in zip(self.critic.variables, self.target_critic.variables):
        #     logger.info('  {} <- {}'.format(target_var.name, var.name))
        # logger.info('setting up actor target updates ...')
        # for var, target_var in zip(self.actor.variables, self.target_actor.variables):
        #     logger.info('  {} <- {}'.format(target_var.name, var.name))
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        self.replayBuffer = MemoryStore.ReplayBuffer(self.para.repalyBufferSize)

    def get_action(self, obs, noise=0., scale=1.):
        """
        Calculate the action according to the observation.
        :param obs: system state or output of sensors
        :param noise: noise applied for original action
        :param scale: scale applied for original action
        :return: action output
        """
        obs_norm = np.array(obs.reshape([-1, self.para.dim_s]), dtype=np.float32)
        actor_tf = self.actor(obs_norm)
        actor_tf_norm = np.clip(np.random.normal(actor_tf, noise), -1, 1) * scale
        bound = self.para.range_a[1] - self.para.range_a[0]
        acotr_out = actor_tf_norm * bound + self.para.range_a[0]
        return acotr_out

    def experience_append(self, ob_now, action, reward, ob_next):
        """
        add one/multi experience to the replay buffer
        :param ob_now: state or observation now
        :param action: action
        :param reward: reward
        :param ob_next: state or observation in next
        :return:
        """
        ob_now_norm = np.array(ob_now.reshape([-1, self.para.dim_s]), dtype=np.float32)
        action_norm = np.array(action.reshape([-1, 1]), dtype=np.float32)
        reward_norm = np.array(reward.reshape([-1, 1]), dtype=np.float32)
        ob_next_norm = np.array(ob_next.reshape([-1, self.para.dim_s]), dtype=np.float32)
        self.replayBuffer.buffer_append((ob_now_norm,
                                         action_norm,
                                         reward_norm,
                                         ob_next_norm
                                         ))

    @tf.function
    def update_target_nets(self):
        for var, target_var in zip(self.actor.variables, self.target_actor.variables):
            target_var.assign((1. - self.para.ddpgTau) * target_var + self.para.ddpgTau * var)
        for var, target_var in zip(self.critic.variables, self.target_critic.variables):
            target_var.assign((1. - self.para.ddpgTau) * target_var + self.para.ddpgTau * var)

    @tf.function
    def get_actor_grads(self, normalized_obs0):
        with tf.GradientTape() as tape:
            actor_tf = self.actor(normalized_obs0)
            normalized_critic_with_actor_tf = self.critic(normalized_obs0, actor_tf)
            # critic_with_actor_tf = denormalize(tf.clip_by_value(normalized_critic_with_actor_tf, self.para.return_range[0], self.para.return_range[1]), self.ret_rms)
            critic_with_actor_tf = tf.clip_by_value(normalized_critic_with_actor_tf, self.para.range_return[0], self.para.range_return[1])
            actor_loss = -tf.reduce_mean(critic_with_actor_tf)
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        # if self.clip_norm:
        #     actor_grads = [tf.clip_by_norm(grad, clip_norm=self.clip_norm) for grad in actor_grads]
        # if MPI is not None:
        #     actor_grads = tf.concat([tf.reshape(g, (-1,)) for g in actor_grads], axis=0)
        return actor_grads, actor_loss

    @tf.function
    def get_critic_grads(self, normalized_obs0, actions, target_Q):
        with tf.GradientTape() as tape:
            normalized_critic_tf = self.critic(normalized_obs0, actions)
            # normalized_critic_target_tf = tf.clip_by_value(normalize(target_Q, self.ret_rms), self.return_range[0], self.return_range[1])
            normalized_critic_target_tf = tf.clip_by_value(target_Q, self.para.range_return[0], self.para.range_return[1])
            critic_loss = tf.reduce_mean(tf.square(normalized_critic_tf - normalized_critic_target_tf))
            # The first is input layer, which is ignored here.
            # if self.critic_l2_reg > 0.:
            #     # Ignore the first input layer.
            #     for layer in self.critic.network_builder.layers[1:]:
            #         # The original l2_regularizer takes half of sum square.
            #         critic_loss += (self.critic_l2_reg / 2.)* tf.reduce_sum(tf.square(layer.kernel))
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        # if self.clip_norm:
        #     critic_grads = [tf.clip_by_norm(grad, clip_norm=self.clip_norm) for grad in critic_grads]
        # if MPI is not None:
        #     critic_grads = tf.concat([tf.reshape(g, (-1,)) for g in critic_grads], axis=0)
        return critic_grads, critic_loss

    @tf.function
    def compute_normalized_obs0_and_target_Q(self, obs0, obs1, rewards, terminals1):
        # normalized_obs0 = tf.clip_by_value(normalize(obs0, self.obs_rms), self.observation_range[0], self.observation_range[1])
        # normalized_obs1 = tf.clip_by_value(normalize(obs1, self.obs_rms), self.observation_range[0], self.observation_range[1])
        normalized_obs0 = tf.clip_by_value(normalize(obs0, None), self.para.range_s[0], self.para.range_s[1])
        normalized_obs1 = tf.clip_by_value(normalize(obs1, None), self.para.range_s[0], self.para.range_s[1])
        # Q_obs1 = denormalize(self.target_critic(normalized_obs1, self.target_actor(normalized_obs1)), self.ret_rms)
        Q_obs1 = self.target_critic(normalized_obs1, self.target_actor(normalized_obs1))
        # print('test1 ', normalized_obs0, normalized_obs1, Q_obs1)
        target_Q = rewards + (1. - terminals1) * self.para.ddpgGamma * Q_obs1
        # print('test1 ', normalized_obs0, normalized_obs1, Q_obs1)
        return normalized_obs0, target_Q

    def train(self):
        """
        train the actor and critic with inside replay buffer
        :return:
        """
        if self.replayBuffer.size() > self.para.batchSize:
            batch_buffer = self.replayBuffer.buffer_sample_batch(self.para.batchSize)
            bs = np.squeeze(np.array([_[0] for _ in batch_buffer]), axis=1)
            ba = np.squeeze(np.array([_[1] for _ in batch_buffer]), axis=1)
            br = np.squeeze(np.array([_[2] for _ in batch_buffer]), axis=1)
            bs_ = np.squeeze(np.array([_[3] for _ in batch_buffer]), axis=1)
            bt = np.zeros(32, dtype=np.float32)
            return self.train_batch(bs, ba, br, bs_, bt)
        else:
            return False

    def train_batch(self, bs, ba, br, bs_, b_t):
        """
        train the actor and critic with the provided train_data
        :param bs: a batch of state_now
        :param ba: a batch of action
        :param br: a batch of reward
        :param bs_: a batch of state_next
        :param b_t: a batch of done
        :return: critic_error and actor_loss
        """
        # batch = self.memory.sample(batch_size=self.batch_size)
        obs0, obs1 = tf.constant(bs), tf.constant(bs_)
        actions, rewards, terminals1 = tf.constant(ba), tf.constant(br), tf.constant(b_t, dtype=tf.float32)
        normalized_obs0, target_Q = self.compute_normalized_obs0_and_target_Q(obs0, obs1, rewards, terminals1)

        # if self.normalize_returns and self.enable_popart:
        #     old_mean = self.ret_rms.mean
        #     old_std = self.ret_rms.std
        #     self.ret_rms.update(target_Q.flatten())
        #     # renormalize Q outputs
        #     new_mean = self.ret_rms.mean
        #     new_std = self.ret_rms.std
        #     for vs in [self.critic.output_vars, self.target_critic.output_vars]:
        #         kernel, bias = vs
        #         kernel.assign(kernel * old_std / new_std)
        #         bias.assign((bias * old_std + old_mean - new_mean) / new_std)


        actor_grads, actor_loss = self.get_actor_grads(normalized_obs0)
        critic_grads, critic_loss = self.get_critic_grads(normalized_obs0, actions, target_Q)

        # if MPI is not None:
        #     self.actor_optimizer.apply_gradients(actor_grads, self.actor_lr)
        #     self.critic_optimizer.apply_gradients(critic_grads, self.critic_lr)
        # else:
        #     self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        #     self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        return critic_loss, actor_loss


# %%
if __name__ == "__main__":
    testFlag = 2
    if testFlag == 0:
        # test the functions
        # %%
        bs = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [0.21, 0.22, 0.23, 0.24, 0.25, 0.26] ], dtype=np.float32)
        bs_ = np.array([[0.31, 0.32, 0.33, 0.34, 0.35, 0.36],
                        [0.41, 0.42, 0.43, 0.44, 0.45, 0.46] ], dtype=np.float32)
        ba = np.array([[0.3, 0.4, 0.6], [0.1, 0.2, 0.3]], dtype=np.float32)
        br = np.array([0.5, 0.6], dtype=np.float32).reshape([2, 1])
        bt = np.array([0., 0.], dtype=np.float32).reshape([2,1])
        ddpgPara = DdpgPara(dim_s=6, dim_a=3)
        controler = DdpgModle(ddpgPara)
        obs, tq = controler.compute_normalized_obs0_and_target_Q(bs, bs_, br, bt)
        controler.get_actor_grads(bs)
        controler.get_critic_grads(bs, ba, tq)
        controler.train_batch(bs, ba, br, bs_, bt)
    # %%

    elif testFlag == 1:
        # %%
        # test on the carpole
        import sys
        import gym
        import time

        sys.path.append('..')
        import MemoryStore as MemoryStore

        # env = gym.make('CartPole-v0')
        mem_rb = MemoryStore.ReplayBuffer(100000)
        env = gym.make('Pendulum-v0')
        env = env.unwrapped
        env.seed(1)
        print('Action_bound', env.action_space.high, env.action_space.low)

        # controller
        ddpgPara = DdpgPara(dim_s=3, dim_a=1, critic_hidden=[200], actor_hidden=[200])
        controler = DdpgModle(ddpgPara)
        var = 1

        is_render = False
        score_last = -1000.
        for cnt_episode in range(200):
            state_now = env.reset()
            score = 0
            q_sum = tf.constant(0.)
            exist_cnt = 0
            tick_start = time.time()
            for _ in range(400):

                # a = env.action_space.sample()
                a2 = controler.get_action(state_now.reshape([1, 3]))
                a2 = np.clip(np.random.normal(a2, var), -1, 1)
                state_next, reward, flag, _ = env.step(a2 * 4)  # take a random action
                if flag == True:
                    print('finish')
                    break
                mem_rb.buffer_append((state_now.reshape([1, 3]),
                                      a2.reshape([1, 1]),
                                      reward.reshape([1, 1]) / 10.,
                                      state_next.reshape([1, 3])
                                      ))
                score += reward
                state_now = state_next
                # training
                if (mem_rb.size() > 10000):
                    batch_buffer = mem_rb.buffer_sample_batch(32)
                    bs = np.squeeze(np.array([_[0] for _ in batch_buffer]), axis=1)
                    ba = np.squeeze(np.array([_[1] for _ in batch_buffer]), axis=1)
                    br = np.squeeze(np.array([_[2] for _ in batch_buffer]), axis=1)
                    bs_ = np.squeeze(np.array([_[3] for _ in batch_buffer]), axis=1)
                    bs = np.array(bs, dtype=np.float32)
                    ba = np.array(ba, dtype=np.float32)
                    br = np.array(br, dtype=np.float32)
                    bs_ = np.array(bs_, dtype=np.float32)
                    bt = np.zeros(32, dtype=np.float32)

                    loss_critic, loss_actor = controler.train_batch(bs, ba, br, bs_, bt)
                    q_sum += loss_critic
                    controler.update_target_nets()
                    var = var * 0.995
                    if (var < 0.06):
                        var = 0.06
                exist_cnt += 1
                if is_render:
                    env.render()
            tick_end = time.time()
            # print(score, score_last)
            if (score[0] > -500) and (score_last > -500):
                is_render = True

            score_last = score[0]
            print('Episode ', cnt_episode, ': ',
                  'step_cnt', exist_cnt,
                  ', score', score[0],
                  ', q', q_sum.numpy() / exist_cnt,
                  ',time', (tick_end - tick_start),
                  ',render', is_render)
            mem_rb.episode_append(0)

        env.close()

    elif testFlag == 2:
        # %%
        # test on the carpole
        import sys
        import gym
        import time

        sys.path.append('..')
        #import QuadrotorFly.MemoryStore as MemoryStore

        # env = gym.make('CartPole-v0')
        env = gym.make('Pendulum-v0')
        env = env.unwrapped
        env.seed(1)
        print('Action_bound', env.action_space.high, env.action_space.low)

        # controller
        ddpgPara = DdpgPara(dim_s=3, dim_a=1, critic_hidden=[200], actor_hidden=[200], range_a=[-2, 2])
        controler = DdpgModle(ddpgPara)
        var = 1

        is_render = False
        score_last = -1000.
        for cnt_episode in range(200):
            state_now = env.reset()
            score = 0
            q_sum = tf.constant(0.)
            exist_cnt = 0
            tick_start = time.time()
            for _ in range(400):

                # a = env.action_space.sample()
                a2 = controler.get_action(state_now, var)
                # a2 = np.clip(np.random.normal(a2, var), -1, 1)
                state_next, reward, flag, _ = env.step(a2)  # take a random action
                if flag:
                    print('finish')
                    break
                controler.experience_append(state_now, a2, reward / 10, state_next)
                score += reward
                state_now = state_next
                # training
                if controler.replayBuffer.size() > 10000:
                    loss_critic, loss_actor = controler.train()
                    q_sum += loss_critic
                    controler.update_target_nets()
                    var = var * 0.995
                    if var < 0.06:
                        var = 0.06
                exist_cnt += 1
                if is_render:
                    env.render()
            tick_end = time.time()
            # print(score, score_last)
            if (score[0] > -500) and (score_last > -500):
                is_render = True

            score_last = score[0]
            print('Episode ', cnt_episode, ': ',
                  'step_cnt', exist_cnt,
                  ', score', score[0],
                  ', q', q_sum.numpy() / exist_cnt,
                  ',time', (tick_end - tick_start),
                  ',render', is_render)

        env.close()
