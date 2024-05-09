#!/usr/bin/env python3

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayBuffer

import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
import threading

import math
import random

import point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


device_check = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Defining the variables for collision detection and goal reached distance
thresh_goal = 0.5
thresh_collide = 0.35
deltat = 0.3
odom_last = None
laser_dim = 20
velodyne_data = np.ones(laser_dim) * 10


def check_goal_pos(x,y):
    # Check if the random goal position is located on an obstacle and do not accept it if it is
    goal_check = False
    
    if x > -0.55 and 1.7 > y > -1.7:
        goal_check = True

    if 4.0 < x < 5.0 and 0 < y < 1.0:
        goal_check = True
    

    return goal_check



def metrics(network, epoch, episodes=5):

    avg_reward = 0.0
    col = 0
    for _ in range(episodes):
        env.get_logger().info(f"Current Episode {_}")
        count = 0
        state = env.reset_robot()
        prev_batch = False
        while not prev_batch and count < 301:
            action = network.gen_action(np.array(state))
            env.get_logger().info(f"Current Action : {action}")
            a_input = [(action[0] + 1) / 2, action[1]]
            state, reward, prev_batch, _ = env.move(a_input)
            avg_reward += reward
            count += 1
            if reward < -90:
                col += 1
    avg_reward /= episodes
    avg_col = col / episodes
    env.get_logger().info("########################")
    env.get_logger().info("Evaluation Episodes: {}, Epoch: {}, Average Reward: {}".format(episodes, epoch, avg_reward))
    env.get_logger().info("########################")
    return avg_reward



class NN(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(NN, self).__init__()
        self.layer1 = nn.Linear(state_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, action_dim)
        self.tanh = nn.Tanh()

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, state):
        state = F.relu(self.layer1(state))
        state = F.relu(self.layer2(state))
        action = self.tanh(self.layer3(state))
        return action
    


class DQNAgent(object):
    def __init__(self, state_dim, action_dim, max_action, batch_size=64, lrate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, replay_buffer_size=10000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.batch_size = batch_size
        self.lr = lrate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.lr = lrate
        self.sum_writer = SummaryWriter(log_dir="./rlearning_ws/src/dqn_pkg/scripts/runs")

        self.counter=0


        self.policy_net = NN(state_dim, action_dim).to(device_check)
        self.target_net = NN(state_dim, action_dim).to(device_check)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.mse_loss = nn.MSELoss()


    def gen_action(self, state):

        if np.random.rand() <= self.epsilon:
            return np.random.uniform(-self.max_action, self.max_action, size=self.action_dim)

        state = torch.Tensor(state.reshape(1, -1)).to(device_check)
        with torch.no_grad():
            q_values = self.policy_net(state)
            print(q_values)
        return q_values.cpu().data.numpy().flatten()
    

    def train_replay(self):

        avg_loss = 0
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        state, action, reward, nxt_state, prev_batch = zip(batch)

        state = torch.Tensor(state).to(device_check)
        action = torch.Tensor(action).to(device_check)
        reward = torch.Tensor(reward).to(device_check)
        nxt_state = torch.Tensor(nxt_state).to(device_check)
        prev_batch = torch.Tensor(prev_batch).to(device_check)


        q_val = self.policy_net(state)
        q_val = torch.gather(q_val, 1, action.unsqueeze(1).long())

        nxt_q_val = self.target_net(nxt_state).detach()
        nxt_q_val = torch.max(nxt_q_val,1)[0]
        tar_q_val = reward + (1-prev_batch) * self.gamma * nxt_q_val.unsqueeze(1)

        q_loss = self.mse_loss(q_val, tar_q_val.unsqueeze(1))

        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        avg_loss += q_loss


        env.get_logger().info(f"Current Results")
        env.get_logger().info(f"Q-Loss, Iteration: {avg_loss / num_iter}, {self.counter}")
        self.sum_writer.add_scalar("Network Loss", avg_loss / num_iter, self.counter)




