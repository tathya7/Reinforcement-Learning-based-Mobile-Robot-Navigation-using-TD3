#!/usr/bin/env python3

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter
from replay_buffer import ReplayBuffer
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



# To check for goal region
thresh_goal = 0.5
thresh_collide = 0.35
deltat = 0.3
odom_last = None
laser_dim = 20
velodyne_data = np.ones(laser_dim) * 10


# Checking if the goal is insice obstacle or not
def check_goal_pos(x,y):

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



# Creating actor class with deep neural network layers
class Actor(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 600)
        self.l2 = nn.Linear(600, 400)
        self.l3 = nn.Linear(400, 200)
        self.l4 = nn.Linear(200, action_dim)
        self.tanh = nn.Tanh()


    def forward(self, state):
        state = F.relu(self.l1(state))
        state = F.relu(self.l2(state))
        state = F.relu(self.l3(state))
        action = self.tanh(self.l4(state))
        return action
    


# Creating the critic network
class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Defining layers for first Q Network
        self.l1 = nn.Linear(state_dim, 600)
        self.l2s = nn.Linear(600, 400)
        self.l2a = nn.Linear(action_dim, 400)
        self.l3 = nn.Linear(400, 1)

        # Defining layers for second Q Network
        self.l4 = nn.Linear(state_dim, 600)
        self.l5s = nn.Linear(600, 400)
        self.l5a = nn.Linear(action_dim, 400)
        self.l6 = nn.Linear(400, 1)


    # Forward pass of the neural network
    def forward(self, state, action):

        # Getting the first state using first Q Network
        state1 = F.relu(self.l1(state))
        self.l2s(state1)
        self.l2a(action)
        state11 = torch.mm(state1, self.l2s.weight.data.t())
        state12 = torch.mm(action, self.l2a.weight.data.t())
        state1 = F.relu(state11 + state12 + self.l2a.bias.data)
        q1 = self.l3(state1)

        state2 = F.relu(self.l4(state))
        self.l5s(state2)
        self.l5a(action)
        state21 = torch.mm(state2, self.l5s.weight.data.t())
        state22 = torch.mm(action, self.l5a.weight.data.t())
        state2 = F.relu(state21 + state22 + self.l5a.bias.data)
        q2 = self.l6(state2)
        return q1, q2
    

# Model Training using Actor and Critic
class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):

        # Actor and Target Actor Network
        self.actor = Actor(state_dim, action_dim).to(device_check)
        self.actor_target = Actor(state_dim, action_dim).to(device_check)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        # Critic and Target Critic Network
        self.critic = Critic(state_dim, action_dim).to(device_check)
        self.critic_target = Critic(state_dim, action_dim).to(device_check)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action
        self.sum_writer = SummaryWriter(log_dir="./rlearning_ws/src/td3_pkg/scripts/runs")
        self.counter = 0

    # Generates a particular action using the actor netowrk
    def gen_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device_check)
        return self.actor(state).cpu().data.numpy().flatten()
    


    def train_cycle(self, replay_buffer, num_iter, batch_size=125, gamma=1, tau=0.008, policy_noise=0.3, policy_noise_clip=0.7, policy_freq=4):

        # Variable for average q-value
        avg_q = 0
        max_q = -inf
        avg_loss = 0
        for num in range(num_iter):

            # Sampling batch of past experiences from replay buffer
            (batch_states, batch_actions, batch_rewards, batch_check_prev, batch_next_states) = replay_buffer.sample_batch(batch_size)

            # Converting it to pytorch tensor
            state = torch.Tensor(batch_states).to(device_check)
            nxt_state = torch.Tensor(batch_next_states).to(device_check)
            action = torch.Tensor(batch_actions).to(device_check)
            reward = torch.Tensor(batch_rewards).to(device_check)
            prev_batch = torch.Tensor(batch_check_prev).to(device_check)

            # Computing the next action from actor network while adding noise
            nxt_action = self.actor_target(nxt_state)
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device_check)
            noise = noise.clamp(-policy_noise_clip, policy_noise_clip)
            nxt_action = (nxt_action + noise).clamp(-self.max_action, self.max_action)

            # Getting the Q Values for two critic networks
            targetq1, targetq2 = self.critic_target(nxt_state, nxt_action)

            # Choosing the minimum from both
            targetq = torch.min(targetq1, targetq2)

            # Taking the mean of the Q-Values
            avg_q += torch.mean(targetq)

            # Storing maximum q values
            max_q = max(max_q, torch.max(targetq))

            # Using bellman update rule
            targetq = reward + ((1 - prev_batch) * gamma * targetq).detach()

            # finding the online Q Values
            curr_q1, curr_q2 = self.critic(state, action)

            # Obtaining the loss of network
            mse_loss = F.mse_loss(curr_q1, targetq) + F.mse_loss(curr_q2, targetq)

            self.critic_optimizer.zero_grad()
            mse_loss.backward()
            self.critic_optimizer.step()
            
            # Updating the actor network after every frequency
            if num % policy_freq == 0:
                gradient_actor,_ = self.critic(state, self.actor(state))
                gradient_actor = -gradient_actor.mean()
                self.actor_optimizer.zero_grad()
                gradient_actor.backward()
                self.actor_optimizer.step()

                # Updating target networks
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

            avg_loss += mse_loss
        self.counter += 1
        env.get_logger().info(f"Current Results")
        env.get_logger().info(f"Q-Loss, Average Q, Maximum QQ, Iteration: {avg_loss / num_iter}, {avg_q / num_iter}, {max_q}, {self.counter}")

        # Logging the values for tensorboard 
        self.sum_writer.add_scalar("Network Loss", avg_loss / num_iter, self.counter)
        self.sum_writer.add_scalar("Average Q-Value", avg_q / num_iter, self.counter)
        self.sum_writer.add_scalar("Maximum Q-Value", max_q, self.counter)

    def save_data(self, fname, dir):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (dir, fname))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (dir, fname))

    def load_data(self, fname, dir):
        self.actor.load_state_dict(torch.load("%s/%s_actor.pth" % (dir, fname)))
        self.critic.load_state_dict(torch.load("%s/%s_critic.pth" % (dir, fname)))



class GazeboEnv(Node):

    def __init__(self):
        super().__init__('env')
        self.laser_dim = 20
        self.x_pos = 0
        self.y_pos = 0

        self.x_goal = 1.0
        self.y_goal = 0.0

        self.upper_thresh = 5.0
        self.lower_thresh = -5.0

        self.set_self_state = ModelState()
        self.set_self_state.model_name = "rl"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

        # Defining multiple publishers and clients 
        self.vel_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.set_state = self.create_publisher(ModelState, "gazebo/set_model_state", 10)
        self.goal_pub_mark = self.create_publisher(MarkerArray, "goal_point", 3)
        self.vel_pub_mark = self.create_publisher(MarkerArray, "linear_velocity", 1)
        self.ang_vel_mark = self.create_publisher(MarkerArray, "angular_velocity", 1)

        self.unpause = self.create_client(Empty, "/unpause_physics")
        self.pause = self.create_client(Empty, "/pause_physics")
        self.reset_proxy = self.create_client(Empty, "/reset_world")
        self.req = Empty.Request



    # Function to move a robot
    def move(self, action):
        global velodyne_data
        target_reached = False
        
        # Using Twist Topic to publish velocity getting from the models action generated
        vel = Twist()
        vel.linear.x = float(action[0])
        vel.angular.z = float(action[1])
        self.vel_pub.publish(vel)
        self.pub_marker_rviz(action)

        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for the client to be available')

        try:
            self.unpause.call_async(Empty.Request())
        except:
            print("Call Failed to resume the world")

        time.sleep(deltat)

        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for the client to be available')

        try:
            pass
            self.pause.call_async(Empty.Request())
        except (rclpy.ServiceException) as e:
            print("Service call Failed to resume the world")


        prev_batch, collision, laser_min_value = self.detect_collision(velodyne_data)
        velodyne_state = []
        velodyne_state[:] = velodyne_data[:]
        laser_state = [velodyne_state]

        # Getting robot position from Odometry Data
        self.x_pos = odom_last.pose.pose.position.x
        self.y_pos = odom_last.pose.pose.position.y
        quaternion = Quaternion(
            odom_last.pose.pose.orientation.w,
            odom_last.pose.pose.orientation.x,
            odom_last.pose.pose.orientation.y,
            odom_last.pose.pose.orientation.z,
        )
        
        euler_angles = quaternion.to_euler(degrees=False)
        angle = round(euler_angles[2], 4)

        # Calculate distance to the goal from the current robot position
        dist2goal = np.linalg.norm(
            [self.x_pos - self.x_goal, self.y_pos - self.y_goal]
        )

        # Defining the angle between the robots heading and heading toward the goal
        x_diff = self.x_goal - self.x_pos
        y_diff = self.y_goal - self.y_pos
        dot = x_diff * 1 + y_diff * 0
        abs1 = math.sqrt(math.pow(x_diff, 2) + math.pow(y_diff, 2))
        abs2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        alpha = math.acos(dot / (abs1 * abs2))

        if y_diff < 0:
            if x_diff < 0:
                alpha = -alpha
            else:
                alpha = 0 - alpha
        theta = alpha - angle

        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        # Goal Checking condition
        if dist2goal < thresh_goal:
            env.get_logger().info("Goal Reached, Moving to New Goal")
            target_reached = True
            prev_batch = True

        robot_state = [dist2goal, theta, action[0], action[1]]
        state = np.append(laser_state, robot_state)
        reward = self.allocate_reward(target_reached, collision, action, laser_min_value)
        return state, reward, prev_batch, target_reached
    

    # Function reset the robot position after a condition is met
    def reset_robot(self):

        while not self.reset_proxy.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Resetting the state of environment')

        try:
            self.reset_proxy.call_async(Empty.Request())
        except rclpy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        curr_robot_state = self.set_self_state

        x = 0
        y = 0
        correct_pos = False
        while not correct_pos:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            correct_pos = check_goal_pos(x, y)
        curr_robot_state.pose.position.x = x
        curr_robot_state.pose.position.y = y
        curr_robot_state.pose.orientation.x = quaternion.x
        curr_robot_state.pose.orientation.y = quaternion.y
        curr_robot_state.pose.orientation.z = quaternion.z
        curr_robot_state.pose.orientation.w = quaternion.w
        self.set_state.publish(curr_robot_state)

        self.x_pos = curr_robot_state.pose.position.x
        self.y_pos = curr_robot_state.pose.position.y

        self.update_goal()
        # randomly scatter boxes in the environment
        # self.random_box()
        # self.publish_markers([0.0, 0.0])

        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for the client to be available')

        try:
            self.unpause.call_async(Empty.Request())
        except:
            print("Call Failed to resume the world")

        time.sleep(deltat)

        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for the client to be available')

        try:
            pass
            self.pause.call_async(Empty.Request())
        except (rclpy.ServiceException) as e:
            print("Service call Failed to resume the world")    

        velodyne_state = []
        velodyne_state[:] = velodyne_data[:]
        laser_state = [velodyne_state]

        dist2goal = np.linalg.norm(
            [self.x_pos - self.x_goal, self.y_pos - self.y_goal]
        )

        x_diff = self.x_goal - self.x_pos
        y_diff = self.y_goal - self.y_pos

        dot = x_diff * 1 + y_diff * 0

        abs1 = math.sqrt(math.pow(x_diff, 2) + math.pow(y_diff, 2))
        abs2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        alpha = math.acos(dot / (abs1 * abs2))

        if y_diff < 0:
            if x_diff < 0:
                alpha = -alpha
            else:
                alpha = 0 - alpha
        theta = alpha - angle


        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        robot_state = [dist2goal, theta, 0.0, 0.0]
        state = np.append(laser_state, robot_state)
        return state
    



    # Function to update the new goal frequently
    def update_goal(self):

        if self.upper_thresh < 10:
            self.upper_thresh += 0.004
        if self.lower_thresh > -10:
            self.lower_thresh -= 0.004

        goal_check = False

        while not goal_check:
            self.x_goal = self.x_pos + random.uniform(self.upper_thresh, self.lower_thresh)
            self.y_goal = self.y_pos + random.uniform(self.upper_thresh, self.lower_thresh)
            goal_check = check_goal_pos(self.x_goal, self.y_goal)



    # Visualizing the RViz point cloud data
    def pub_marker_rviz(self, action):
        markerArray = MarkerArray()
        m = Marker()
        m.header.frame_id = "odom"
        m.type = m.CYLINDER
        m.action = m.ADD
        m.scale.x = 0.1
        m.scale.y = 0.1
        m.scale.z = 0.01
        m.color.a = 1.0
        m.color.r = 0.0
        m.color.g = 0.0
        m.color.b = 1.0
        m.pose.orientation.w = 1.0
        m.pose.position.x = self.x_goal
        m.pose.position.y = self.y_goal
        m.pose.position.z = 0.0

        markerArray.markers.append(m)

        self.goal_pub_mark.publish(markerArray)

        markerArray2 = MarkerArray()
        m2 = Marker()
        m2.header.frame_id = "odom"
        m2.type = m.CUBE
        m2.action = m.ADD
        m2.scale.x = float(abs(action[0]))
        m2.scale.y = 0.1
        m2.scale.z = 0.01
        m2.color.a = 1.0
        m2.color.r = 0.0
        m2.color.g = 1.0
        m2.color.b = 0.0
        m2.pose.orientation.w = 1.0
        m2.pose.position.x = 5.0
        m2.pose.position.y = 0.0
        m2.pose.position.z = 0.0

        markerArray2.markers.append(m2)
        self.vel_pub_mark.publish(markerArray2)

        markerArray3 = MarkerArray()
        m3 = Marker()
        m3.header.frame_id = "odom"
        m3.type = m.CUBE
        m3.action = m.ADD
        m3.scale.x = float(abs(action[1]))
        m3.scale.y = 0.1
        m3.scale.z = 0.01
        m3.color.a = 1.0
        m3.color.r = 0.0
        m3.color.g = 1.0
        m3.color.b = 0.0
        m3.pose.orientation.w = 1.0
        m3.pose.position.x = 5.0
        m3.pose.position.y = 0.2
        m3.pose.position.z = 0.0

        markerArray3.markers.append(m3)
        self.ang_vel_mark.publish(markerArray3)    

    # Checking the collision from lidar data
    @staticmethod
    def detect_collision(laser_data):

        laser_min_value = min(laser_data)
        if laser_min_value < thresh_collide:
            env.get_logger().info("Robot Collided!")
            return True, True, laser_min_value
        return False, False, laser_min_value

    # Defining reward based on task achieved
    @staticmethod
    def allocate_reward(target_reached, collision, action, laser_min_value):
        if target_reached:
            env.get_logger().info("Reward : 150")
            return 150.0
        elif collision:
            env.get_logger().info("Reward : -200")
            return -200.0
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            return action[0] / 2 - abs(action[1]) / 2 - r3(laser_min_value) / 2



class OdomSubs(Node):

    def __init__(self):
        super().__init__('Odom_subs')
        self.odom_subscription = self.create_subscription( Odometry, '/odom',self.odom_callback,10)
        self.odom_subscription

    def odom_callback(self, odom_data):
        global odom_last
        odom_last = odom_data



class VelodyneSubs(Node):

    def __init__(self):
        super().__init__('velodyne_subscriber')

        # Creating velodyne subscription
        self.velo_subscription = self.create_subscription(PointCloud2, "/velodyne_points", self.velodyne_callback, 10)
        self.velo_subscription

        # Defining angular gaps for lidar rays
        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / laser_dim]]
        for m in range(laser_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / laser_dim]
            )
        self.gaps[-1][-1] += 0.03

    def velodyne_callback(self, v):
        global velodyne_data

        # Reading the data from the lidar
        data_lst = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        velodyne_data = np.ones(laser_dim) * 10

        for i in range(len(data_lst)):

            # Checking if the point is above ground level
            if data_lst[i][2] > -0.2:

                # Calculating distance and angle of the point to the goal
                dot = data_lst[i][0] * 1 + data_lst[i][1] * 0
                abs1 = math.sqrt(math.pow(data_lst[i][0], 2) + math.pow(data_lst[i][1], 2))
                abs2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                alpha = math.acos(dot / (abs1 * abs2)) * np.sign(data_lst[i][1])
                dist2goal = math.sqrt(data_lst[i][0] ** 2 + data_lst[i][1] ** 2 + data_lst[i][2] ** 2)

                # Updating the laser data
                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= alpha < self.gaps[j][1]:
                        velodyne_data[j] = min(velodyne_data[j], dist2goal)
                        break



if __name__ == '__main__':

    rclpy.init(args=None)

    seed = 0  

    # Evaluation frequency to set the time limit of finding the path
    eval_freq = 5e3  

    # MAx steps per episode
    max_steps_per_ep = 300  

    # Total episodes
    num_eval_ep = 10 
    max_timesteps = 3e6 

    # Adding some exploration noise 
    exploration_noise = 0.8  
    exploration_decay_steps = (50000)
    exploration_noise_min = 0.1 

    # Mini-batch size to train from replay buffer
    batch_size = 40  

    # Discount Factor
    gamma = 0.99999  

    # Soft target update variable    
    tau = 0.005  


    policy_noise = 0.2 
    policy_noise_clip = 0.5 

    # Actor network update frequency
    policy_freq = 2  
    buffer_size = 1e6 

    # Filename to store the policy
    file_name = "td3_velodyne"  
    save_model = True  
    load_model = False  
    random_near_obstacle = True 

    # saving the trained models
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if save_model and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    # Laser dim takes 20 lidar readings and robot dim takes 2 angular and 2 linear velocities
    laser_dim = 20
    robot_dim = 4

    torch.manual_seed(seed)
    np.random.seed(seed)
    state_dim = laser_dim + robot_dim
    action_dim = 2
    max_action = 1

    #Creating the TD3 Networkand Replay Buffer
    network = TD3(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(buffer_size, seed)
    if load_model:
        try:
            network.load_data(file_name, "./pytorch_models")
        except:
            print("Could Not Load")

    # Evaluation number counter
    num_eval = []

    timestep = 0
    total_timestep = 0
    episode_counter = 0
    prev_batch = True
    epoch = 1

    random_action_counter = 0
    random_action = []

    # Initializing the Simulation controller, Odom Subscriber and Velodyne Subscriber
    env = GazeboEnv()
    odom_subscriber = OdomSubs()
    velodyne_subscriber = VelodyneSubs()
    
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(odom_subscriber)
    executor.add_node(velodyne_subscriber)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    
    rate = odom_subscriber.create_rate(1)

    try:
        while rclpy.ok():
            if timestep < max_timesteps:

                if prev_batch:
                    env.get_logger().info(f"Current Timestep : {timestep}")
                    if timestep != 0:
                        env.get_logger().info(f"Training")
                        network.train_cycle(
                        replay_buffer,
                        episode_timesteps,
                        batch_size,
                        gamma,
                        tau,
                        policy_noise,
                        policy_noise_clip,
                        policy_freq,
                        )

                    if total_timestep >= eval_freq:
                        env.get_logger().info("Validating")
                        total_timestep %= eval_freq
                        num_eval.append(
                            metrics(network=network, epoch=epoch, episodes=num_eval_ep)
                        )

                        network.save(file_name, directory="./pytorch_models")
                        np.save("./results/%s" % (file_name), num_eval)
                        epoch += 1

                    state = env.reset_robot()
                    prev_batch = False

                    episode_reward = 0
                    episode_timesteps = 0
                    episode_counter += 1

                # Adding noise to the actions generated to make the algorithm more robust
                if exploration_noise > exploration_noise_min:
                    exploration_noise = exploration_noise - ((1 - exploration_noise_min) / exploration_decay_steps)

                action = network.gen_action(np.array(state))
                action = (action + np.random.normal(0, exploration_noise, size=action_dim)).clip(
                     -max_action, max_action
                )



                # Giving the linear and angular velocity
                a_input = [(action[0] + 1) / 2, action[1]]
                nxt_state, reward, prev_batch, target = env.move(a_input)
                prev_batch_bool = 0 if episode_timesteps + 1 == max_steps_per_ep else int(prev_batch)
                prev_batch = 1 if episode_timesteps + 1 == max_steps_per_ep else int(prev_batch)
                episode_reward += reward

                # Adding the tuple of state, action and reward to the replay buffer to store experiences
                replay_buffer.add(state, action, reward, prev_batch_bool, nxt_state)

                state = nxt_state
                episode_timesteps += 1
                timestep += 1
                total_timestep += 1

    except KeyboardInterrupt:
        pass

    rclpy.shutdown()