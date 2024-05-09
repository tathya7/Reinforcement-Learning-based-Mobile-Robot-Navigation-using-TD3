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

        self.l1 = nn.Linear(state_dim, 800)
        self.l2 = nn.Linear(800, 600)
        self.l3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()


    def forward(self, state):
        state = F.relu(self.l1(state))
        state = F.relu(self.l2(state))
        action = self.tanh(self.l3(state))
        return action
    



class TD3(object):
    def __init__(self, state_dim, action_dim):
        
        self.actor = Actor(state_dim, action_dim).to(device_check)

    def gen_action(self, state):
        # Generate an action from the trained model
        state = torch.Tensor(state.reshape(1, -1)).to(device_check)
        return self.actor(state).cpu().data.numpy().flatten()

    def load_data(self, fname, dir):
        self.actor.load_state_dict(torch.load("%s/%s_actor.pth" % (dir, fname)))

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

        # Set up the ROS publishers and subscribers
        self.vel_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.set_state = self.create_publisher(ModelState, "gazebo/set_model_state", 10)
        self.goal_pub_mark = self.create_publisher(MarkerArray, "goal_point", 3)
        self.vel_pub_mark = self.create_publisher(MarkerArray, "linear_velocity", 1)
        self.ang_vel_mark = self.create_publisher(MarkerArray, "angular_velocity", 1)

        self.unpause = self.create_client(Empty, "/unpause_physics")
        self.pause = self.create_client(Empty, "/pause_physics")
        self.reset_proxy = self.create_client(Empty, "/reset_world")
        self.req = Empty.Request

    def move(self, action):
        global velodyne_data
        target_reached = False
        
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

        if dist2goal < thresh_goal:
            env.get_logger().info("Goal Reached, Moving to New Goal")
            target_reached = True
            prev_batch = True

        robot_state = [dist2goal, theta, action[0], action[1]]
        state = np.append(laser_state, robot_state)
        reward = self.allocate_reward(target_reached, collision, action, laser_min_value)
        return state, reward, prev_batch, target_reached

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

        x = 10
        y = 10
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

        self.random_box()
        self.pub_marker_rviz([0.0, 0.0])

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

    def update_goal(self):
        if self.upper_thresh < 10:
            self.upper_thresh += 0.003
        if self.lower_thresh > -10:
            self.lower_thresh -= 0.003

        goal_check = False

        while not goal_check:
            self.x_goal = self.x_pos + random.uniform(self.upper_thresh, self.lower_thresh)
            self.y_goal = self.y_pos + random.uniform(self.upper_thresh, self.lower_thresh)
            goal_check = check_goal_pos(self.x_goal, self.y_goal)
    

    def random_box(self):
            for i in range(4):
                name = "cardboard_box_" + str(i)

                x = 0
                y = 0
                box_ok = False
                while not box_ok:
                    x = np.random.uniform(-6, 6)
                    y = np.random.uniform(-6, 6)
                    box_ok = check_goal_pos(x, y)
                    distance_to_robot = np.linalg.norm([x - self.x_pos, y - self.y_pos])
                    distance_to_goal = np.linalg.norm([x - self.x_goal, y - self.y_goal])
                    if distance_to_robot < 1.5 or distance_to_goal < 1.5:
                        box_ok = False
                box_state = ModelState()
                box_state.model_name = name
                box_state.pose.position.x = x
                box_state.pose.position.y = y
                box_state.pose.position.z = 0.0
                box_state.pose.orientation.x = 0.0
                box_state.pose.orientation.y = 0.0
                box_state.pose.orientation.z = 0.0
                box_state.pose.orientation.w = 1.0
                self.set_state.publish(box_state)

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
        m.color.b = 0.8
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


    @staticmethod
    def detect_collision(laser_data):
        # Detect a collision from laser data
        laser_min_value = min(laser_data)
        if laser_min_value < thresh_collide:
            env.get_logger().info("Robot Collided!")
            return True, True, laser_min_value
        return False, False, laser_min_value

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
        self.odom_subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.odom_subscription

    def odom_callback(self, odom_data):
        global odom_last
        odom_last = odom_data



class VelodyneSubs(Node):

    def __init__(self):
        super().__init__('velodyne_subscriber')
        self.velo_subscription = self.create_subscription(PointCloud2, "/velodyne_points", self.velodyne_callback, 10)
        self.velo_subscription

        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / laser_dim]]
        for m in range(laser_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / laser_dim]
            )
        self.gaps[-1][-1] += 0.03

    def velodyne_callback(self, v):
        global velodyne_data
        data_lst = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        velodyne_data = np.ones(laser_dim) * 10

        for i in range(len(data_lst)):

            if data_lst[i][2] > -0.2:
                
                dot = data_lst[i][0] * 1 + data_lst[i][1] * 0
                abs1 = math.sqrt(math.pow(data_lst[i][0], 2) + math.pow(data_lst[i][1], 2))
                abs2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                alpha = math.acos(dot / (abs1 * abs2)) * np.sign(data_lst[i][1])
                dist2goal = math.sqrt(data_lst[i][0] ** 2 + data_lst[i][1] ** 2 + data_lst[i][2] ** 2)

                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= alpha < self.gaps[j][1]:
                        velodyne_data[j] = min(velodyne_data[j], dist2goal)
                        break

if __name__ == '__main__':

    rclpy.init(args=None)

    seed = 0  
    max_steps_per_ep = 300  
    laser_dim = 20
    robot_dim = 4
    file_name = "td3_velodyne" 

    torch.manual_seed(seed)
    np.random.seed(seed)
    state_dim = laser_dim + robot_dim
    action_dim = 2

    network = TD3(state_dim, action_dim)
    try:
        network.load_data(file_name, "/root/rlearning_ws/src/td3_pkg/scripts/pytorch_models/")
    except:
        raise ValueError("Could not load the stored model parameters")

    prev_batch = True
    episode_timesteps = 0

    env = GazeboEnv()
    odom_subscriber = OdomSubs()
    velodyne_subscriber = VelodyneSubs()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(odom_subscriber)
    executor.add_node(velodyne_subscriber)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    
    rate = odom_subscriber.create_rate(2)

    while rclpy.ok():

        if prev_batch:
            state = env.reset_robot()

            action = network.gen_action(np.array(state))
            a_input = [(action[0] + 1) / 2, action[1]]
            nxt_state, reward, prev_batch, target = env.move(a_input)
            prev_batch = 1 if episode_timesteps + 1 == max_steps_per_ep else int(prev_batch)

            prev_batch = False
            episode_timesteps = 0
        else:
            action = network.gen_action(np.array(state))
            a_input = [(action[0] + 1) / 2, action[1]]
            nxt_state, reward, prev_batch, target = env.move(a_input)
            prev_batch = 1 if episode_timesteps + 1 == max_steps_per_ep else int(prev_batch)

            state = nxt_state
            episode_timesteps += 1

    rclpy.shutdown()

# import time
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import rclpy
# from rclpy.node import Node
# import threading

# import math
# import random

# import point_cloud2 as pc2
# from gazebo_msgs.msg import ModelState
# from geometry_msgs.msg import Twist
# from nav_msgs.msg import Odometry
# from sensor_msgs.msg import PointCloud2
# from squaternion import Quaternion
# from std_srvs.srv import Empty
# from visualization_msgs.msg import Marker
# from visualization_msgs.msg import MarkerArray

# GOAL_REACHED_DIST = 0.3
# COLLISION_DIST = 0.35
# TIME_DELTA = 0.2

# last_odom = None
# environment_dim = 20
# velodyne_data = np.ones(environment_dim) * 10

# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(Actor, self).__init__()

#         self.layer_1 = nn.Linear(state_dim, 800)
#         self.layer_2 = nn.Linear(800, 600)
#         self.layer_3 = nn.Linear(600, action_dim)
#         self.tanh = nn.Tanh()

#     def forward(self, s):
#         s = F.relu(self.layer_1(s))
#         s = F.relu(self.layer_2(s))
#         a = self.tanh(self.layer_3(s))
#         return a

# # td3 network
# class td3(object):
#     def __init__(self, state_dim, action_dim):
#         # Initialize the Actor network
#         self.actor = Actor(state_dim, action_dim).to(device)

#     def get_action(self, state):
#         # Function to get the action from the actor
#         state = torch.Tensor(state.reshape(1, -1)).to(device)
#         return self.actor(state).cpu().data.numpy().flatten()

#     def load(self, filename, directory):
#         # Function to load network parameters
#         self.actor.load_state_dict(
#             torch.load("%s/%s_actor.pth" % (directory, filename))
#         )

# # Check if the random goal position is located on an obstacle and do not accept it if it is
# def check_pos(x, y):
#     goal_ok = False
    
#     if x > -0.55 and 1.7 > y > -1.7:
#         goal_ok = True

#     return goal_ok

# class GazeboEnv(Node):
#     """Superclass for all Gazebo environments."""

#     def __init__(self):
#         super().__init__('env')

#         self.environment_dim = 20
#         self.odom_x = 0
#         self.odom_y = 0

#         self.goal_x = 1
#         self.goal_y = 0.0

#         self.upper = 5.0
#         self.lower = -5.0

#         self.set_self_state = ModelState()
#         self.set_self_state.model_name = "r1"
#         self.set_self_state.pose.position.x = 0.0
#         self.set_self_state.pose.position.y = 0.0
#         self.set_self_state.pose.position.z = 0.0
#         self.set_self_state.pose.orientation.x = 0.0
#         self.set_self_state.pose.orientation.y = 0.0
#         self.set_self_state.pose.orientation.z = 0.0
#         self.set_self_state.pose.orientation.w = 1.0

#         # Set up the ROS publishers and subscribers
#         self.vel_pub = self.create_publisher(Twist, "/cmd_vel", 1)
#         self.set_state = self.create_publisher(ModelState, "gazebo/set_model_state", 10)

#         self.unpause = self.create_client(Empty, "/unpause_physics")
#         self.pause = self.create_client(Empty, "/pause_physics")
#         self.reset_proxy = self.create_client(Empty, "/reset_world")
#         self.req = Empty.Request

#         self.publisher = self.create_publisher(MarkerArray, "goal_point", 3)
#         self.publisher2 = self.create_publisher(MarkerArray, "linear_velocity", 1)
#         self.publisher3 = self.create_publisher(MarkerArray, "angular_velocity", 1)

#     # Perform an action and read a new state
#     def step(self, action):
#         global velodyne_data
#         target = False
        
#         # Publish the robot action
#         vel_cmd = Twist()
#         vel_cmd.linear.x = float(action[0])
#         vel_cmd.angular.z = float(action[1])
#         self.vel_pub.publish(vel_cmd)
#         self.publish_markers(action)

#         #rospy.wait_for_service("/gazebo/unpause_physics")
#         while not self.unpause.wait_for_service(timeout_sec=1.0):
#             self.get_logger().info('service not available, waiting again...')

#         try:
#             self.unpause.call_async(Empty.Request())
#         except:
#             print("/unpause_physics service call failed")

#         # propagate state for TIME_DELTA seconds
#         time.sleep(TIME_DELTA)

#         while not self.pause.wait_for_service(timeout_sec=1.0):
#             self.get_logger().info('service not available, waiting again...')

#         try:
#             pass
#             self.pause.call_async(Empty.Request())
#         except (rclpy.ServiceException) as e:
#             print("/gazebo/pause_physics service call failed")

#         # read velodyne laser state
#         done, collision, min_laser = self.observe_collision(velodyne_data)
#         v_state = []
#         v_state[:] = velodyne_data[:]
#         laser_state = [v_state]

#         # Calculate robot heading from odometry data
#         self.odom_x = last_odom.pose.pose.position.x
#         self.odom_y = last_odom.pose.pose.position.y
#         quaternion = Quaternion(
#             last_odom.pose.pose.orientation.w,
#             last_odom.pose.pose.orientation.x,
#             last_odom.pose.pose.orientation.y,
#             last_odom.pose.pose.orientation.z,
#         )
#         euler = quaternion.to_euler(degrees=False)
#         angle = round(euler[2], 4)

#         # Calculate distance to the goal from the robot
#         distance = np.linalg.norm(
#             [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
#         )

#         # Calculate the relative angle between the robots heading and heading toward the goal
#         skew_x = self.goal_x - self.odom_x
#         skew_y = self.goal_y - self.odom_y
#         dot = skew_x * 1 + skew_y * 0
#         mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
#         mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
#         beta = math.acos(dot / (mag1 * mag2))
#         if skew_y < 0:
#             if skew_x < 0:
#                 beta = -beta
#             else:
#                 beta = 0 - beta
#         theta = beta - angle
#         if theta > np.pi:
#             theta = np.pi - theta
#             theta = -np.pi - theta
#         if theta < -np.pi:
#             theta = -np.pi - theta
#             theta = np.pi - theta

#         # Detect if the goal has been reached and give a large positive reward
#         if distance < GOAL_REACHED_DIST:
#             env.get_logger().info("GOAL is reached!")
#             target = True
#             done = True

#         robot_state = [distance, theta, action[0], action[1]]
#         state = np.append(laser_state, robot_state)
#         reward = self.get_reward(target, collision, action, min_laser)

#         return state, reward, done, target

#     def reset(self):

#         # Resets the state of the environment and returns an initial observation.
#         while not self.reset_proxy.wait_for_service(timeout_sec=1.0):
#             self.get_logger().info('reset : service not available, waiting again...')

#         try:
#             self.reset_proxy.call_async(Empty.Request())
#         except rclpy.ServiceException as e:
#             print("/gazebo/reset_simulation service call failed")

#         angle = np.random.uniform(-np.pi, np.pi)
#         quaternion = Quaternion.from_euler(0.0, 0.0, angle)
#         object_state = self.set_self_state

#         x = 10
#         y = 10
#         position_ok = False
#         while not position_ok:
#             x = np.random.uniform(-4.5, 4.5)
#             y = np.random.uniform(-4.5, 4.5)
#             position_ok = check_pos(x, y)
#         object_state.pose.position.x = x
#         object_state.pose.position.y = y
#         # object_state.pose.position.z = 0.
#         object_state.pose.orientation.x = quaternion.x
#         object_state.pose.orientation.y = quaternion.y
#         object_state.pose.orientation.z = quaternion.z
#         object_state.pose.orientation.w = quaternion.w
#         self.set_state.publish(object_state)

#         self.odom_x = object_state.pose.position.x
#         self.odom_y = object_state.pose.position.y

#         # set a random goal in empty space in environment
#         self.change_goal()
#         # randomly scatter boxes in the environment
#         self.random_box()
#         self.publish_markers([0.0, 0.0])

#         while not self.unpause.wait_for_service(timeout_sec=1.0):
#             self.node.get_logger().info('service not available, waiting again...')

#         try:
#             self.unpause.call_async(Empty.Request())
#         except:
#             print("/gazebo/unpause_physics service call failed")

#         time.sleep(TIME_DELTA)

#         while not self.pause.wait_for_service(timeout_sec=1.0):
#             self.node.get_logger().info('service not available, waiting again...')

#         try:
#             self.pause.call_async(Empty.Request())
#         except:
#             print("/gazebo/pause_physics service call failed")

#         v_state = []
#         v_state[:] = velodyne_data[:]
#         laser_state = [v_state]

#         distance = np.linalg.norm(
#             [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
#         )

#         skew_x = self.goal_x - self.odom_x
#         skew_y = self.goal_y - self.odom_y

#         dot = skew_x * 1 + skew_y * 0
#         mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
#         mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
#         beta = math.acos(dot / (mag1 * mag2))

#         if skew_y < 0:
#             if skew_x < 0:
#                 beta = -beta
#             else:
#                 beta = 0 - beta
#         theta = beta - angle

#         if theta > np.pi:
#             theta = np.pi - theta
#             theta = -np.pi - theta
#         if theta < -np.pi:
#             theta = -np.pi - theta
#             theta = np.pi - theta

#         robot_state = [distance, theta, 0.0, 0.0]
#         state = np.append(laser_state, robot_state)
#         return state

#     def change_goal(self):
#         # Place a new goal and check if its location is not on one of the obstacles
#         if self.upper < 10:
#             self.upper += 0.004
#         if self.lower > -10:
#             self.lower -= 0.004

#         goal_ok = False

#         while not goal_ok:
#             self.goal_x = self.odom_x + random.uniform(self.upper, self.lower)
#             self.goal_y = self.odom_y + random.uniform(self.upper, self.lower)
#             goal_ok = check_pos(self.goal_x, self.goal_y)
    

#     def random_box(self):
#         # Randomly change the location of the boxes in the environment on each reset to randomize the training
#         # environment
#         for i in range(4):
#             name = "cardboard_box_" + str(i)

#             x = 0
#             y = 0
#             box_ok = False
#             while not box_ok:
#                 x = np.random.uniform(-6, 6)
#                 y = np.random.uniform(-6, 6)
#                 box_ok = check_pos(x, y)
#                 distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
#                 distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
#                 if distance_to_robot < 1.5 or distance_to_goal < 1.5:
#                     box_ok = False
#             box_state = ModelState()
#             box_state.model_name = name
#             box_state.pose.position.x = x
#             box_state.pose.position.y = y
#             box_state.pose.position.z = 0.0
#             box_state.pose.orientation.x = 0.0
#             box_state.pose.orientation.y = 0.0
#             box_state.pose.orientation.z = 0.0
#             box_state.pose.orientation.w = 1.0
#             self.set_state.publish(box_state)

#     def publish_markers(self, action):
#         # Publish visual data in Rviz
#         markerArray = MarkerArray()
#         marker = Marker()
#         marker.header.frame_id = "odom"
#         marker.type = marker.CYLINDER
#         marker.action = marker.ADD
#         marker.scale.x = 0.1
#         marker.scale.y = 0.1
#         marker.scale.z = 0.01
#         marker.color.a = 1.0
#         marker.color.r = 0.0
#         marker.color.g = 1.0
#         marker.color.b = 0.0
#         marker.pose.orientation.w = 1.0
#         marker.pose.position.x = self.goal_x
#         marker.pose.position.y = self.goal_y
#         marker.pose.position.z = 0.0

#         markerArray.markers.append(marker)

#         self.publisher.publish(markerArray)

#         markerArray2 = MarkerArray()
#         marker2 = Marker()
#         marker2.header.frame_id = "odom"
#         marker2.type = marker.CUBE
#         marker2.action = marker.ADD
#         marker2.scale.x = float(abs(action[0]))
#         marker2.scale.y = 0.1
#         marker2.scale.z = 0.01
#         marker2.color.a = 1.0
#         marker2.color.r = 1.0
#         marker2.color.g = 0.0
#         marker2.color.b = 0.0
#         marker2.pose.orientation.w = 1.0
#         marker2.pose.position.x = 5.0
#         marker2.pose.position.y = 0.0
#         marker2.pose.position.z = 0.0

#         markerArray2.markers.append(marker2)
#         self.publisher2.publish(markerArray2)

#         markerArray3 = MarkerArray()
#         marker3 = Marker()
#         marker3.header.frame_id = "odom"
#         marker3.type = marker.CUBE
#         marker3.action = marker.ADD
#         marker3.scale.x = float(abs(action[1]))
#         marker3.scale.y = 0.1
#         marker3.scale.z = 0.01
#         marker3.color.a = 1.0
#         marker3.color.r = 1.0
#         marker3.color.g = 0.0
#         marker3.color.b = 0.0
#         marker3.pose.orientation.w = 1.0
#         marker3.pose.position.x = 5.0
#         marker3.pose.position.y = 0.2
#         marker3.pose.position.z = 0.0

#         markerArray3.markers.append(marker3)
#         self.publisher3.publish(markerArray3)

#     @staticmethod
#     def observe_collision(laser_data):
#         # Detect a collision from laser data
#         min_laser = min(laser_data)
#         if min_laser < COLLISION_DIST:
#             env.get_logger().info("Collision is detected!")
#             return True, True, min_laser
#         return False, False, min_laser

#     @staticmethod
#     def get_reward(target, collision, action, min_laser):
#         if target:
#             env.get_logger().info("reward 100")
#             return 100.0
#         elif collision:
#             env.get_logger().info("reward -100")
#             return -100.0
#         else:
#             r3 = lambda x: 1 - x if x < 1 else 0.0
#             return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2

# class Odom_subscriber(Node):

#     def __init__(self):
#         super().__init__('odom_subscriber')
#         self.subscription = self.create_subscription(
#             Odometry,
#             '/odom',
#             self.odom_callback,
#             10)
#         self.subscription

#     def odom_callback(self, od_data):
#         global last_odom
#         last_odom = od_data

# class Velodyne_subscriber(Node):

#     def __init__(self):
#         super().__init__('velodyne_subscriber')
#         self.subscription = self.create_subscription(
#             PointCloud2,
#             "/velodyne_points",
#             self.velodyne_callback,
#             10)
#         self.subscription

#         self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / environment_dim]]
#         for m in range(environment_dim - 1):
#             self.gaps.append(
#                 [self.gaps[m][1], self.gaps[m][1] + np.pi / environment_dim]
#             )
#         self.gaps[-1][-1] += 0.03

#     def velodyne_callback(self, v):
#         global velodyne_data
#         data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
#         velodyne_data = np.ones(environment_dim) * 10
#         for i in range(len(data)):
#             if data[i][2] > -0.2:
#                 dot = data[i][0] * 1 + data[i][1] * 0
#                 mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
#                 mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
#                 beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
#                 dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)

#                 for j in range(len(self.gaps)):
#                     if self.gaps[j][0] <= beta < self.gaps[j][1]:
#                         velodyne_data[j] = min(velodyne_data[j], dist)
#                         break

# if __name__ == '__main__':

#     rclpy.init(args=None)

#     # Set the parameters for the implementation
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
#     seed = 0  # Random seed number
#     max_ep = 500  # maximum number of steps per episode
#     file_name = "td3_velodyne"  # name of the file to load the policy from
#     environment_dim = 20
#     robot_dim = 4

#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     state_dim = environment_dim + robot_dim
#     action_dim = 2

#     # Create the network
#     network = td3(state_dim, action_dim)
#     try:
#         network.load(file_name, "/root/rlearning_ws/src/td3_pkg/scripts/pytorch_models/")
#     except:
#         raise ValueError("Could not load the stored model parameters")

#     done = True
#     episode_timesteps = 0

#     # Create the testing environment
#     env = GazeboEnv()
#     odom_subscriber = Odom_subscriber()
#     velodyne_subscriber = Velodyne_subscriber()

#     executor = rclpy.executors.MultiThreadedExecutor()
#     executor.add_node(odom_subscriber)
#     executor.add_node(velodyne_subscriber)

#     executor_thread = threading.Thread(target=executor.spin, daemon=True)
#     executor_thread.start()
    
#     rate = odom_subscriber.create_rate(2)

#     # Begin the testing loop
#     while rclpy.ok():

#         # On termination of episode
#         if done:
#             state = env.reset()

#             action = network.get_action(np.array(state))
#             # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
#             a_in = [(action[0] + 1) / 2, action[1]]
#             next_state, reward, done, target = env.step(a_in)
#             done = 1 if episode_timesteps + 1 == max_ep else int(done)

#             done = False
#             episode_timesteps = 0
#         else:
#             action = network.get_action(np.array(state))
#             # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
#             a_in = [(action[0] + 1) / 2, action[1]]
#             next_state, reward, done, target = env.step(a_in)
#             done = 1 if episode_timesteps + 1 == max_ep else int(done)

#             state = next_state
#             episode_timesteps += 1

#     rclpy.shutdown()

