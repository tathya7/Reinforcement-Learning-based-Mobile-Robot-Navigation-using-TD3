# Mobile Robot Navigation using Twin Deep Delayed Deterministic Policy Gradient (TD3)

Reinforcement Learning, an evolving discipline, is primarily employed to accomplish tasks by guiding the actions of a robot based on rewards received. In our project, we've endeavored to implement one such algorithm to train a mobile robot within a specific environment, aiming to reach a designated goal location over a set number of iterations. Subsequently, we evaluated the trained model's performance within the same environment. Our approach involved utilizing TD3, an off-policy algorithm, for this purpose.

## Results

This video, shows the testing of the trained model, where it can be seen, the robot will go to the goal location denoted by Green Dot without colliding with obstacles.

Testing Simulation : https://youtu.be/MAGf0qRb2nc

Training Simulation : https://youtu.be/t24d_GHjucQ

![ezgif com-video-to-gif-converter (3)](https://github.com/tathya7/Reinforcement-Learning-based-Mobile-Robot-Navigation/assets/105652825/f9f4656e-c134-4c5c-ad87-14aeedfc152e)



# How To Run

1. Install Python 3.10, ROS2 Humble, Gazebo on Ubuntu 22.04

2. Install all the dependencies like Pytorch, Tensorboard , Quaternion for the algorithm to run using:

    ```
    pip install <dependency name>
    ```

3. If your current version is not supported, then you can use the docker file provided in the repo to create a container and in that container, cloning the repo and building it will make it run. Follow the readme file in rlearning_docker first. Then perform the following from step 2.

2. Download two packages provided or use the github link to clone the repo    
    ```
    git clone https://github.com/tathya7/Reinforcement-Learning-based-Mobile-Robot-Navigation-using-TD3.git
    ```
    - **td3_pkg** contains the original source code and gazebo environment
    - **velodyne_simulator** is used for RViz Visualization of Velodyne LiDAR

3. Create a workspace, copy the package into the workspace and build, it should complelety build.

    ```
    mkdir -p <workspace name>/src
    " Paste both the packages in src directory (td3_pkg and velodyne_simulator) "
    colcon build
    source install/setup.bash
    ```

4. For Training the model
    ```
    ros2 launch td3_pkg training_simulation.launch.py
    ```

5. For Testing the model which uses the trained model 
    ```
    ros2 launch td3_pkg test_simulation.launch.py
    ```

### Notes

- There are time when you face an error, one common error is in building the package and it throws an ament_cmake error, it resolve this use this command to source ROS2

    ```
    source /opt/ros/humble/setup.bash
    ```

- If you don't have a suitable version of ROS, this may or may not run and so it is advisable to use the docker container and clone the repository inside the container.


### Credits

- Author of the referred repository - https://github.com/reiniscimurs/DRL-robot-navigation.git 

