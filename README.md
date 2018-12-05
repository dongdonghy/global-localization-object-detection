# Global Localization Using Object Detection with Semantic Map

## Overview
Global localization is a key problem in autonomous robot. We use semantic map and object detection to do the global localization with MLE method.

## Dependencies
* Jetpack=3.3
* TensorFlow=1.9.0
* Opencv
* Matplotlib
* PIL
* ROS Kinetic

## Hardware
* Jetson TX2
* Lidar
* USB camera
* Autonomous robot
* Odometry by encoder or IMU

## Motivation
* In ROS system, if we use move_base package, we need to input an 2D initial pose by hand:
![image](https://github.com/dongdonghy/global-localization-object-detection/raw/master/images/artificial_pose.jpg)

<p align="center">
    <img src="https://github.com/dongdonghy/global-localization-object-detection/raw/master/images/artificial_pose.jpg" alt="Sample"  width="300" height="150">
    <p align="center">
        <em>artificial_pose</em>
    </p>
</p>


* Therefore, we want to calculate the initial pose automatically.


## How to Run

### Object Detection Model
* train an object detection model using tensorflow
* export the frozen model, and put it into `frozen_model` folder
* put the whole package into a ROS workspace
![image](https://github.com/dongdonghy/global-localization-object-detection/raw/master/images/object_detection.png)

### Semantic Map
* we build a semantic map with Gmapping and object detection.
* the backgroud is the grid map, and the points in the map represent the object position.
![image](https://github.com/dongdonghy/global-localization-object-detection/raw/master/images/semantic_map.png)

### ROS prepration
Before initial pose, you need to run the following node in ROS
* map server to output a map
* robot control: publish the cmd_vel and subscribe Odometry
* Lidar like Hokuyo, output the `scan` data

### Global Localization
* Run `python initial_pose.py` in `scripts` folder. 
* subscribe `scan`, `imu/data` topic, and need a USB camera
* publish `cmd/vel` to rotation, `webcam_image`, and the final `initialpose` 
![image](https://github.com/dongdonghy/global-localization-object-detection/raw/master/images/initial_pose.png)


### Other function
* `camera_save.py`: simple script to save the camera image
* `visilize.py`: an example script to test the frozen model with a video
* `send_goal.cpp`: we also provide a function which can send the navigation goal through voice recognition. Here we use the baidu package:
https://github.com/DinnerHowe/baidu_speech
* a center path algorithm, which you need to alter the grid_path.cpp in global planner package of navation stack.
![image](https://github.com/dongdonghy/global-localization-object-detection/raw/master/images/center_path.pnd)

