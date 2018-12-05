#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import cv2
import time
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import tf_conversions
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import Imu
#for ssd
import os
import math
import random
import time
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import pdb

from sensor_msgs.msg import Image as ros_image
from cv_bridge import CvBridge, CvBridgeError
import sys

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/home/nvidia/Documents/initial_pose_data/videos/output'+str(time.time())+'.mp4', fourcc, 20.0, (640,480))

#detection
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile('/home/nvidia/Documents/initial_pose_data/frozen_model/frozen_inference_graph.pb', 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=tf_config)
    tf.import_graph_def(od_graph_def, name='')

tf_input = tf_sess.graph.get_tensor_by_name('image_tensor:0')
tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')


rospy.init_node('mle_pose', anonymous=True)
img_pub = rospy.Publisher('webcam_image', ros_image, queue_size=2)
def process_image(img):
    # Run SSD network.
    rclasses=[]
    rscores=[]
    rbboxes=[]

    image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    #image_np = load_image_into_numpy_array(image)
    image_resized = np.array(image)

    t0=time.time()
    detection_scores, detection_boxes, detection_classes = tf_sess.run([tf_scores, tf_boxes, tf_classes], feed_dict={tf_input: image_resized[None, ...]})

    scores=detection_scores[0]
    boxes=detection_boxes[0]
    classes=detection_classes[0]

    for i in range(scores.shape[0]):
        if scores[i]>0.6 and classes[i]!=8 and classes[i]!=2 and classes[i]!=4 and classes[i]!= 10:
            ymin=int(boxes[i][0])
            xmin=int(boxes[i][1])
            ymax=int(boxes[i][2])
            xmax=int(boxes[i][3])
            rclasses.append(classes[i])
            rscores.append(scores[i])
            rbboxes.append([boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]])

    for i in range(scores.shape[0]):
        if scores[i]>0.5:
            xmin=int(boxes[i][1]*640)
            ymin=int(boxes[i][0]*480)
            xmax=int(boxes[i][3]*640)
            ymax=int(boxes[i][2]*480)
            cv2.rectangle(image_resized, (xmin,ymin), (xmax, ymax), (0,0,255), 2)

    bridge = CvBridge()
    msg = bridge.cv2_to_imgmsg(image_resized, encoding="rgb8")
    img_pub.publish(msg)

    return rclasses, rscores, rbboxes

laser_data=[]
yaw=0

def laser_callback(msg):

    global laser_data
    laser_data=list(msg.ranges)

def odom_callback(msg):

    global yaw
    quaternion = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
    (roll,pitch,yaw) = tf_conversions.transformations.euler_from_quaternion(quaternion)
    #print yaw

def set_inital_pose(x, y, theta):
    # Define a set inital pose publisher.
    rospy.loginfo("start set pose...")
    p = PoseWithCovarianceStamped()
    p.header.stamp = rospy.Time.now()
    p.header.frame_id = "map"
    p.pose.pose.position.x = x
    p.pose.pose.position.y = y
    p.pose.pose.position.z = 1.0
    (p.pose.pose.orientation.x,
     p.pose.pose.orientation.y,
     p.pose.pose.orientation.z,
     p.pose.pose.orientation.w) = tf_conversions.transformations.quaternion_from_euler(0, 0, theta)
    p.pose.covariance[6 * 0 + 0] = 0.5 * 0.5
    p.pose.covariance[6 * 1 + 1] = 0.5 * 0.5
    p.pose.covariance[6 * 3 + 3] = math.pi / 12.0 * math.pi / 12.0

    setpose_pub.publish(p)

def cal_angle(delta_x, delta_y, delta_theta):

    theta=0
    if delta_x==0 and delta_y<0:
        theta=3.1415-delta_theta
    elif delta_x==0 and delta_y>0:
        theta=0-delta_theta
    elif delta_x>0 and delta_y>0:
        theta=math.atan(delta_y/delta_x)-delta_theta
    elif delta_x>0 and delta_y<0:
        theta=6.283+math.atan(delta_y/delta_x)-delta_theta
    elif delta_x<0 and delta_y>0:
        theta=3.1415+math.atan(delta_y/delta_x)-delta_theta
    elif delta_x<0 and delta_y<0:
        theta=math.atan(delta_y/delta_x)+3.1415-delta_theta

    if theta<0:
        theta=theta+6.283
    #print theta
    return theta


key_object = [" ", "bluechair", "yellowchair", "yellowdoor", "whitedoor", "statue", "transcan", "dropbox", "glassbox", "firedrant", "window", "watertank"]

#0 bg 1:bluechair 2:yellowchair 3:yellowdoor 4:whitedoor 5:statue 6:trashcan 7:dropbox 8:glassbox 9:firedrant 10:window 11:watertank

# object position in semantic map
object_position=[[[]], [[33.729, -14.886]], [[14.052, -27.629], [11.351, -27.725]], [[32.978, -30.646], [26.404, -24.739]], [[26.505, -15.033],[34.141, -10.344],[34.3,-29.482]], [[33.302, -21.79]], [[26.803, -20.001]], [[26.951, -18.541]], [[33.900,-19.000], [33.739, -24.400]], [[26.428, -27.076]], [[]], [[14.869,3.761], [13.922, 3.771], [7,262, 3.682], [5.965, 3.703]]]

if __name__ == '__main__':



    rospy.Subscriber('scan', LaserScan, laser_callback)
    rospy.Subscriber('/imu/data', Imu, odom_callback)
    cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
    setpose_pub = rospy.Publisher("initialpose",PoseWithCovarianceStamped,latch=True, queue_size=1)
    move_cmd = Twist()

    cap=cv2.VideoCapture(1) 
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    print height
    print width
    start_time=time.time()
    num=0
    target=[]
    original_yaw=yaw
    print original_yaw
    delta_yaw=0
    flag=0
    t0=time.time()
    while delta_yaw<6.28 and (not rospy.is_shutdown()):

        if delta_yaw<0 and time.time()-t0>15:
            flag=6.2832
        delta_yaw=yaw-original_yaw+flag
        #print 'delta:  ' + str(delta_yaw)

        move_cmd.angular.z=0.1
        #cmd_vel.publish(move_cmd)
        ret, frame = cap.read()
        out.write(frame)
        time0=time.time()
        rclasses, rscores, rbboxes =  process_image(frame)

        #print "inference time:  " + str(time.time()-time0)
        cmd_vel.publish(move_cmd)
        end_time=time.time()
        num+=1
        for i in range(len(rclasses)):

            if (int(rbboxes[i][1]*640)+int(rbboxes[i][3]*640))/2>300 and (int(rbboxes[i][1]*640)+int(rbboxes[i][3]*640))/2<360:
 
                if len(target)<1:
                    r=sum(laser_data[359:364])/5
                    if r<10.0:
                        print 'class:  ' + str(key_object[int(rclasses[i])])
                        print 'distance:   ' + str(r+0.4)
                        print 'angle:  ' + str(delta_yaw*180/3.1415)
                        target.append([int(rclasses[i]), r+0.4, delta_yaw])
     
                elif int(rclasses[i]) != target[len(target)-1][0]:
                    r=sum(laser_data[359:364])/5
                    if r<10.0:
                        print 'class:  ' + str(key_object[int(rclasses[i])])
                        print 'distance:   ' + str(r)
                        print 'angle:  ' + str(delta_yaw*180/3.1415)
                        target.append([int(rclasses[i]), r+0.4, delta_yaw])
                elif (delta_yaw-target[len(target)-1][2])>0.2:
                    r=sum(laser_data[359:364])/5
                    if r<10.0:
                        print 'class:  ' + str(key_object[int(rclasses[i])])
                        print 'distance:   ' + str(r)
                        print 'angle:  ' + str(delta_yaw*180/3.1415)
                        target.append([int(rclasses[i]), r+0.4, delta_yaw])


    print target
    move_cmd.angular.z=0.0
    cmd_vel.publish(move_cmd)
    cmd_vel.publish(move_cmd)

    min_sum=9999999999
    for x in range(26, 33):
        for y in range(-30, -9):
            num_3=0
            num_4=0
            num_8=0
            cur_sum=[0 for i in range(24)]
            for i in range(len(target)):
                if target[i][0]!=3 and target[i][0]!=8 and target[i][0]!=4:
                    for z in range(len(cur_sum)):
                        cur_sum[z]+=abs((object_position[target[i][0]][0][0]-x)**2+(object_position[target[i][0]][0][1]-y)**2-target[i][1]**2)**0.5

                if target[i][0]==3 and num_3==0:
                    for j in range(12):
                        cur_sum[j]+=abs((object_position[target[i][0]][0][0]-x)**2+(object_position[target[i][0]][0][1]-y)**2-target[i][1]**2)**0.5
                        cur_sum[j+12]+=abs((object_position[target[i][0]][1][0]-x)**2+(object_position[target[i][0]][1][1]-y)**2-target[i][1]**2)**0.5
                    num_3+=1
                if target[i][0]==3 and num_3==1:
                    for j in range(12):
                        cur_sum[j]+=abs((object_position[target[i][0]][1][0]-x)**2+(object_position[target[i][0]][1][1]-y)**2-target[i][1]**2)**0.5
                        cur_sum[j+12]+=abs((object_position[target[i][0]][0][0]-x)**2+(object_position[target[i][0]][0][1]-y)**2-target[i][1]**2)**0.5       

                if target[i][0]==8 and num_8==0:
                    for j in [0,1,2,3,4,5,12,13,14,15,16,17]:
                        cur_sum[j]+=abs((object_position[target[i][0]][0][0]-x)**2+(object_position[target[i][0]][0][1]-y)**2-target[i][1]**2)**0.5
                        cur_sum[j+6]+=abs((object_position[target[i][0]][1][0]-x)**2+(object_position[target[i][0]][1][1]-y)**2-target[i][1]**2)**0.5
                    num_8+=1
                if target[i][0]==8 and num_8==1:
                    for j in [0,1,2,3,4,5,12,13,14,15,16,17]:
                        cur_sum[j]+=abs((object_position[target[i][0]][1][0]-x)**2+(object_position[target[i][0]][1][1]-y)**2-target[i][1]**2)**0.5
                        cur_sum[j+6]+=abs((object_position[target[i][0]][0][0]-x)**2+(object_position[target[i][0]][0][1]-y)**2-target[i][1]**2)**0.5

                if target[i][0]==4 and num_4==0:
                    for j in [0,1,6,7,12,13,18,19]:
                        cur_sum[j]+=abs((object_position[target[i][0]][0][0]-x)**2+(object_position[target[i][0]][0][1]-y)**2-target[i][1]**2)**0.5
                        cur_sum[j+2]+=abs((object_position[target[i][0]][1][0]-x)**2+(object_position[target[i][0]][1][1]-y)**2-target[i][1]**2)**0.5
                        cur_sum[j+4]+=abs((object_position[target[i][0]][2][0]-x)**2+(object_position[target[i][0]][2][1]-y)**2-target[i][1]**2)**0.5
                    num_4+=1
                if target[i][0]==4 and num_4==1:
                    for j in [2,4,8,10,14,16,20,22]:
                        cur_sum[j]+=abs((object_position[target[i][0]][0][0]-x)**2+(object_position[target[i][0]][0][1]-y)**2-target[i][1]**2)**0.5
                    for j in [0,5,6,11,12,17,18,23]:
                        cur_sum[j]+=abs((object_position[target[i][0]][1][0]-x)**2+(object_position[target[i][0]][1][1]-y)**2-target[i][1]**2)**0.5
                    for j in [1,3,7,9,13,15,19,21]:
                        cur_sum[j]+=abs((object_position[target[i][0]][2][0]-x)**2+(object_position[target[i][0]][2][1]-y)**2-target[i][1]**2)**0.5
                    num_4+=1
                if target[i][0]==4 and num_4==2:
                    for j in [3,5,9,11,15,17,21,23]:
                        cur_sum[j]+=abs((object_position[target[i][0]][0][0]-x)**2+(object_position[target[i][0]][0][1]-y)**2-target[i][1]**2)**0.5
                    for j in [1,4,7,10,13,16,19,22]:
                        cur_sum[j]+=abs((object_position[target[i][0]][1][0]-x)**2+(object_position[target[i][0]][1][1]-y)**2-target[i][1]**2)**0.5
                    for j in [0,2,6,8,12,14,18,20]:
                        cur_sum[j]+=abs((object_position[target[i][0]][2][0]-x)**2+(object_position[target[i][0]][2][1]-y)**2-target[i][1]**2)**0.5    
            if min(cur_sum)<min_sum:
                min_sum=min(cur_sum)
                meter_x=x
                meter_y=y

    print "meter_x:    " + str(meter_x)
    print "meter_y:    " + str(meter_y)
    print "min_sum:    " + str(min_sum)

    min_sum=9999999999
    final_angle=0
    sm_x=0
    sm_y=0
    each_angle=[]
    for x in range((meter_x-1)*10, (meter_x+1)*10):
        x=x/10.0
        for y in range((meter_y-1)*10, (meter_y+1)*10):
            y=y/10.0
            cur_sum=[[0,0] for i in range(24)]
            angles=np.zeros((24,len(target)))
            num_3=0
            num_4=0
            num_8=0
            for i in range(len(target)):
                if target[i][0]!=3 and target[i][0]!=8 and target[i][0]!=4:
                    for z in range(len(cur_sum)):
                        cur_sum[z][0]+=abs((object_position[target[i][0]][0][0]-x)**2+(object_position[target[i][0]][0][1]-y)**2-target[i][1]**2)**0.5
                        cur_sum[z][1]+=cal_angle(object_position[target[i][0]][0][0]-x, object_position[target[i][0]][0][1]-y, target[i][2])
                        angles[z][i]=cal_angle(object_position[target[i][0]][0][0]-x, object_position[target[i][0]][0][1]-y, target[i][2])

                if target[i][0]==3 and num_3==1:
                    for j in range(12):
                        cur_sum[j][0]+=abs((object_position[target[i][0]][1][0]-x)**2+(object_position[target[i][0]][1][1]-y)**2-target[i][1]**2)**0.5
                        cur_sum[j][1]+=cal_angle(object_position[target[i][0]][1][0]-x, object_position[target[i][0]][1][1]-y, target[i][2])
                        angles[j][i]=cal_angle(object_position[target[i][0]][1][0]-x, object_position[target[i][0]][1][1]-y, target[i][2])
                        cur_sum[j+12][0]+=abs((object_position[target[i][0]][0][0]-x)**2+(object_position[target[i][0]][0][1]-y)**2-target[i][1]**2)**0.5 
                        cur_sum[j+12][1]+=cal_angle(object_position[target[i][0]][0][0]-x, object_position[target[i][0]][0][1]-y, target[i][2])      
                        angles[j+12][i]=cal_angle(object_position[target[i][0]][0][0]-x, object_position[target[i][0]][0][1]-y, target[i][2])

                if target[i][0]==3 and num_3==0:
                    for j in range(12):
                        cur_sum[j][0]+=abs((object_position[target[i][0]][0][0]-x)**2+(object_position[target[i][0]][0][1]-y)**2-target[i][1]**2)**0.5
                        cur_sum[j][1]+=cal_angle(object_position[target[i][0]][0][0]-x, object_position[target[i][0]][0][1]-y, target[i][2])
                        angles[j][i]=cal_angle(object_position[target[i][0]][0][0]-x, object_position[target[i][0]][0][1]-y, target[i][2])
                        cur_sum[j+12][0]+=abs((object_position[target[i][0]][1][0]-x)**2+(object_position[target[i][0]][1][1]-y)**2-target[i][1]**2)**0.5
                        cur_sum[j+12][1]+=cal_angle(object_position[target[i][0]][1][0]-x, object_position[target[i][0]][1][1]-y, target[i][2])
                        angles[j+12][i]=cal_angle(object_position[target[i][0]][1][0]-x, object_position[target[i][0]][1][1]-y, target[i][2])
                    num_3+=1

                if target[i][0]==8 and num_8==1:
                    for j in [0,1,2,3,4,5,12,13,14,15,16,17]:
                        cur_sum[j][0]+=abs((object_position[target[i][0]][1][0]-x)**2+(object_position[target[i][0]][1][1]-y)**2-target[i][1]**2)**0.5
                        cur_sum[j][1]+=cal_angle(object_position[target[i][0]][1][0]-x, object_position[target[i][0]][1][1]-y, target[i][2])
                        angles[j][i]=cal_angle(object_position[target[i][0]][1][0]-x, object_position[target[i][0]][1][1]-y, target[i][2])
                        cur_sum[j+6][0]+=abs((object_position[target[i][0]][0][0]-x)**2+(object_position[target[i][0]][0][1]-y)**2-target[i][1]**2)**0.5
                        cur_sum[j+6][1]+=cal_angle(object_position[target[i][0]][0][0]-x, object_position[target[i][0]][0][1]-y, target[i][2])
                        angles[j+6][i]=cal_angle(object_position[target[i][0]][0][0]-x, object_position[target[i][0]][0][1]-y, target[i][2])

                if target[i][0]==8 and num_8==0:
                    for j in [0,1,2,3,4,5,12,13,14,15,16,17]:
                        cur_sum[j][0]+=abs((object_position[target[i][0]][0][0]-x)**2+(object_position[target[i][0]][0][1]-y)**2-target[i][1]**2)**0.5
                        cur_sum[j][1]+=cal_angle(object_position[target[i][0]][0][0]-x, object_position[target[i][0]][0][1]-y, target[i][2])
                        angles[j][i]=cal_angle(object_position[target[i][0]][0][0]-x, object_position[target[i][0]][0][1]-y, target[i][2])
                        cur_sum[j+6][0]+=abs((object_position[target[i][0]][1][0]-x)**2+(object_position[target[i][0]][1][1]-y)**2-target[i][1]**2)**0.5
                        cur_sum[j+6][1]+=cal_angle(object_position[target[i][0]][1][0]-x, object_position[target[i][0]][1][1]-y, target[i][2])
                        angles[j+6][i]=cal_angle(object_position[target[i][0]][1][0]-x, object_position[target[i][0]][1][1]-y, target[i][2])
                    num_8+=1

                if target[i][0]==4 and num_4==2:
                    for j in [3,5,9,11,15,17,21,23]:
                        cur_sum[j][0]+=abs((object_position[target[i][0]][0][0]-x)**2+(object_position[target[i][0]][0][1]-y)**2-target[i][1]**2)**0.5
                        cur_sum[j][1]+=cal_angle(object_position[target[i][0]][0][0]-x, object_position[target[i][0]][0][1]-y, target[i][2])
                        angles[j][i]=cal_angle(object_position[target[i][0]][0][0]-x, object_position[target[i][0]][0][1]-y, target[i][2])
                    for j in [1,4,7,10,13,16,19,22]:
                        cur_sum[j][0]+=abs((object_position[target[i][0]][1][0]-x)**2+(object_position[target[i][0]][1][1]-y)**2-target[i][1]**2)**0.5
                        cur_sum[j][1]+=cal_angle(object_position[target[i][0]][1][0]-x, object_position[target[i][0]][1][1]-y, target[i][2])
                        angles[j][i]=cal_angle(object_position[target[i][0]][1][0]-x, object_position[target[i][0]][1][1]-y, target[i][2])
                    for j in [0,2,6,8,12,14,18,20]:
                        cur_sum[j][0]+=abs((object_position[target[i][0]][2][0]-x)**2+(object_position[target[i][0]][2][1]-y)**2-target[i][1]**2)**0.5
                        cur_sum[j][1]+=cal_angle(object_position[target[i][0]][2][0]-x, object_position[target[i][0]][2][1]-y, target[i][2])
                        angles[j][i]=cal_angle(object_position[target[i][0]][2][0]-x, object_position[target[i][0]][2][1]-y, target[i][2])

                if target[i][0]==4 and num_4==1:
                    for j in [2,4,8,10,14,16,20,22]:
                        cur_sum[j][0]+=abs((object_position[target[i][0]][0][0]-x)**2+(object_position[target[i][0]][0][1]-y)**2-target[i][1]**2)**0.5
                        cur_sum[j][1]+=cal_angle(object_position[target[i][0]][0][0]-x, object_position[target[i][0]][0][1]-y, target[i][2])
                        angles[j][i]=cal_angle(object_position[target[i][0]][0][0]-x, object_position[target[i][0]][0][1]-y, target[i][2])
                    for j in [0,5,6,11,12,17,18,23]:
                        cur_sum[j][0]+=abs((object_position[target[i][0]][1][0]-x)**2+(object_position[target[i][0]][1][1]-y)**2-target[i][1]**2)**0.5
                        cur_sum[j][1]+=cal_angle(object_position[target[i][0]][1][0]-x, object_position[target[i][0]][1][1]-y, target[i][2])
                        angles[j][i]=cal_angle(object_position[target[i][0]][1][0]-x, object_position[target[i][0]][1][1]-y, target[i][2])
                    for j in [1,3,7,9,13,15,19,21]:
                        cur_sum[j][0]+=abs((object_position[target[i][0]][2][0]-x)**2+(object_position[target[i][0]][2][1]-y)**2-target[i][1]**2)**0.5
                        cur_sum[j][1]+=cal_angle(object_position[target[i][0]][2][0]-x, object_position[target[i][0]][2][1]-y, target[i][2])
                        angles[j][i]=cal_angle(object_position[target[i][0]][2][0]-x, object_position[target[i][0]][2][1]-y, target[i][2])
                    num_4+=1

                if target[i][0]==4 and num_4==0:
                    for j in [0,1,6,7,12,13,18,19]:
                        cur_sum[j][0]+=abs((object_position[target[i][0]][0][0]-x)**2+(object_position[target[i][0]][0][1]-y)**2-target[i][1]**2)**0.5
                        cur_sum[j][1]+=cal_angle(object_position[target[i][0]][0][0]-x, object_position[target[i][0]][0][1]-y, target[i][2])
                        angles[j][i]=cal_angle(object_position[target[i][0]][0][0]-x, object_position[target[i][0]][0][1]-y, target[i][2])
                        cur_sum[j+2][0]+=abs((object_position[target[i][0]][1][0]-x)**2+(object_position[target[i][0]][1][1]-y)**2-target[i][1]**2)**0.5
                        cur_sum[j+2][1]+=cal_angle(object_position[target[i][0]][1][0]-x, object_position[target[i][0]][1][1]-y, target[i][2])
                        angles[j+2][i]=cal_angle(object_position[target[i][0]][1][0]-x, object_position[target[i][0]][1][1]-y, target[i][2])
                        cur_sum[j+4][0]+=abs((object_position[target[i][0]][2][0]-x)**2+(object_position[target[i][0]][2][1]-y)**2-target[i][1]**2)**0.5
                        cur_sum[j+4][1]+=cal_angle(object_position[target[i][0]][2][0]-x, object_position[target[i][0]][2][1]-y, target[i][2])
                        angles[j+4][i]=cal_angle(object_position[target[i][0]][2][0]-x, object_position[target[i][0]][2][1]-y, target[i][2])
                    num_4+=1

            for index, value in enumerate(cur_sum):
                if value[0]<min_sum:
                    min_sum=value[0]
                    sm_x=x
                    sm_y=y
                    min_index=index
                    final_angle=value[1]/len(target)
                    each_angles=angles[index]

    if min(each_angles)<0.8 and max(each_angles)>5.4:
        for i in range(len(each_angles)):
            if each_angles[i]>5.4:
                each_angles[i]=each_angles[i]-6.2832
    final_angle=sum(each_angles)/len(target)

    if final_angle>3.1415:
        final_angle=final_angle-6.2832
    print "sm_x:    " + str(sm_x)
    print "sm_y:    " + str(sm_y)
    print "index:   " + str(min_index)
    print "min_sum: " + str(min_sum)
    print "final_angle:  " + str(final_angle)
    print 'all angles:' + str(each_angles)

    test_set_pose_flag = True
    test_set_pose_cnt = 3

    while test_set_pose_flag == True:

        set_inital_pose(sm_x, sm_y, final_angle)
        test_set_pose_cnt -= 1
        if test_set_pose_cnt == 0:
            test_set_pose_flag = False

    rospy.spin()
    cap.release()
    out.release()
    cv2.destroyAllWindows()





























