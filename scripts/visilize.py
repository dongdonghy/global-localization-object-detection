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


def process_image(img):
    # Run SSD network.
    rclasses=[]
    rscores=[]
    rbboxes=[]

    image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    #image_np = load_image_into_numpy_array(image)
    image_resized = np.array(image)

    detection_scores, detection_boxes, detection_classes = tf_sess.run([tf_scores, tf_boxes, tf_classes], feed_dict={tf_input: image_resized[None, ...]})

    scores=detection_scores[0]
    boxes=detection_boxes[0]
    classes=detection_classes[0]

    #pdb.set_trace()
    for i in range(scores.shape[0]):

        if scores[i]>0.5:
            ymin=int(boxes[i][0])
            xmin=int(boxes[i][1])
            ymax=int(boxes[i][2])
            xmax=int(boxes[i][3])
            rclasses.append(classes[i])
            rscores.append(scores[i])
            rbboxes.append([boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]])

    return rclasses, rscores, rbboxes


if __name__ == '__main__':

    
    cap=cv2.VideoCapture('/home/nvidia/Documents/videos/test7.mp4') 
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
 
    while True:
 
        ret, frame = cap.read()
        classes, scores, boxes =  process_image(frame)
        for i in range(len(scores)):
            if scores[i]>0.5:
                xmin=int(boxes[i][1]*width)
                ymin=int(boxes[i][0]*height)
                xmax=int(boxes[i][3]*width)
                ymax=int(boxes[i][2]*height)
                cv2.rectangle(frame, (xmin,ymin), (xmax, ymax), (0,0,255), 2)
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()





























