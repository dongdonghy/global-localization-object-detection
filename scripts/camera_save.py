#!/usr/bin/env python

import numpy as np
import cv2
import time

cap = cv2.VideoCapture(1)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/home/nvidia/Documents/initial_pose_data/videos/output'+str(time.time())+'.mp4', fourcc, 20.0, (640,480))


while(True):
    ret, frame = cap.read()
    #cv2.imshow("capture", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()
