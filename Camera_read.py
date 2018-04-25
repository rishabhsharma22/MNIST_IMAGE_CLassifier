# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 15:07:35 2018

@author: Rishabh Sharma
"""

import numpy as np
import cv2


cap = cv2.VideoCapture(0)

while(True):
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    cv2.rectangle(gray, (0,0),(int(height/2),int(width/2)),(255),3 )
    region = gray[0:int(width/2),0:int(height/2)]
#    cv2.imshow('frame',frame)
    cv2.imshow('frame',region)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()