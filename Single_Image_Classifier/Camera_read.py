# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 15:07:35 2018

@author: Rishabh Sharma
"""

import numpy as np
import cv2
from keras.models import load_model
import time
import sys

cap = cv2.VideoCapture(0)
forcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',forcc, 20.0, (640,480))
loaded_model = load_model('mnist_cnn_batchnorm2.h5')

while(True):
    try:
        ret,frame = cap.read()
    #    img = cv2.imread('7.png')
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        cv2.rectangle(frame, (int(height/4),int(width/4)),(int(height/2),int(width/2)),(255,0,0),3 )
        cv2.rectangle(gray, (int(height/4),int(width/4)),(int(height/2),int(width/2)),(255),3 )
        region = gray[int(width/4):int(width/2),int(height/4):int(height/2)]
    #    print(region)
        res, foreground_correction = cv2.threshold(region,80,255,cv2.THRESH_BINARY_INV)
        
        
        region_worked = cv2.resize(foreground_correction,(28,28),interpolation = cv2.INTER_CUBIC)
        
        cv2.imshow('region worked',region_worked)
        region_worked = np.reshape(region_worked,(28,28,1))
        region_worked = np.expand_dims(region_worked,axis = 0)
    #    loaded_model.summary()
        ans = loaded_model.predict(region_worked)
#        sys.stdout.write("{}\n".format(ans))
        ans = np.where(ans[0]==1)
#        time.sleep(0.1)
    #    print(ans)
    #    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #    cv2.imshow('img',img_gray)
    #    img_gray_test = cv2.resize(img_gray,(28,28),interpolation = cv2.INTER_CUBIC)
    #    
    #    img_gray_test = np.reshape(img_gray_test,(28,28,1))
    #    img_gray_test = np.expand_dims(img_gray_test,axis = 0)
    #    ans = loaded_model.predict(img_gray_test)
#        sys.stdout.write("{}\n".format(ans[0][0]))
        sys.stdout.flush()
        cv2.putText(frame,'The number in the image is '+str(ans[0][0]),(0,int(height)-10),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255),4)
#        cv2.imshow('region',region)
#        cv2.imshow('processing',foreground_correction)
        out.write(frame)
        cv2.imshow('frame1',frame)
#        cv2.imshow('frame',gray)
    #    time.sleep(0.5)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        pass
    
cap.release()
out.release()
cv2.destroyAllWindows()