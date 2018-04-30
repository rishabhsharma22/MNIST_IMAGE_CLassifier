# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 15:53:20 2018

@author: Rishabh Sharma
"""

import numpy as np
import cv2
from keras.models import load_model
import time
import sys
import matplotlib.pyplot as plt
#import imutils

img = cv2.imread("image1.jpg",0)

loaded_model = load_model('mnist_cnn_batchnorm2.h5')

cap = cv2.VideoCapture(0)

forcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output1.avi',forcc, 20.0, (640,480))

while(True):
    try:
        ret,frame1 = cap.read()
        frame = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        ret1,thresh = cv2.threshold(frame,90,255,cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        	cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1]
        digitCnts = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area > 1000 and area <40000:
#                print(area)
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(frame1,(x,y),(x+w,y+h),(255,0,0),0)
                img1 = thresh[y:y+h,x:x+w]
                pred = cv2.resize(img1,(28,28),interpolation = cv2.INTER_CUBIC)
                pred = np.reshape(pred,(28,28,1))
                pred = np.expand_dims(pred,axis = 0)
                ans = loaded_model.predict(pred)
                ans = np.where(ans[0]==max(ans[0]))
#                print(ans[0][0])
                
                cv2.putText(frame1,"Number:"+str(ans[0][0]),(int(x),int(y+h+10)),cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0),2)
        
        sys.stdout.flush()
        cv2.imshow("frame",frame1)
#        cv2.imshow("thresh",thresh)
#        time.sleep(0.25)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        pass
    
cap.release()
out.release()
cv2.destroyAllWindows()

	# compute the bounding box of the contour
#	(x, y, w, h) = cv2.boundingRect(c)
 
	# if the contour is sufficiently large, it must be a digit
    


