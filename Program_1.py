# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:24:18 2023

@author: Kirovets
"""
import cv2
import sys
import numpy as np

# cv2.imread function reads an image from the local path.
img = cv2.imread('red_ball.jpg')
if img is None:
    sys.exit("Could not read the image.")    

# COLOR DETECTION STEP
# convert image to hsv image
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# lower bound and upper bound for red colors
bright_red_lower_bound = np.array([0, 70, 50])	 
bright_red_upper_bound = np.array([10, 255, 255])
dark_red_lower_bounds = np.array([160, 100, 100])
dark_red_upper_bounds = np.array([179, 255, 255])
# find the colors within the boundaries
bright_red_mask = cv2.inRange(hsv, bright_red_lower_bound, bright_red_upper_bound)
dark_red_mask = cv2.inRange(hsv, dark_red_lower_bounds, dark_red_upper_bounds)
# after masking the red shades out, add the two masks 
combined_mask = cv2.addWeighted(bright_red_mask, 1.0, dark_red_mask, 1.0, 0.0)
# Remove unnecessary noise from mask
# define kernel size  
kernel = np.ones((7,7),np.uint8)
# cv2.MORPH_CLOSE removes unnecessary black noises from the white region.
combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
# cv2.MORPH_OPEN removes white noise from the black region of the mask.
combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

# FIND CENTROID STEP
# convert the grayscale image to binary image
ret,thresh = cv2.threshold(combined_mask,0,255,0)
# calculate moments of binary image
M = cv2.moments(thresh)
# calculate x,y coordinate of center
if M["m00"] != 0:
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
else:
    cX, cY = 0, 0 
# put text and highlight the center
cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
cv2.putText(img, "red ball", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# SHOW RESULT STEP
cv2.imshow("Thresh", thresh)
cv2.imshow("Image", img)
cv2.waitKey(0)