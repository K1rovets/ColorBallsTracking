import numpy as np
import cv2

# lower bound and upper bound for color
"""
red_lower_bound = np.array([0, 150, 110])	 
red_upper_bound = np.array([10, 255, 255])
    
green_lower_bound = np.array([40, 50, 38])	 
green_upper_bound = np.array([70, 255, 255])
    
blue_lower_bound = np.array([101, 50, 38])	 
blue_upper_bound = np.array([130, 255, 255])
    
yellow_lower_bound = np.array([20, 50, 70])	 
yellow_upper_bound = np.array([35, 255, 255])
"""
# define the lower and upper boundaries of the colors in the HSV color space
lower = {'red':(0, 150, 110), 'green':(40, 50, 38), 'blue':(101, 50, 38), 'yellow':(20, 50, 70)} 
upper = {'red':(10, 255, 255), 'green':(70, 255, 255), 'blue':(130, 255, 255), 'yellow':(40, 255, 255)}

cap = cv2.VideoCapture('rgb_ball_720.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break    

    # COLOR DETECTION STEP
    # convert to hsv colorspace
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for key, value in lower.items():
        # find the colors within the boundaries
        mask = cv2.inRange(hsv, lower[key], upper[key]) 
        # binarize the image
        binr = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] 
        # Remove unnecessary noise from mask
        # define kernel size - np.ones((7,7),np.uint8) create a 5×5 8 bit integer matrix. 
        kernel = np.ones((7,7),np.uint8)
        # cv2.MORPH_CLOSE removes unnecessary black noises from the white region.
        binr = cv2.morphologyEx(binr, cv2.MORPH_CLOSE, kernel, iterations=2)
        # cv2.MORPH_OPEN removes white noise from the black region of the mask.
        binr = cv2.morphologyEx(binr, cv2.MORPH_OPEN, kernel, iterations=2)    
        # Segment only the detected region
        segmented_img = cv2.bitwise_and(frame, frame, mask=binr)
        
        #FIND CENTROID STEP
        # convert image to grayscale image
        gray_frame = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)    
        # convert the grayscale image to binary image
        ret,thresh = cv2.threshold(gray_frame, 0, 255, 0)    
        # calculate moments of binary image
        M = cv2.moments(thresh)
        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        # put text and highlight the center
        cv2.circle(frame, (cX, cY), 3, (255, 255, 255), -1) 
        cv2.putText(frame, key + " ball", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # Find contours from the mask
        contours, hierarchy = cv2.findContours(binr.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw contour on original image
        output = cv2.drawContours(frame, contours, -1, (255, 255, 255), 3)
    
    #SHOW RESULT
    cv2.imshow('frame, click q to quit', frame)    
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()