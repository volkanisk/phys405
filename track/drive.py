import cv2
import nidaqmx
import numpy as np
import time

def mapper(pixel_dist):
    max_voltage = 5
    min_voltage = 0
    max_pixel_sq= 1/ (500**2)   #max pixel 300
    min_pixel_sq= 1/ (1**2)     #min pixel 100

    volt_distance= (1/pixel_dist**2 - max_pixel_sq) * max_voltage /(min_pixel_sq - max_pixel_sq)
    return volt_distance

cap = cv2.VideoCapture(0)     #from camera list get lensless camera

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=100)

max_voltage = 5
min_voltage = 0

start_time = time.time()
history= []
while cap.open:
    ret, frame = cap.read()
    frame = frame[350:,:]

    # nidaqmx.Task().ao_channels.add_ao_voltage_chan("Dev1/ao0", max_val=max_voltage, min_val=min_voltage)
    # nidaqmx.Task().ao_channels.add_ao_voltage_chan("Dev1/ao1", max_val=max_voltage, min_val=min_voltage)

    # 1. Object Detection
    mask = object_detector.apply(frame)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    cv2.imshow("mask", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    detections = []
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 100:
            # cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            detections.append([x,y,w,h])
    if detections != []:
        if len(detections) == 2:
            end_time = time.time()-start_time
            # print("NEW \n",detections,len(detections),end_time)
            distance = ((detections[0][0] - detections[1][0])**2 +
                        (detections[0][1] - detections[1][1]) ** 2)**0.5

            if len(history) == 3:
                history.pop(0)
                history.append([distance,end_time])
            if len(history)<3:
                history.append([distance,end_time])
            print(history)

    cv2.imshow("frame", frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cap.close()
cv2.destroyAllWindows()