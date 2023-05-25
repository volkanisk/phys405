import cv2
import nidaqmx
import numpy as np
import time
import matplotlib.pyplot as plt

def mapper(pixel_dist):
    "Converts pixel distance to voltage with 1/d^2 relation"
    max_voltage = 5
    min_voltage = 0
    max_pixel_sq= 1/ (500**2)   #max pixel 300
    min_pixel_sq= 1/ (1**2)     #min pixel 100

    volt_distance= (1/pixel_dist**2 - max_pixel_sq) * max_voltage /(min_pixel_sq - max_pixel_sq)
    return volt_distance


def signal_creator(history):
    "Acquires oldest data in history and creates voltage array using mapper"
    pix_distance = history[0][0]
    voltage = mapper(pix_distance)
    voltage_array = np.asarray([voltage,voltage])
    return voltage_array



cap = cv2.VideoCapture(0)   #from camera list get camera, if additional camera turn this to 1

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=100) # hist and var should be tried

# Device creator
max_voltage = 5
min_voltage = 0
nidaqmx.Task().ao_channels.add_ao_voltage_chan("Dev1/ao0", max_val=max_voltage, min_val=min_voltage)
nidaqmx.Task().ao_channels.add_ao_voltage_chan("Dev1/ao1", max_val=max_voltage, min_val=min_voltage)

# Initializations
start_time = time.time()
history= []
total_history = []


while cap.open:
    # Getting frames
    ret, frame = cap.read()
    frame = frame[350:,:]   # will be adjusted with new camera


    # 1. Object Detection
    mask = object_detector.apply(frame)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    cv2.imshow("mask", mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Finding and storing big boxes
    detections = []
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)

        #Storing elements bigger than area treshold
        if area > 100:  # Area threshold can be adjusted
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            detections.append([x,y,w,h])
            end_time = time.time() - start_time

    # If only two objects detected
    if len(detections) == 2:
        threshold = 0.05    # time threshold between detections (s) bakılacak
        distance = ((detections[0][0] - detections[1][0])**2 +
                    (detections[0][1] - detections[1][1]) ** 2)**0.5  # y value can be changed to y+h

        total_history.append([distance,end_time])

        # Storing values
        if len(history) < 3:    # initialization
            history.append([distance, end_time])
        elif (end_time - history[2][1]) > threshold:   # If time passed enough
            if len(history) == 3:   # Pop oldest, append new
                history.pop(0)
                history.append([distance,end_time])

                # Signal writing
                voltage_array = signal_creator(history)
                nidaqmx.Task().write(voltage_array, auto_start=True)


    cv2.imshow("frame", frame)      # showing frame

    # Press "q" to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

plt.plot(total_history[:][1], total_history[:][0])
plt.title("Time vs distance of balls")
plt.show()

cap.release()
cap.close()
cv2.destroyAllWindows()