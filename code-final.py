import cv2
import nidaqmx
import numpy as np
import math
import time
import matplotlib.pyplot as plt


def mapper(pixel_dist):
    "Converts pixel distance to voltage with 1/d^2 relation"
    max_voltage = 5
    min_voltage = 0.01

    max_pixel_sq= 1/ (500)   #max pixel 500
    min_pixel_sq= 1/ (180)     #min pixel 100

    volt_distance= min_voltage+ (1/pixel_dist - max_pixel_sq) * (max_voltage-min_voltage) /(min_pixel_sq - max_pixel_sq)
    if pixel_dist >500:
        volt_distance = 0.01
    elif pixel_dist < 180:
        volt_distance = 4.99
    return volt_distance

def signal_creator(history):
    "Acquires oldest data in history and creates voltage array using mapper"
    pix_distance = history[6][0]    # for recipocal 0, nonrecipocal 6
    voltage = mapper(pix_distance)
    voltage_array = np.asarray([voltage,voltage])
    return voltage_array



cap = cv2.VideoCapture(0)   #from camera list get camera, if additional camera turn this to 1

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=80) # hist and var should be tried

# Device creator
max_voltage = 5
min_voltage = 0


# Initializations
start_time = time.time()
history= []
total_times = []
total_distances= []
total_x_1 = []
total_y_1 = []

total_x_2 = []
total_y_2 = []

with nidaqmx.Task() as task:
    task.ao_channels.add_ao_voltage_chan("Dev1/ao0", max_val=max_voltage, min_val=min_voltage)
    task.ao_channels.add_ao_voltage_chan("Dev1/ao1", max_val=max_voltage, min_val=min_voltage)
    print(mapper(300))
    print(mapper(150))
    print(mapper(100))
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

            total_times.append(end_time)
            total_distances.append(distance)

            total_x_1.append(detections[0][0])
            total_y_1.append(detections[0][1])
            total_x_2.append(detections[1][0])
            total_y_2.append(detections[1][1])

            # Storing values
            if len(history) < 7:    # initialization
                history.append([distance, end_time])
            elif (end_time - history[2][1]) > threshold:   # If time passed enough
                if len(history) == 7:   # Pop oldest, append new
                    history.pop(0)
                    history.append([distance,end_time])


                    # Signal writing
                    print(history)
                    voltage_array = signal_creator(history)
                    print(voltage_array)
                    task.write(voltage_array, auto_start=True)
                    time.sleep(threshold/2)
                    # task.write([0,0], auto_start= True)


        cv2.imshow("frame", frame)      # showing frame

        # Press "q" to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# plt.plot(total_times, total_distances)
# plt.title("Time vs distance of balls")
# plt.show()
# print(total_times)
# print(total_distances)

mean = (np.mean(total_x_1) + np.mean(total_x_2)) /2
plt.plot(total_times,np.abs(mean-np.asarray(total_x_1)))
plt.plot(total_times,np.abs(mean-np.asarray(total_x_2)))
plt.title("Time vs x of balls")
plt.show()
print(total_times)
print(total_distances)

cap.release()
cap.close()
cv2.destroyAllWindows()