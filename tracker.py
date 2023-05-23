import cv2

cap = cv2.VideoCapture(0)     #from camera list get lensless camera

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=100)

while cap.open:
    ret, frame = cap.read()
    frame = frame[350:,:]

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
        print("NEW \n",detections,len(detections))
    cv2.imshow("frame", frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cap.close()
cv2.destroyAllWindows()