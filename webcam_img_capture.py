import cv2
import numpy as np

cam = cv2.VideoCapture("rtsp://10.20.30.40:8554/test")     #VUP Showcase Camera
#cam = cv2.VideoCapture("rtspsrc location=rtsp://10.20.30.40:8554/test latency=20 ! queue ! rtph264depay ! queue ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw,format=NV12 ! videoconvert ! video/x-raw,format=NV12 ! appsink", cv2.CAP_GSTREAMER)
#cam = cv2.VideoCapture("10.144.255.181")

cv2.namedWindow("test")

img_counter = 0
mat = np.array([[1.35870397e+03, 0.00000000e+00, 1.00463350e+03],
 [0.00000000e+00, 1.35899970e+03, 5.46221549e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

dist  = np. array([-3.94526378e-01, 1.85633819e-01, 9.34000802e-04, 9.81036008e-05, -5.50802105e-02])

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    #img_undist = cv2.undistort(frame, mat, dist)

    cv2.imshow("test", frame)
    

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "frame_{}.jpg".format(img_counter)
        cv2.imwrite("calibration_images/labeling_4/" + img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()