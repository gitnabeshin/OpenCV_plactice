import cv2

# 0: internal camera
# 1: USB camera
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    cv2.imshow('camera', img)
    cv2.waitKey(1)
