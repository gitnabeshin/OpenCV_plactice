import numpy as np
import cv2

width = 200
height = 100
value = 128  # 0-255

channels = 3 #RGB
col_value = (0, 0, 255) #BGR

# black
img1 = np.zeros((height, width), np.uint8)
# gray
img2 = np.full((height, width), value, np.uint8)
# color (0, 0, 0)
img3 = np.zeros((height, width, channels), np.uint8)
# color col_value
img4 = np.full((height, width, channels), col_value, np.uint8)

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)

cv2.imshow('img3', img3)
cv2.imshow('img4', img4)

# wait until pressing '0' on popup window
cv2.waitKey(0)
cv2.destroyAllWindows()
