import numpy as np
import cv2

img = cv2.imread('../img/crab.png', cv2.IMREAD_COLOR)
print(f'shape = {img.shape}')
print(f'dtype = {img.dtype}')

x = 200
y = 100
channel = 0

# print 3 channel(B, G, R) val
bgr_val = img[y, x]
print(f'val_bgr({x}, {y}) = {bgr_val}')

# print 0th channel(Blue) val
b_val = img[y, x, channel]
print(f'val_b({x}, {y}) = {b_val}')

# change BGR pixel value
img[y, x] = [255, 255, 255]
print(f'AFTER: val_bgr({x}, {y}) = {img[y, x]}')

# change B value
img[y, x, channel] = 0
print(f'AFTER: val_b({x}, {y}) = {img[y, x, channel]}')

# ROI: Regin Of Interest
img_roi = img[35:490, 120:600]

cv2.imshow('img', img)
cv2.imshow('img_roi', img_roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
