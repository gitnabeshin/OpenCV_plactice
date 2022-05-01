import numpy as np
import cv2

# 250 + 5
x = np.uint8([250])
y = np.uint8([5])
z = cv2.add(x, y)
print(f'z={z}')

# 250 + 10
x2 = np.uint8([250])
y2 = np.uint8([10])
z2 = cv2.add(x2, y2)
print(f'z2={z2}')

# 25 - 30
x3 = np.uint8([25])
y3 = np.uint8([30])
z3 = cv2.subtract(x3, y3)
print(f'z3={z3}')

src1 = cv2.imread('../img/crab.png', cv2.IMREAD_COLOR)

# create mask image
height, width , channnels = src1.shape[:3]
src2 = np.zeros((height, width, channnels), np.uint8)
cv2.rectangle(src2, (120, 35), (600, 490), (255, 255, 255), thickness=-1)

# exec AND operation on each pixel
dst_and = cv2.bitwise_and(src1, src2)

# exec OR operation on each pixel
dst_or = cv2.bitwise_or(src1, src2)

# cv2.imshow('src1', src1)
# cv2.imshow('src2', src2)
cv2.imshow('dst_and', dst_and)
cv2.imshow('dst_or', dst_or)
cv2.waitKey(0)
cv2.destroyAllWindows()

