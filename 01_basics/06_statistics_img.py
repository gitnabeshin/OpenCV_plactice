import numpy as np
import cv2

# -------------------------------
# min, max

img = np.array([[1, 2, 3, 4, 5]])

min = img.min()
max = img.max()

print(f'img = ', img)
print(f'  min = {min}, max = {max}')

# -------------------------------
# minMaxLoc()

img = np.array([[1, 2, 3], [4, 5, 6]])

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img)

print(f'img = ', img)
print(f'  min = ', min_val)
print(f'  max = ', max_val)
print(f'  min_loc = ', min_loc)
print(f'  max_loc = ', max_loc)

# -------------------------------
# mean(), sum()

img = np.array([[1, 2, 3, 4, 5]])

mean = img.mean()
sum = img.sum()

print(f'img = ', img)
print(f'  mean = ', mean)
print(f'  sum = ', sum)


# -------------------------------
# countNonZero()

img = np.array([[1, 0, 0, 4, 0]])

count = cv2.countNonZero(img)

print(f'img = ', img)
print(f'  count = ', count)
