import cv2
import numpy as np
import matplotlib.pyplot as plt

img_bgr = cv2.imread('../img/keyboard.jpeg')
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(img_hsv)

plt.figure(figsize=(10, 6))

# make 2 * 3 matrix image area
plt.subplot(2, 3, 1)

plt.title('Original(RGB)')
plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

plt.subplot(2, 3, 2)
plt.title('Hue(H)')
plt.imshow(h, cmap='gray')

plt.subplot(2, 3, 3)
plt.title('Saturation(S)')
plt.imshow(s, cmap='gray')

plt.subplot(2, 3, 4)
plt.title('Value, Brightness(V)')
plt.imshow(v, cmap='gray')

# create mask img (blue area)
hsv_cp = img_hsv.copy()
# blue area
hsv_cp[(h>80) & (h<140) & (s>70)] = 255
mask = cv2.bitwise_xor(hsv_cp, img_hsv)

plt.subplot(2, 3, 5)
plt.title('mask')
plt.imshow(mask, cmap='gray')

# mask_3ch = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
# result = cv2.bitwise_and(img_bgr, mask_3ch)
result = cv2.bitwise_and(img_bgr, mask)

plt.subplot(2, 3, 6)
plt.title('result')
# plt.imshow(result, cmap='gray')
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

plt.show()
