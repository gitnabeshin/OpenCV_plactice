import numpy as np
import cv2

# ---------------------
# addWeighted()

src1 = cv2.imread('../img/crab.png', cv2.IMREAD_COLOR)
src2 = cv2.imread('../img/crab2.png', cv2.IMREAD_COLOR)

alpha = 0.5
beta = 0.5
gamma = 0.0

# brend 2 imgs with weight
dst = cv2.addWeighted(src1, alpha, src2, beta, gamma)

cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -------------------
# absdiff()

img1 = np.array([1, 2, 3, 4, 5])
img2 = np.array([5, 4, 3, 2, 1])

diff = cv2.absdiff(img1, img2)

print(diff)

