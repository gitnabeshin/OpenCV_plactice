import numpy as np
import cv2

# ---------------------
# copy

img_org = cv2.imread('../img/crab.png', cv2.IMREAD_COLOR)

mask = np.full(img_org.shape, 255, np.uint8)
# copy to another area
cv_copy_img = cv2.copyTo(img_org, mask)

# copy to another area
img_numpy_copy = img_org.copy()

# original
img_shallow_copy = img_org

# edit original img
cv2.rectangle(img_org, (0, 0), (300, 300), (255, 255, 255), thickness=-1)

cv2.imshow('cv_copy_img', cv_copy_img)
cv2.imshow('img_numpy_copy', cv_copy_img)
cv2.imshow('img_shallow_copy', img_shallow_copy)

cv2.waitKey(0)
cv2.destroyAllWindows()

# -------------------------
# flip

img = cv2.imread('../img/crab.png', cv2.IMREAD_COLOR)

# 0: flip on x axis
dst_flip = cv2.flip(img, 0)

cv2.imshow('img', img)
cv2.imshow('dst_flip', dst_flip)
cv2.waitKey(0)
cv2.destroyAllWindows()


# -------------------------
# rotate

# cv2.ROTATE_90_CLOCKWISE
# cv2.ROTATE_90_COUNTERCLOCKWISE
# cv2.ROTATE_180
dst_rotate = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

cv2.imshow('img', img)
cv2.imshow('dst_rotate', dst_rotate)
cv2.waitKey(0)
cv2.destroyAllWindows()


# -------------------------
# clip

img = np.array([[0, 1, 2, 3, 4, 5, 6]])

clip1 = img.clip(1, 5)
print(f'clip1 = {clip1}')

clip2 = img.clip(None, 5)
print(f'clip2 = {clip2}')
