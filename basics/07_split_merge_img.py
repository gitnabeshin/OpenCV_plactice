import cv2

# ---------------------------
# split()

img = cv2.imread('../img/crab.png', cv2.IMREAD_COLOR)

b_plane, g_plane, r_plane = cv2.split(img)

cv2.imshow('img', img)
cv2.imshow('b_plane', b_plane)
cv2.imshow('g_plane', g_plane)
cv2.imshow('r_plane', r_plane)


# ---------------------------
# merge()

merged = cv2.merge((b_plane, g_plane, r_plane))

cv2.imshow('merged', merged)

cv2.waitKey(0)
cv2.destroyAllWindows()

# ---------------------------
# hconcat()

img2 = cv2.imread('../img/crab2.png', cv2.IMREAD_COLOR)

hconcat_img = cv2.hconcat([img, img2])
vconcat_img = cv2.vconcat([img, img2])

cv2.imshow('hconcat_img', hconcat_img)
cv2.imshow('vconcat_img', vconcat_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

