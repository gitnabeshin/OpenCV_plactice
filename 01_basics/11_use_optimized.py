import cv2

print(f'BEFORE: {cv2.useOptimized()}')
cv2.setUseOptimized(False)
print(f'AFTER: {cv2.useOptimized()}')
