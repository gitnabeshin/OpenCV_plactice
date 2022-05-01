import numpy as np
import cv2
import sys

filename = 'output.xml'

fs = cv2.FileStorage(filename, cv2.FileStorage_WRITE)

if fs.isOpened() is False:
    print('open error.')
    sys.exit(1)

R = np.eye(3, 3)
T = np.zeros((3, 1))

fs.write('R_MAT', R)
fs.write('T_MAT', T)

fs.writeComment('This is a comment.')

fs.release()

# ----------------------------------------------

fs2 = cv2.FileStorage(filename, cv2.FileStorage_READ)

if fs2.isOpened() is False:
    print('open error.')
    sys.exit(1)

R = fs2.getNode('R_MAT').mat()
T = fs2.getNode('T_MAT').mat()

print(R)
print(T)

fs2.release()
