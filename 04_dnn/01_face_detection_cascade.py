# ------------------------------------------------------------------------------------
# make sure download DNN Model from support page
# $ cd 04_dnn/model
# $ wget https://gihyo.jp/assets/files/book/2022/978-4-297-12775-6/download/7.2.zip
# ------------------------------------------------------------------------------------

import os
import numpy as np
import cv2

def fd_main():

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise IOError("can't open camera")

    # directory = os.path.dirname(__file__)
    directory = './model/7.2/haarcascade'

    # Load Classifier
    path = os.path.join(directory, "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(path)
    if cascade is None:
        raise IOError("Can't open cascade")

    while True:
        result, image = capture.read()
        if result is False:
            cv2.waitKey(0)
            break

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect face
        height, width = gray_image.shape
        min_size = (int(width/10), int(height/10))
        boxes = cascade.detectMultiScale(gray_image, minSize=min_size)

        # draw bounding  box
        for box in boxes:
            color = (0, 0, 255)
            thickness = 2
            cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)

        cv2.putText(image, 'PRESS q to exit.', (12, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.imshow('face_detection', image)

        key = cv2.waitKey(10)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    fd_main()
