# ------------------------------------------------------------------------------------
# High level API
# make sure download DNN Model from support page
# $ cd 04_dnn/model
# $ wget https://gihyo.jp/assets/files/book/2022/978-4-297-12775-6/download/7.2.zip
# ------------------------------------------------------------------------------------

import os
import numpy as np
import cv2

def dnn_main():

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise IOError("can't open camera")

    # directory = os.path.dirname(__file__)
    directory = './model/7.2/opencv_face_detector'

    # load DNN from file
    weights = os.path.join(directory, "opencv_face_detector_fp16.caffemodel")
    config = os.path.join(directory, "opencv_face_detector_fp16.prototxt")
    # weights = os.path.join(directory, "opencv_face_detector_fp8.caffemodel")
    # config = os.path.join(directory, "opencv_face_detector_fp8.prototxt")
    # weights = os.path.join(directory, "opencv_face_detector.caffemodel")
    # config = os.path.join(directory, "opencv_face_detector.prototxt")
    model = cv2.dnn_DetectionModel(weights, config)

    # set params
    scale = 1.0
    size = (300, 300)
    mean = (104.0, 177.0, 123.0)
    swap = False
    crop = False
    model.setInputParams(scale, size, mean, swap, crop)

    while True:
        result, image = capture.read()
        if result is False:
            cv2.waitKey(0)
            break

        # face detection
        confidence_threshold = 0.6
        # nms:Non-Maximum Suppression (0.4: intersection rate of 2 candidate region)
        nms_threshold = 0.4
        _, _, boxes = model.detect(image, confidence_threshold, nms_threshold)
        # class_id, condidence, boxes

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
    dnn_main()
