# ------------------------------------------------------------------------------------
# Text Detection(Differenciable Binarization)
# make sure download DNN Model from support page
# $ cd 04_dnn/model
# $ wget https://gihyo.jp/assets/files/book/2022/978-4-297-12775-6/download/7.6.zip
# ------------------------------------------------------------------------------------

import os
import numpy as np
import cv2
import time

def dnn_main():

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise IOError("can't open camera")

    # directory = os.path.dirname(__file__)
    directory = './model/7.6/db'

    # load DNN from file
    weights = os.path.join(directory, "DB_IC15_resnet18.onnx")    # English, Number
    # weights = os.path.join(directory, "DB_IC15_resnet50.onnx")      # English, Number
    # weights = os.path.join(directory, "DB_TD500_resnet18.onnx")   # English, Chinese, Number
    # weights = os.path.join(directory, "DB_TD500_resnet50.onnx")   # English, Chinese, Number
    model = cv2.dnn_TextDetectionModel_DB(weights)

    # set params
    scale = 1.0 /255.0
    # size = (736, 736)    # MSRA-TD500
    size = (736, 1280)     # ICDAR2015
    mean = (122.6, 116.7, 104.0)
    swap = False
    crop = False
    model.setInputParams(scale, size, mean, swap, crop)

    # set text detection param
    binary_threshold = 0.3
    polygon_threshold = 0.5
    max_candidates = 200
    unclip_ratio = 2.0
    model.setBinaryThreshold(binary_threshold)
    model.setPolygonThreshold(polygon_threshold)
    model.setMaxCandidates(max_candidates)
    model.setUnclipRatio(unclip_ratio)

    pTime = 0
    cTime = 0

    while True:
        result, image = capture.read()
        if result is False:
            cv2.waitKey(0)
            break

        # get detection result
        vertices, confidences = model.detect(image)
        # rotated_rectangles, confidences = model.detectTextRectangles(image)
        # vertices = get_vertices(totated_rectangles)  # need to be converted

        for vertex in vertices:
            vertex = np.array(vertex)
            close = True
            color = (255, 0, 0)
            thickness = 2
            cv2.polylines(image, [vertex], close, color, thickness, cv2.LINE_AA)

        # Frame rate
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(image, 'PRESS q to exit.', (12, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.putText(image,  f'FPS: {int(fps)}', (12, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.imshow('text detection', image)

        key = cv2.waitKey(10)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    dnn_main()
