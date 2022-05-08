# ------------------------------------------------------------------------------------
# Text Recognition(CRNN-CTC, DenseNet-CTC)
# make sure download DNN Model from support page
# $ cd 04_dnn/model
# $ wget https://gihyo.jp/assets/files/book/2022/978-4-297-12775-6/download/7.6.zip
# ------------------------------------------------------------------------------------

import os
import numpy as np
import cv2
import time

def read_vocabularies(file):
    vocaburaries = None
    with open(file, mode='r', encoding="utf-8") as f:
        vocaburaries = f.read().splitlines()
    return vocaburaries

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
    detect_model = cv2.dnn_TextDetectionModel_DB(weights)

    # set params
    scale = 1.0 /255.0
    # size = (736, 736)    # MSRA-TD500
    size = (736, 1280)     # ICDAR2015
    mean = (122.6, 116.7, 104.0)
    swap = False
    crop = False
    detect_model.setInputParams(scale, size, mean, swap, crop)

    # set text detection param
    binary_threshold = 0.3
    polygon_threshold = 0.5
    max_candidates = 200
    unclip_ratio = 2.0
    detect_model.setBinaryThreshold(binary_threshold)
    detect_model.setPolygonThreshold(polygon_threshold)
    detect_model.setMaxCandidates(max_candidates)
    detect_model.setUnclipRatio(unclip_ratio)


    # directory = os.path.dirname(__file__)
    directory = './model/7.6/crnn-ctc'
    # directory = './model/7.6/densenet-ctc'

    # load DNN from file
    # weights = os.path.join(directory, "crnn.onnx")            # English(lowercase), Number, 36 vocabularies
    weights = os.path.join(directory, "crnn_cs.onnx")           # English(uppercase, lowercase), Number, symbol, 94 vocabularies
    # weights = os.path.join(directory, "crnn_cs_CN.onnx")      # English(uppercase, lowercase), Chinese, Number, symbol, 3944 vocabularies
    # weights = os.path.join(directory, "DenseNet_CTC.onnx")    # English(lowercase), Number, 36 vocabularies
    recognition_model = cv2.dnn_TextRecognitionModel(weights)

    # check model name needs for gray input image
    is_require_gray = False
    if "crnn.onnx" in weights:
        is_require_gray = True
    if "DenseNet_CTC" in weights:
        is_require_gray = True

    # set params
    scale = 1.0 /255.0
    size = (100, 32)
    mean = (127.5, 127.5, 127.5)
    swap = True
    crop = False
    recognition_model.setInputParams(scale, size, mean, swap, crop)

    # set decode type
    type = "CTC-greedy"
    # type = "CTC-prefix-beam-search"
    recognition_model.setDecodeType(type)

    # set label
    # vocabularies = read_vocaburaries("./alphabet_36.txt")
    vocabularies = read_vocabularies("./alphabet_94.txt")
    # vocaburaries = read_vocaburaries("./alphabet_3944.txt")
    recognition_model.setVocabulary(vocabularies)

    pTime = 0
    cTime = 0

    while True:
        result, image = capture.read()
        if result is False:
            cv2.waitKey(0)
            break

        # get detection result
        vertices, confidences = detect_model.detect(image)
        # rotated_rectangles, confidences = detect_model.detectTextRectangles(image)
        # vertices = get_vertices(totated_rectangles)  # need to be converted

        size = (100, 32)
        for vertex in vertices:
            # draw line
            vertex = np.array(vertex)
            close = True
            color = (255, 0, 0)
            thickness = 2
            cv2.polylines(image, [vertex], close, color, thickness, cv2.LINE_AA)

            # transform text area
            source_points= np.array(vertex, dtype=np.float32)
            target_points = np.array([[0, size[1]], [0, 0], [size[0], 0], [size[0], size[1]]], dtype=np.float32)
            transform_matrix = cv2.getPerspectiveTransform(source_points, target_points)
            text_image = cv2.warpPerspective(image, transform_matrix, size)

            # convert text image to gray only for crnn.onnx, DenseNet.onnx
            _, _, channels = text_image.shape
            if is_require_gray and channels != 1:
                text_image = cv2.cvtColor(text_image, cv2.COLOR_BGR2GRAY)

            # get result
            text = recognition_model.recognize(text_image)

            # add text label
            position = vertex[1] - (0, 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1.0
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(image, text, position, font, scale, color, thickness, cv2.LINE_AA)

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
