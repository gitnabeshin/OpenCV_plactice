# ------------------------------------------------------------------------------------
# Object Detection (YOLO v4)
# make sure download DNN Model from support page
# $ cd 04_dnn/model
# $ wget https://gihyo.jp/assets/files/book/2022/978-4-297-12775-6/download/7.3.zip
# $ wget https://github.com/pjreddie/darknet/blob/master/data/coco.names
# ------------------------------------------------------------------------------------

import os
import numpy as np
import cv2
import time

def read_classes(file):
    classes = None
    with open(file, mode='r', encoding="utf-8") as f:
        classes = f.read().splitlines()
    return classes

def get_colors(num):
    colors= []
    np.random.seed(0)
    for i in range(num):
        color = np.random.randint(0, 256, [3]).tolist()
        colors.append(color)
    return colors

def dnn_main():

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise IOError("can't open camera")

    # directory = os.path.dirname(__file__)
    directory = './model/7.3/scaled-yolov4'

    # load DNN from file
    weights = os.path.join(directory, "yolov4-csp.weights")
    config = os.path.join(directory, "yolov4-csp.cfg")
    # weights = os.path.join(directory, "yolov4-p5.weights")
    # config = os.path.join(directory, "yolov4-p5.cfg")
    # weights = os.path.join(directory, "yolov4-p6.weights")
    # config = os.path.join(directory, "yolov4-p6.cfg")
    # weights = os.path.join(directory, "yolov4x-mish.weights")
    # config = os.path.join(directory, "yolov4x-mish.cfg")
    model = cv2.dnn_DetectionModel(weights, config)

    # load classname & colorlist
    name_file = os.path.join('./', "coco.names")
    classes = read_classes(name_file)
    colors = get_colors(len(classes))

    # set params
    scale = 1.0 /255.0
    size = (512, 512)      # yolov4-csp
    # size = (640, 640)    # yolov4-p5
    # size = (896, 896)    # yolov4-p6
    # size = (1280, 1280)  # yolov4x-mish
    mean = (0.00, 0.0, 0.0)
    swap = True
    crop = False
    model.setInputParams(scale, size, mean, swap, crop)
    # proccess NMS each Class
    model.setNmsAcrossClasses(False)

    pTime = 0
    cTime = 0

    while True:
        result, image = capture.read()
        if result is False:
            cv2.waitKey(0)
            break

        # face detection
        confidence_threshold = 0.6
        # nms:Non-Maximum Suppression (0.4: intersection rate of 2 candidate region)
        nms_threshold = 0.4
        class_ids, confidences, boxes = model.detect(image, confidence_threshold, nms_threshold)

        # draw bounding box
        for class_id, confidence, box in zip(class_ids, confidences, boxes):
            thickness = 2
            color = colors[class_id]
            cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)
            cv2.putText(image, f'{classes[class_id]}({confidence:.3f})', (box[0], box[1] - 5), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

        # Frame rate
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(image, 'PRESS q to exit.', (12, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.putText(image,  f'FPS: {int(fps)}', (12, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.imshow('object_detection', image)

        key = cv2.waitKey(10)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    dnn_main()
