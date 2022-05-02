# ------------------------------------------------------------------------------------
# Low level API
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
    net = cv2.dnn.readNet(weights, config)

    # Set BACKEND, TARGET for Inference
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    while True:
        result, image = capture.read()
        if result is False:
            cv2.waitKey(0)
            break

        # convert image to blob
        scale = 1.0
        size = (300, 300)
        mean = (104.0, 177.0, 123.0)
        swap = False
        crop = False
        blob = cv2.dnn.blobFromImage(image, scale, size, mean, swap, crop)

        # set blob on input of network
        input_layer = "data"
        net.setInput(blob, input_layer)

        # get output of network (detections)
        output_layers = net.getUnconnectedOutLayersNames()
        detections = net.forward(output_layers)

        # parse result
        boxes = []
        confidences = []
        detections = np.squeeze(np.array(detections))
        # print(detections)
        for detection in detections:

            # print(detection)
            # [0.  1.  0.0821088  0.17108363 0.29118916 0.29362014 0.42872474]

            confidence = float(detection[2])
            confidences.append(confidence)
            x = int(detection[3] * image.shape[1])
            y = int(detection[4] * image.shape[0])
            width = int(detection[5] * image.shape[1]) - x
            height = int(detection[6] * image.shape[0]) - y
            boxes.append((x, y, width, height))

        # integrate dupulication???
        confidence_threshold = 0.3
        nms_threshold = 0.4
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
        # print(indices)
        if len(indices) > 0:
            indices = indices.flatten()
        # print(f'    {indices}')

        # draw bounding  box
        for index in indices:
            box = boxes[index]
            color = (255, 0, 0)
            thickness = 2
            cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)
            cv2.putText(image, f'Confidence={confidences[index]:.3f}', (box[0], box[1] - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        cv2.putText(image, 'PRESS q to exit.', (12, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.imshow('face_detection', image)

        key = cv2.waitKey(10)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    dnn_main()
