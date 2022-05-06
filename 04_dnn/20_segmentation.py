# ------------------------------------------------------------------------------------
# Segmentaion (DeepLab v3)
# make sure download DNN Model from support page
# $ cd 04_dnn/model
# $ wget https://gihyo.jp/assets/files/book/2022/978-4-297-12775-6/download/7.5.zip
# $ wget https://github.com/NVIDIA/DIGITS/blob/master/examples/semantic-segmentation/pascal-voc-classes.txt
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
    directory = './model/7.5/deeplab-v3'

    # load DNN from file
    weights = os.path.join(directory, "optimized_graph_voc.pb")
    # weights = os.path.join(directory, "optimized_graph_cityscapes.pb")
    model = cv2.dnn_SegmentationModel(weights)

    # load classname & colorlist
    name_file = os.path.join('./', "pascal-voc-classes.txt")
    classes = read_classes(name_file)
    colors = get_colors(len(classes))
    # set background color black
    if len(classes) > 0:
        colors[0] = (0, 0, 0)

    # set params
    scale = 1.0 /255.0
    size = (513, 513)         # VOC
    # size = (2049, 1025)     # CityScapes
    mean = (127.5, 127.5, 127.5)
    swap = True
    crop = False
    model.setInputParams(scale, size, mean, swap, crop)

    pTime = 0
    cTime = 0

    while True:
        result, image = capture.read()
        if result is False:
            cv2.waitKey(0)
            break

        # get maskdata(each pixel has own class id)
        mask = model.segment(image)

        # convert list(mask) to ndarray(color table. same format to image)
        color_mask = np.array(colors, dtype=np.uint8)[mask]

        # expand mask to input image size
        height, width, _ = image.shape
        color_mask = cv2.resize(color_mask, (width, height), cv2.INTER_NEAREST)

        # Alpha brending
        alpha = 0.5
        beta = 1.0 - alpha
        gamma = 0.0
        image = cv2.addWeighted(image, alpha, color_mask, beta, gamma)

        # Frame rate
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(image, 'PRESS q to exit.', (12, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.putText(image,  f'FPS: {int(fps)}', (12, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.imshow('segmentation', image)
        # cv2.imshow('mask', color_mask)

        key = cv2.waitKey(10)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    dnn_main()
