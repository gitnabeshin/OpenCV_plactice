# ------------------------------------------------------------------------------------
# KeyPoint Detection(Light Weight OpenPose)
# make sure download DNN Model from support page
# $ cd 04_dnn/model
# $ wget https://gihyo.jp/assets/files/book/2022/978-4-297-12775-6/download/7.7.zip
# ------------------------------------------------------------------------------------

import os
import numpy as np
import cv2
import time

# colors for 18 keypoints of Lightweight OpenPose
def get_colors():
    colors = [(255,   0,   0), (255,  80,   0), (255, 180,   0),
              (255, 255,   0), (180, 255,   0), ( 80, 255,   0),
              (  0, 255,   0), (  0, 255,  80), (  0, 255, 180),
              (  0 ,255, 255), (  0, 180, 255), (  0,  80, 255),
              (  0,   0, 255), ( 85,   0, 255), (180,   0, 255),
              (255,   0, 255), (255,   0, 180), (255,   0,  85)]
    return colors

def dnn_main():

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise IOError("can't open camera")

    # directory = os.path.dirname(__file__)
    directory = './model/7.7/lightweight-openpose'

    # load DNN from file
    weights = os.path.join(directory, "human-pose-estimation.onnx")
    model = cv2.dnn_KeypointsModel(weights)

    # set params
    scale = 1.0 /255.0
    size = (256, 456)
    mean = (128.0, 128.0, 128.0)
    swap = False
    crop = False
    model.setInputParams(scale, size, mean, swap, crop)

    colors = get_colors()

    pTime = 0
    cTime = 0

    while True:
        result, image = capture.read()
        if result is False:
            cv2.waitKey(0)
            break

        rows, cols, _ = image.shape
        size = (256, int((256 / cols) * rows))
        model.setInputSize(size)

        confidence_threshold = 0.6
        keypoints = model.estimate(image, confidence_threshold)

        # draw keypoints
        for index, keypoint in enumerate(keypoints):
            point = tuple(map(int, keypoint.tolist()))
            radius = 5
            color = colors[index]
            thickness = -1
            cv2.circle(image, point, radius, color, thickness, cv2.LINE_AA)

        # fraw bones omit

        # graw heatmap. each heatmap size is (46 * 58).
        heatmaps = model.predict(image)
        heatmaps = np.squeeze(np.array(heatmaps)) # (19, 58, 46) : all heatmap
        heatmap = cv2.resize(heatmaps[-1], (cols, rows))
        heatmap = cv2.applyColorMap(np.uint8(255 * (1.0 - heatmap)), cv2.COLORMAP_JET)

        alpha = 0.5
        beta = 1 - alpha
        blend = np.zeros(image.shape, np.uint8)
        cv2.addWeighted(image, alpha, heatmap, beta, 0.0, blend)

        # Frame rate
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        # cv2.putText(image, 'PRESS q to exit.', (12, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        # cv2.putText(image,  f'FPS: {int(fps)}', (12, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        # cv2.imshow('Keypoint detection', image)
        cv2.putText(blend, 'PRESS q to exit.', (12, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.putText(blend,  f'FPS: {int(fps)}', (12, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.imshow('Keypoint detection', blend)

        key = cv2.waitKey(10)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    dnn_main()
