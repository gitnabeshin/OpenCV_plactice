import cv2

img = cv2.imread('../img/crab.png', cv2.IMREAD_COLOR)

timer = cv2.TickMeter()

for i in range(3):

    timer.reset()

    timer.start()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    timer.stop()

    measurment = timer.getTimeMilli()

    print(f'measurement_time = {measurment} (ms)')
    print(f'                 = {measurment:.3f} (ms)')
