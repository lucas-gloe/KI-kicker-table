from threading import Thread

import cv2


class VideoGetFromFile:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """
    PICTURE = "../Code/calibration_image.JPG"

    def __init__(self, src=PICTURE):

        self.SCALE_FACTOR = 60  # percent of original size

        self.stream = cv2.imread("../Code/calibration_image.JPG")

        self.frame = self.stream

        width = int(self.frame.shape[1] * self.SCALE_FACTOR / 100)
        height = int(self.frame.shape[0] * self.SCALE_FACTOR / 100)
        dim = (width, height)

        # resize image
        self.frame = cv2.resize(self.frame, dim, interpolation = cv2.INTER_AREA)

        self.grabbed = True

        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                self.stopped, self.frame = self.stream
                self.stopped = True

    def stop(self):
        self.stopped = True
