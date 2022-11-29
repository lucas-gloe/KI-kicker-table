from threading import Thread

import cv2


class VideoGetterFromFile:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """
    PICTURE = "../Code/calibration_image.JPG"

    def __init__(self, scale_factor, src=PICTURE):
        """

        """

        self.stream = cv2.imread("../Code/calibration_image.JPG")

        self.frame = self.stream



        self.grabbed = True

        self.stopped = False

    def start(self):
        """

        """
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        """

        """
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                self.stopped, self.frame = self.stream
                self.stopped = True

    def stop(self):
        """

        """
        self.stopped = True
