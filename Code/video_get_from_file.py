from threading import Thread

import cv2


class VideoGetFromFile:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    # VIDEO_FILE = "../Video/1080p/120fps/lang.MP4"
    # VIDEO_FILE = "../Video/720p/60fps/60fps.MOV"
    VIDEO_FILE = "../Video/1080p/240fps/240 länger.MP4"
    PICTURE = "../Code/calibration_image.JPG"
    def __init__(self, src=PICTURE):

        self.stream = cv2.imread("../Code/calibration_image.JPG")
        #self.stream = cv2.imread("../Video/1080p/240fps/240 länger.MP4")

        self.frame = self.stream
        # if self.frame == self.stream:
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
