from threading import Thread

import cv2


class VideoGetFromFile:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    VIDEO_FILE = "../Video/720p/30fps/lang.mov"

    def __init__(self, src=VIDEO_FILE):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # height of the frame
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # width of the frame
        # self.stream.set(cv2.CAP_PROP_FPS, 30) #FPS output from camera

        (self.grabbed, self.frame) = self.stream.read()

        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True
