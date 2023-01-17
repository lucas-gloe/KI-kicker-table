from threading import Thread

import cv2


class VideoGetterFromPic:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """
    
    PICTURE = "./calibration_image.JPG"

    def __init__(self, scale_factor, src=PICTURE):
        """

        """
        self.count = 0
        self.frames_to_process = []

        self.stream = cv2.imread("./calibration_image.JPG")
        self.frame = self.stream
        self.frames_to_process.append(self.frame)

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
                self.frame = self.stream
                self.frames_to_process.append(self.frame)
                self.stopped = False

    def get_frame(self):
        return self.frames_to_process[0]

    def stop(self):
        """

        """
        self.stopped = True
        self.stream.release()
