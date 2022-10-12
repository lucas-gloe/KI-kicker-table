import cv2
from threading import Thread


class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    DEFAULT_CAM = 0
    CAMERA_1 = 1

    def __init__(self, src=DEFAULT_CAM):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) #heigth of the frame
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) #width of the frame
        self.stream.set(cv2.CAP_PROP_FPS, 60) #FPS output from camera

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