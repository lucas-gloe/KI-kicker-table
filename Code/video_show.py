import cv2
from threading import Thread


class VideoShower:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, frame=None):
        """
        initialize new grabbed camera frame
        """
        self.frame = frame

        self.stopped = False

    def start(self):
        """
        start the camera thread for showing the frame
        :return: thread properties
        """
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        """
        show the grabbed frame to the screen
        """
        while not self.stopped:
            cv2.imshow("Video", self.frame)
            #cv2.waitKey(0)
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        """
        stop showing the grabbed frame to the screen
        """
        self.stopped = True
