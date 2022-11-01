import cv2
from threading import Thread


class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, frame=None):
        """

        """
        self.frame = frame
        self.stopped = False

    def start(self):
        """

        """
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        """

        """
        while not self.stopped:
            cv2.imshow("Video", self.frame)
            # cv2.waitKey(33) # bei 30fps
            # cv2.waitKey(0)
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        """

        """
        self.stopped = True
