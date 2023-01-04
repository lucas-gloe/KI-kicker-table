import cv2
from threading import Thread


class VideoShower:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, frame=None, ball_mask=None):
        """
        initialize new grabbed camera frame
        """
        self.frames_to_show = [frame]
        self.balls_to_show = [ball_mask]

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
            cv2.imshow("Video", self.frames_to_show[0])
            cv2.imshow("ball", self.balls_to_show[0])
            if len(self.frames_to_show) > 1:
                self.frames_to_show.pop(0)
                self.balls_to_show.pop(0)

            if cv2.waitKey(1) == ord("q"):
                self.stopped = True
            cv2.waitKey(0)

    def stop(self):
        """
        stop showing the grabbed frame to the screen
        """
        self.stopped = True
