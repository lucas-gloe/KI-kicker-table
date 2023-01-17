from threading import Thread

import cv2


class VideoGetterFromVid:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    #VID = r"C:\Users\gloec\OneDrive\Dokumente\GitHub\KI-kicker-table\OpenCV\Video\1080p\240fps\test_schuss_lang.mov"
    VID = r"C:\Users\gloec\OneDrive\Dokumente\GitHub\KI-kicker-table\OpenCV\Video\1080p\120fps\schuss_schnell.MOV"
    #VID = "../Video/1080p/30fps/test_vid.mov"

    def __init__(self, scale_factor, src=VID):
        """

        """
        self.SCALE_PERCENT = scale_factor  # percent of original size
        self.frames_to_process = []

        self.stream = cv2.VideoCapture(src)

        # read frame
        (self.grabbed, frame) = self.stream.read()

        # resizing frame for speed optimisation
        width = int(frame.shape[1] * self.SCALE_PERCENT / 100)
        height = int(frame.shape[0] * self.SCALE_PERCENT / 100)
        self.dim = (width, height)

        self.frame = cv2.resize(frame, self.dim)

        # TODO: ohne kalibrieung direkt in den shower packen ohne game class
        self.frames_to_process.append(self.frame)

        self.grabbed = True
        self.stopped = False
        self.frame_processed = False

    def start(self):
        """
        start the camerathread for grabbing the frame
        :return: thread properties
        """
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        """
        get new frame from camera stream
        """
        while not self.stopped:
            # TODO: sicherstellen, dass jeder frame auch ein neuer ist
            self.grabbed, frame = self.stream.read()
            if not self.grabbed:
                self.stop()
            else:
                self.frame = cv2.resize(frame, self.dim)
                self.frames_to_process.append(self.frame)
                self.stopped = False

    def get_frame(self):
        return self.frames_to_process[0]

    def stop(self):
        """

        """
        self.stopped = True
        self.stream.release()