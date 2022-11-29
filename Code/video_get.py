import cv2
from threading import Thread


class VideoGetter:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """
    # constances
    DEFAULT_CAM = 0
    CAMERA_1 = 1

    def __init__(self, scale_percent, src=CAMERA_1):
        """
        initialize camera stream and resize the full hd resolution with a certain scale
        """

        self.SCALE_PERCENT = scale_percent  # percent of original size

        # for camera uses

        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # heigth of the frame
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # width of the frame
        self.stream.set(cv2.CAP_PROP_FPS, 240)  # FPS output from camera

        (self.grabbed, frame) = self.stream.read()

        width = int(frame.shape[1] * self.SCALE_PERCENT / 100)
        height = int(frame.shape[0] * self.SCALE_PERCENT / 100)
        self.dim = (width, height)

        self.frame = cv2.resize(frame, self.dim, interpolation=cv2.INTER_AREA)

        self.stopped = False

    def start(self):
        """
        start the camerathread for grabbing the frame
        :return: thread properties
        """
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        """
        get new frame from camerastream
        """
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, frame) = self.stream.read()

                self.frame = cv2.resize(frame, self.dim, interpolation=cv2.INTER_AREA)

    def stop(self):
        """
        stop gabbing frames from camera
        """
        self.stopped = True
