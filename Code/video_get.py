import cv2
from threading import Thread


class VideoGetter:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """
    # constants
    DEFAULT_CAM = 0
    CAMERA_1 = 1

    def __init__(self, scale_percent, src=CAMERA_1):
        """
        initialize camera stream and resize the full hd resolution with a certain scale
        """
        self.count = 0
        self.frames_to_process = []

        self.SCALE_PERCENT = scale_percent  # percent of original size

        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # heigth of the frame
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # width of the frame
        self.stream.set(cv2.CAP_PROP_FPS, 240)  # FPS output from camera

        (self.grabbed, frame) = self.stream.read()

        # resizing frame for speed optimisation
        width = int(frame.shape[1] * self.SCALE_PERCENT / 100)
        height = int(frame.shape[0] * self.SCALE_PERCENT / 100)
        self.dim = (width, height)

        self.frame = cv2.resize(frame, self.dim, interpolation=cv2.INTER_AREA)
        self.frames_to_process.append(self.frame)

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
            (self.grabbed, frame) = self.stream.read()
            if not self.grabbed:
                self.stop()
            else:
                self.frame = cv2.resize(frame, self.dim, interpolation=cv2.INTER_AREA)
                self.frames_to_process.append(self.frame)

    def get_frame(self):
        return self.frames_to_process[0]

    def stop(self):
        """
        stop gabbing frames from camera
        """
        self.stopped = True
