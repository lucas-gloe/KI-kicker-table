import cv2
import numpy as np


class KalmanFilter:
    """
    Load the kalman filter for predicition purposess and use it to predict the next position of the ball
    Source: VisualComputer, https://www.youtube.com/watch?v=67jwpv49ymA
    """
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    @classmethod
    def predict(cls, coordX, coordY):
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        cls.kf.correct(measured)
        predicted = cls.kf.predict()
        x, y = int(predicted[0]), int(predicted[1])
        tuple = (x, y)
        return tuple
