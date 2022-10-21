from game import Game
from video_get import VideoGet
from video_show import VideoShow
import detect_field
import detect_color
import time
import os.path
import keyboard
from os.path import exists
from video_get_from_file import VideoGetFromFile

import cv2


def main():
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Dedicated thread for showing video frames with VideoShow object.
    Main thread serves only to pass frames between VideoGet and
    VideoShow objects/threads.
    """
    _first_frame = True
    file_exists = False
    video_getter = VideoGet(VideoGet.CAMERA_1).start()
    # video_getter = VideoGetFromFile(VideoGetFromFile.VIDEO_FILE).start()
    video_shower = VideoShow(video_getter.frame).start()
    game = Game().start()

    detected_field = detect_field.DetectField()
    detected_color = detect_color.ColorTracker()

    # calibration_image = cv2.imread(r"../Bilder/test/calibration_image.PNG")

    def calibrate_color(frame, ball_position):
        # The user has to put the ball onto the center spot for calibration.
        # A marker cross will appear on the center spot for some time.

        # At the moment the color calibration is done using a fixed image, that has
        # to be cropped to the right size.
        # Therefore, the size of the images from the camera is needed.
        t_end = time.time()  # + 1
        _done = 0
        # When the fixed image is used for calibration, at least one execution is
        # needed to get the size of the camera images
        while time.time() < t_end or not _done:
            x1 = int(round(ball_position[0] - (frame.shape[1] / 20), 0))
            x2 = int(round(ball_position[0] + (frame.shape[1] / 20), 0))
            y1 = int(round(ball_position[1] - (frame.shape[1] / 20), 0))
            y2 = int(round(ball_position[1] + (frame.shape[0] / 20), 0))

            marked_image = frame.copy()
            # draw the marker cross
            cv2.line(marked_image, (x1, int(ball_position[1])), (x2, int(ball_position[1])), (0, 255, 255), 2)
            cv2.line(marked_image, (int(ball_position[0]), y1), (int(ball_position[0]), y2), (0, 255, 255), 2)

            _done = 1

        calibration_image = cv2.cvtColor(cv2.imread(r"./calibration_image.JPG"), cv2.COLOR_BGR2HSV)

        # The initialization is done with only a small part of the image around the center spot.
        x1 = int(round(ball_position[0] - (calibration_image.shape[1] / 10), 0))
        x2 = int(round(ball_position[0] + (calibration_image.shape[1] / 10), 0))
        y1 = int(round(ball_position[1] - (calibration_image.shape[0] / 10), 0))
        y2 = int(round(ball_position[1] + (calibration_image.shape[0] / 10), 0))

        image_crop = calibration_image[y1:y2, x1:x2]

        # cv2.imwrite("croppedImage.jpg", image_crop)

        ball_color = detected_color.calibrate_ball_color(image_crop)
        team1_color = detected_color.calibrate_team_color(image_crop, 1)
        team2_color = detected_color.calibrate_team_color(image_crop, 2)

        return [ball_color, team1_color, team2_color]

    while True:
        if video_getter.stopped or video_shower.stopped:
            video_shower.stop()
            video_getter.stop()
            break

        frame = video_getter.frame

        if _first_frame:
            video_shower.frame = frame
            cv2.putText(frame, "Zur Kalibrierung ausrichten und \"s\" druecken", (430, 800),
                        cv2.FONT_HERSHEY_PLAIN, 3, (30, 144, 255), 2)
            cv2.circle(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)), 20, (30, 144, 255), 1)
            cv2.circle(frame, (int(frame.shape[1] / 2 - 85), int(frame.shape[0] / 2)), 20, (30, 144, 255), 1)
            cv2.circle(frame, (int(frame.shape[1] / 2 + 85), int(frame.shape[0] / 2)), 20, (30, 144, 255), 1)
            if keyboard.is_pressed("s"):
                cv2.imwrite("calibration_image.JPG", frame)
            file_exists = os.path.exists("./calibration_image.JPG")
            if file_exists:
                calibration_image = cv2.imread(r"./calibration_image.JPG")
                detected_field.get_angle(calibration_image)
                detected_field.get_center_scale(calibration_image)
                field = detected_field.calc_field()

                ball_color, team1_color, team2_color = calibrate_color(frame, detected_field.get_var("center"))
                _first_frame = False

        if not _first_frame:
            frame = game.interpret_frame(frame, ball_color, field, team1_color, team2_color)
            video_shower.frame = frame


if __name__ == "__main__":
    main()
