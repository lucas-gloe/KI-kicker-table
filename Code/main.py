from game import Game
from video_get import VideoGetter
from video_show import VideoShower
from video_get_from_file import VideoGetterFromFile
from gui import GUI
from detect_color import ColorTracker
from detect_field import FieldDetector

import detect_field
import detect_color
from tqdm import tqdm
import os.path
import keyboard
import cv2


def main():
    """
    Dedicated thread for g
    howing video frames with VideoShow object.
    Main thread serves only to pass frames between VideoGet and
    VideoShow objects/threads.
    """
    first_frame = True
    start_gui = True

    SCALE_PERCENT = 25  # percent of original size

    video_getter = VideoGetter(SCALE_PERCENT, VideoGetter.CAMERA_1).start()
    #video_getter = VideoGetterFromFile(SCALE_PERCENT, VideoGetterFromFile.PICTURE).start()
    video_shower = VideoShower(video_getter.frame, video_getter.frame).start()
    game = Game(SCALE_PERCENT).start()

    detected_field = detect_field.FieldDetector(SCALE_PERCENT)
    detected_color = detect_color.ColorTracker(SCALE_PERCENT)

    def _calibrate_color(ball_position_at_center_point):
        """
        The user has to put the ball onto the center spot for calibration. the taken image will be used to read the colors from the marked positions.
        :param_type: calibration img, array
        :return: ball color, team colors
        """

        calibration_image = cv2.cvtColor(cv2.imread(r"./calibration_image.JPG"), cv2.COLOR_BGR2HSV)

        # The initialization is done with only a small part of the image around the center spot.
        x1 = int(round(ball_position_at_center_point[0] - (calibration_image.shape[1] / 10), 0))
        x2 = int(round(ball_position_at_center_point[0] + (calibration_image.shape[1] / 10), 0))
        y1 = int(round(ball_position_at_center_point[1] - (calibration_image.shape[0] / 10), 0))
        y2 = int(round(ball_position_at_center_point[1] + (calibration_image.shape[0] / 10), 0))

        # x1 = int(round(calibration_image.shape[1]/2 - (calibration_image.shape[1] / 10), 0))
        # x2 = int(round(calibration_image.shape[1]/2 + (calibration_image.shape[1] / 10), 0))
        # y1 = int(round(calibration_image.shape[0]/2 - (calibration_image.shape[0] / 10), 0))
        # y2 = int(round(calibration_image.shape[0]/2 + (calibration_image.shape[0] / 10), 0))

        image_crop = calibration_image[y1:y2, x1:x2]

        cv2.imwrite("cropped_calibration_img.JPG", image_crop)

        ball_color = detected_color.calibrate_ball_color(image_crop)
        team1_color = detected_color.calibrate_team_color(image_crop, 1)
        team2_color = detected_color.calibrate_team_color(image_crop, 2)

        return [ball_color, team1_color, team2_color]

    def generator():
        while True:
            yield

    for _ in tqdm(generator()): # while True loop in tqdm generator to check the fps counter
        if video_getter.stopped or video_shower.stopped:
            video_shower.stop()
            video_getter.stop()
            gui.stop()
            break

        frame = video_getter.frame

        if first_frame:
            file_exists = os.path.exists("./calibration_image.JPG")
            if file_exists:
                calibration_image = cv2.imread(r"./calibration_image.JPG")
                detected_field.get_angle(calibration_image)
                detected_field.get_center_scale(calibration_image)
                field = detected_field.calc_field()
                ratio_pxcm = detected_field.get_var("ratio_pxcm")
                ball_color, team1_color, team2_color = _calibrate_color(detected_field.get_var("center"))
                first_frame = False
            if first_frame:
                video_shower.frame = frame
                cv2.circle(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)), int(18*SCALE_PERCENT/100), (30, 144, 255), 1)
                cv2.circle(frame, (int(frame.shape[1] / 2 - int(85*SCALE_PERCENT/100)), int(frame.shape[0] / 2)), int(18*SCALE_PERCENT/100), (30, 144, 255), 1)
                cv2.circle(frame, (int(frame.shape[1] / 2 + int(85*SCALE_PERCENT/100)), int(frame.shape[0] / 2)), int(18*SCALE_PERCENT/100), (30, 144, 255), 1)
                if keyboard.is_pressed("s"):
                    cv2.imwrite("./calibration_image.JPG", frame)

        if not first_frame:
            frame, ball_mask = game.interpret_frame(frame, ball_color, field, team1_color, team2_color, ratio_pxcm)
            if start_gui:
                gui = GUI(game).start()
                start_gui = False
            if not start_gui:
                gui.frame = frame
            video_shower.ball_mask = ball_mask
            video_shower.frame = frame

if __name__ == "__main__":
    main()
