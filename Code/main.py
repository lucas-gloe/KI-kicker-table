from Game import Game
from video_get import VideoGetter
from video_get_from_pic import VideoGetterFromPic
from video_get_from_vid import VideoGetterFromVid
from video_show import VideoShower
from gui import GUI
from detect_color import ColorTracker
from detect_field import FieldDetector
from analysis import Analysis

import detect_field
import detect_color
from tqdm import tqdm
import os.path
import keyboard
import cv2
from configs import video_stream, video_vid, video_pic



def main():
    """
    Main thread serves only to pass frames between VideoGet, game and
    VideoShow objects/threads.
    # TODO: Kommentare der Funktionen erweitern
    """
    first_frame = True
    start_gui = True
    last_frame_id = None

    SCALE_PERCENT = 40  # percent of original size

    if video_stream:
        video_getter = VideoGetter(SCALE_PERCENT, VideoGetter.CAMERA_1).start()
    if video_pic:
        video_getter = VideoGetterFromPic(SCALE_PERCENT, VideoGetterFromPic.PICTURE).start()
    if video_vid:
        video_getter = VideoGetterFromVid(SCALE_PERCENT, VideoGetterFromVid.VID).start()

    video_shower = VideoShower(video_getter.get_frame(), video_getter.get_frame()).start()
    game = Game(SCALE_PERCENT).start()
    #analysis = Analysis().start()
    gui = GUI(game).start()

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
            #video_shower.stop()
            video_getter.stop()
            #gui.stop()
            break

        frame = video_getter.get_frame()
        if first_frame:
            file_exists = os.path.exists("../Code/calibration_image.JPG")
            if file_exists:
                calibration_image = cv2.imread(r"./calibration_image.JPG")
                detected_field.get_angle(calibration_image)
                detected_field.get_center_scale(calibration_image)
                field = detected_field.calc_field()
                ratio_pxcm = detected_field.get_var("ratio_pxcm")
                ball_color, team1_color, team2_color = _calibrate_color(detected_field.get_var("center"))
                first_frame = False
            else:
                out_frame = detected_field.draw_calibration_marker(frame)
                cv2.waitKey(1)
                if keyboard.is_pressed("s"):
                    cv2.imwrite("./calibration_image.JPG", out_frame)

        if not first_frame:
            out_frame, ball_mask = game.interpret_frame(frame, ball_color, field, team1_color, team2_color, ratio_pxcm)
            gui.frame = out_frame

        video_shower.frames_to_show.append(out_frame)
        video_shower.balls_to_show.append(ball_mask)
        if len(video_getter.frames_to_process) > 1:
            video_getter.frames_to_process.pop(0)

        print(len(video_getter.frames_to_process), "getter")
        print(len(video_shower.frames_to_show), "shower")

if __name__ == "__main__":
    main()
