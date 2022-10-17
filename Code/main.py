from game import Game
from video_get import VideoGet
from video_show import VideoShow
from video_get_from_file import VideoGetFromFile


def main():
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Dedicated thread for showing video frames with VideoShow object.
    Main thread serves only to pass frames between VideoGet and
    VideoShow objects/threads.
    """
    _first_frame = True
    video_getter = VideoGet(VideoGet.DEFAULT_CAM).start()
    # video_getter = VideoGetFromFile(VideoGetFromFile.VIDEO_FILE).start()
    video_shower = VideoShow(video_getter.frame).start()
    game = Game().start()

    while True:
        if video_getter.stopped or video_shower.stopped:
            video_shower.stop()
            video_getter.stop()
            break

        frame = video_getter.frame

        if _first_frame:
            # DetectField.get_angle(CalibrationImage)
            # DetectField.get_center_scale(CalibrationImage)
            # Field = DetectField.calc_field()
            # DetectField.calc_goal_area()
            game.region_of_interest(frame)
            _first_frame = False

        frame = game.interpret_frame(frame)

        video_shower.frame = frame
        # cps.increment()


if __name__ == "__main__":
    main()
