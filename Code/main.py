from game import Game
from video_get import VideoGet
from video_show import VideoShow


def main():
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Dedicated thread for showing video frames with VideoShow object.
    Main thread serves only to pass frames between VideoGet and
    VideoShow objects/threads.
    """
    _first_frame = True
    video_getter = VideoGet(VideoGet.CAMERA_1).start()
    #video_getter = VideoGetFromFile(VideoGetFromFile.VIDEO_FILE).start()
    video_shower = VideoShow(video_getter.frame).start()
    game = Game().start()

    while True:
        if video_getter.stopped or video_shower.stopped:
            video_shower.stop()
            video_getter.stop()
            break

        frame = video_getter.frame

        if _first_frame:
            game.region_of_interest(frame)
            _first_frame = False

        frame = game.interpret_frame(frame)

        video_shower.frame = frame
        #cps.increment()

if __name__ == "__main__":
    main()