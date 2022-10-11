import cv2
from VideoGet import VideoGet
from VideoShow import VideoShow
from Game import Game


def main():
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Dedicated thread for showing video frames with VideoShow object.
    Main thread serves only to pass frames between VideoGet and
    VideoShow objects/threads.
    """

    video_getter = VideoGet(1).start()
    video_shower = VideoShow(video_getter.frame).start()
    game = Game().start()

    while True:
        if video_getter.stopped or video_shower.stopped:
            video_shower.stop()
            video_getter.stop()
            break

        frame = video_getter.frame
        #tracked_frame = frame.copy()
        #putIterationsPerSec(tracked_frame, cps.countsPerSec())
        frame = game.interpretFrame(frame)

        video_shower.frame = frame
        #cps.increment()

if __name__ == "__main__":
    main()