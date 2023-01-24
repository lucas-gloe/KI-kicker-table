import cv2
import multiprocessing
import time
import numpy as np
import os.path
from gui import gui_handle
from calibration import calibrate, calibrate_before_first_image
import configs
import frame_preprocessing
import frame_postprocessing


def preprocess_frame(frame_queue, preprocessed_queue, user_input, game_config, game_flags):
    # resizing frame for speed optimisation
    width = int(1920 * configs.SCALE_FACTOR)
    height = int(1080 * configs.SCALE_FACTOR)
    dim = (width, height)
    start_time = None
    if start_time is None:
        start_time = time.time()
        start_time -= 0.001

    while True:
        # start_time_running1 = time.time()
        # start_time_running = time.time()
        frame_id, frame = frame_queue.get()
        # print("one frame iteration take from frame_queue ", (time.time() - start_time_running))
        # start_time_running = time.time()
        preprocessing_result, resized_frame = preprocessing_action(frame, game_config, dim, game_flags)
        # print("one frame iteration total preprocessing ", (time.time() - start_time_running))
        # start_time_running = time.time()
        preprocessed_queue.put((frame_id, resized_frame, preprocessing_result))
        # print("one frame iteration put into preprocessed_queue ", (time.time() - start_time_running))
        if user_input.value == ord('q'):
            print("Worker stopped")
            break
        # print("total time preprocessing FPS:", frame_id / (time.time() - start_time))
        # print("total time preprocessing per frame", (time.time() - start_time_running1))
        # print("")


def preprocessing_action(frame, game_config, dim, game_flags):
    # start_time_running = time.time()
    analysis_results = []
    resized_frame = cv2.resize(frame, dim)
    # print("one frame iteration rezising ", (time.time() - start_time_running))
    # start_time_running = time.time()
    hsv_img = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)
    # print("one frame iteration hsv_color ", (time.time() - start_time_running))
    # start_time_running = time.time()
    ball_position = frame_preprocessing.define_balls_position(hsv_img, game_config, game_flags)
    # print("one frame iteration ball_position ", (time.time() - start_time_running))
    # start_time_running = time.time()
    team_1_positions, team1_on_field, ranks_team1 = frame_preprocessing.define_players_position(hsv_img,
                                                                                                game_config,
                                                                                                "team1_color", 1)
    # print("one frame iteration team_1_positions ", (time.time() - start_time_running))
    # start_time_running = time.time()

    team_2_positions, team2_on_field, ranks_team2 = frame_preprocessing.define_players_position(hsv_img,
                                                                                                game_config,
                                                                                                "team2_color", 2)
    # print("one frame iteration team_2_positions ", (time.time() - start_time_running))
    analysis_results.append(
        (ball_position, team_1_positions, team_2_positions, team1_on_field, team2_on_field, ranks_team1, ranks_team2))
    # print(analysis_results[0][0])
    return analysis_results, resized_frame


def update_game(preprocessed_queue, result_queue, user_input, game_config, ball_positions, game_flags,
                current_game_results):
    # get start time
    start_time = None
    frame_dict = {}
    game = {}
    expect_id = 0
    current_result = {}
    while True:
        frame_id, frame, preprocessing_result = preprocessed_queue.get()
        if user_input.value == ord('q'):
            print("Game stopped")
            break
        if start_time is None:
            start_time = time.time()
            start_time -= 0.001
        frame_dict[frame_id] = (frame, preprocessing_result)
        while (expect_id in frame_dict):
            # start_time_2 = time.time()
            current_frame, current_preprocessing_result = frame_dict[expect_id]
            # print(current_preprocessing_result[0][0])
            fps = expect_id / (time.time() - start_time)

            current_result['fps'] = fps
            current_result['ball_position'] = current_preprocessing_result[0][0]
            current_result['team1_positions'] = current_preprocessing_result[0][1]
            current_result['team2_positions'] = current_preprocessing_result[0][2]
            current_result['team1_on_field'] = current_preprocessing_result[0][3]
            current_result['team2_on_field'] = current_preprocessing_result[0][4]
            current_result['ranks_team1'] = current_preprocessing_result[0][5]
            current_result['ranks_team2'] = current_preprocessing_result[0][6]

            predicted_ball_position = frame_postprocessing.predict_ball(ball_positions, game_flags)

            if predicted_ball_position is not [-1, -1]:
                ball_positions.append(predicted_ball_position)
            else:
                ball_positions.append(current_result['ball_position'])
            # print(ball_positions)
            frame_postprocessing.count_game_score(ball_positions, game_config, current_game_results, game_flags)

            frame_postprocessing.detect_ball_reentering(ball_positions, game_config, game_flags)

            # ball_positions.append(current_result['ball_position'])

            result_queue.put((current_frame, current_result, expect_id))
            # print(current_result)
            del frame_dict[expect_id]
            expect_id += 1

            # print("total time update game", time.time() - start_time_2)
            # print("")


if __name__ == '__main__':
    print("Starting...")

    print("setting up queues")
    # create a queue for frames
    frame_queue = multiprocessing.Queue()
    # create a queue for preprocessed, not sorted frames
    preprocessed_queue = multiprocessing.Queue()
    # create a queue for finished processed and sorted frames
    result_queue = multiprocessing.Queue()

    print("setting up shared memory inputs")
    user_input = multiprocessing.Value('i', ord('A'))

    manager = multiprocessing.Manager()
    game_config = manager.dict()

    manager2 = multiprocessing.Manager()
    total_game_results = manager2.list()
    total_game_results.append([0, 0])

    manager3 = multiprocessing.Manager()
    ball_positions = manager3.list()
    ball_positions.append([-1, -1])

    manager4 = multiprocessing.Manager()
    game_flags = manager4.dict()
    game_flags['show_objects'] = True
    game_flags['show_kicker'] = False
    game_flags['new_game'] = False
    game_flags['_ball_reenters_game'] = False
    game_flags['_goal1_detected'] = False
    game_flags['_goal2_detected'] = False
    game_flags['predicted_value_added'] = False
    game_flags['ball_was_found'] = False
    game_flags["goalInCurrentFrame"] = False

    manager5 = multiprocessing.Manager()
    current_game_results = manager5.dict()
    current_game_results['counter_team1'] = 0
    current_game_results['counter_team2'] = 0

    print("initialize GUI")
    window = calibrate_before_first_image()

    print("Start Video Capture")
    cap = cv2.VideoCapture(configs.source)  # open camera/videostream
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # heigth of the frame
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # width of the frame
    cap.set(cv2.CAP_PROP_FPS, 240)  # set camera frame rate to 240 fps

    # print(cap.get(cv2.CAP_PROP_FPS))

    # start 4 worker processes to process the frames
    num_workers = 4
    for i in range(num_workers):
        preprocessing_worker = multiprocessing.Process(target=preprocess_frame,
                                                       args=(frame_queue, preprocessed_queue, user_input, game_config, game_flags))
        preprocessing_worker.start()
    postprocessing_worker = multiprocessing.Process(target=update_game,
                                                    args=(preprocessed_queue, result_queue, user_input, game_config,
                                                          ball_positions, game_flags, current_game_results))
    postprocessing_worker.start()
    gui_worker = multiprocessing.Process(target=gui_handle,
                                         args=(window, result_queue, user_input, game_config, total_game_results,
                                               ball_positions, game_flags, current_game_results))
    gui_worker.start()

    frame_id = 0
    print("Start processing")

    calibrated = False

    while True:
        # read a frame from the camera
        ret, frame = cap.read()
        # add the frame to the queue for processing
        if calibrated == False:
            file_exists = os.path.exists(r"./calibration_image.JPG")
            if not file_exists:
                width = int(1920 * configs.SCALE_FACTOR)
                height = int(1080 * configs.SCALE_FACTOR)
                dim = (width, height)
                resized_frame = cv2.resize(frame, dim)
                print("file not found, set calibration file by pressing S")
                result_queue.put((resized_frame, None, None))
            else:
                print("start calibration")
                calibration_image = cv2.imread(r"./calibration_image.JPG")
                calibrate(calibration_image, game_config)
                print("finished calibration")
                calibrated = True

        if calibrated == True:

            for i in range(1):  # simulate 240fps with 30fps camera
                frame_queue.put((frame_id, frame))
                frame_id += 1

        if user_input.value == ord('q'):
            for i in range(num_workers):  # put last frames for workers in queue
                frame_queue.put((frame_id, frame))
                frame_id += 1
            print("main stopped")

            break
