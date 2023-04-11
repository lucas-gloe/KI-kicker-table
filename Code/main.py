import cv2
import multiprocessing
import time
import os.path
from gui import gui_handle
from calibration import calibrate, calibrate_before_first_image
import configs
import frame_preprocessing
import frame_postprocessing


def preprocess_frame(frame_queue, preprocessed_queue, user_input, game_config, game_flags):
    """
    worker handles preprocessing the frame by defining objects positions on frame
    Parameters:
        frame_queue(queue):frames before preprocessing
        preprocessed_queue(queue): frames after preprocessing
        user_input(string): break criteria for loop
        game_config(dict): calibration values for current game
        game_flags(dict): flag values for current game
    Returns:
    """
    # resizing frame for speed optimisation
    width = int(1920 * configs.SCALE_FACTOR)
    height = int(1080 * configs.SCALE_FACTOR)
    dim = (width, height)

    while True:
        frame_id, frame = frame_queue.get()
        # actual preprocessing
        preprocessing_result, resized_frame = _preprocessing_action(frame, game_config, dim, game_flags)
        preprocessed_queue.put((frame_id, resized_frame, preprocessing_result))
        if user_input.value == ord('q'):
            print("Worker stopped")
            break


def _preprocessing_action(frame, game_config, dim, game_flags):
    """
    actual preprocessing actions for preprocessing on frame
    Parameters:
        frame(np.ndarray):frame for preprocessing
        game_config(dict): calibration values for current game
        dim(int): parameter for resizing
        game_flags(dict): flag values for current game
    Returns:
        analysis_results(list): parallel interpretation results
        resized_frame(np.array): processed resized frame
    """
    analysis_results = []
    resized_frame = cv2.resize(frame, dim)
    hsv_img = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)
    ball_position = frame_preprocessing.define_balls_position(hsv_img, game_config, game_flags)
    team_1_positions, team1_on_field, ranks_team1 = frame_preprocessing.define_players_position(hsv_img,
                                                                                                game_config,
                                                                                                "team1_color", 1)

    team_2_positions, team2_on_field, ranks_team2 = frame_preprocessing.define_players_position(hsv_img,
                                                                                                game_config,
                                                                                                "team2_color", 2)
    analysis_results.append(
        (ball_position, team_1_positions, team_2_positions, team1_on_field, team2_on_field, ranks_team1, ranks_team2))
    return analysis_results, resized_frame


def update_game(preprocessed_queue, result_queue, user_input, game_config, ball_positions, game_flags,
                current_game_results):
    """
    worker who sorts frames by id and do time-related postprocessess on preprocessed data
    Parameters:
        preprocessed_queue(queue): frame, frame id and frame results after preprocessing
        result_queue(queue): frame, frame id and frame results after postprocessing
        user_input(string): break criteria for loop
        game_config(dict): calibration values for current game
        ball_positions(list): time related ball positions
        game_flags(dict): flag values for current game
        current_game_results(dict): time related interpretation results for each game
    """
    # get start time
    start_time = None
    frame_dict = {}
    predicted_ball_position = [-1, -1]
    expect_id = 0
    current_result = {}
    while True:
        if game_flags['manual_mode']:
            if game_flags['one_iteration']:
                # get current results from preprocessing
                frame_id, frame, preprocessing_result = preprocessed_queue.get()
                if user_input.value == ord('q'):
                    print("Game stopped")
                    break
                # start time counter for FPS tracker
                if start_time is None:
                    start_time = time.time()
                    start_time -= 0.001
                # safe results from preprocessing to dict
                frame_dict[frame_id] = (frame, preprocessing_result)
                # take results from dict ordert by id
                while (expect_id in frame_dict):
                    current_frame, current_preprocessing_result = frame_dict[expect_id]
                    # calculate FPS
                    fps = expect_id / (time.time() - start_time)

                    # add current results from preprocessing to dict for further processing or passing
                    current_result.update({
                        'fps': fps,
                        'ball_position': current_preprocessing_result[0][0],
                        'team1_positions': current_preprocessing_result[0][1],
                        'team2_positions': current_preprocessing_result[0][2],
                        'team1_on_field': current_preprocessing_result[0][3],
                        'team2_on_field': current_preprocessing_result[0][4],
                        'ranks_team1': current_preprocessing_result[0][5],
                        'ranks_team2': current_preprocessing_result[0][6]
                    })

                    if current_result['ball_position'] != [-1, -1]:
                        predicted_ball_position = frame_postprocessing.predict_ball(ball_positions,
                                                                                    current_game_results)
                        ball_positions.append(current_result['ball_position'])
                        game_flags["predicted_value_added"] = False
                    if current_result['ball_position'] == [-1, -1]:
                        if not game_flags["predicted_value_added"]:
                            if predicted_ball_position[0] - 15 < ball_positions[-1][0] < predicted_ball_position[
                                0] + 15:
                                ball_positions.append(predicted_ball_position)
                                current_result["predicted"] = predicted_ball_position
                                game_flags["predicted_value_added"] = True
                        else:
                            ball_positions.append(current_result['ball_position'])

                    frame_postprocessing.count_game_score(ball_positions, game_config, current_game_results, game_flags)
                    frame_postprocessing.detect_ball_reentering(ball_positions, game_config, game_flags)

                    result_queue.put((current_frame, current_result, expect_id))
                    del frame_dict[expect_id]
                    expect_id += 1

                    game_flags['one_iteration'] = False

        elif not game_flags['manual_mode']:
            overall_start = time.time()
            # get current results from preprocessing
            frame_id, frame, preprocessing_result = preprocessed_queue.get()
            # stop update game
            if user_input.value == ord('q'):
                print("Game stopped")
                break
            # start FPS tracker
            if start_time is None:
                start_time = time.time()
                start_time -= 0.001
            # safe results from preprocessing to dict
            frame_dict[frame_id] = (frame, preprocessing_result)
            # take results from dict ordert by id
            while (expect_id in frame_dict):
                current_frame, current_preprocessing_result = frame_dict[expect_id]
                # calculate FPS
                fps = expect_id / (time.time() - start_time)

                # add current results from preprocessing to dict for further processing or passing
                current_result.update({
                    'fps': fps,
                    'ball_position': current_preprocessing_result[0][0],
                    'team1_positions': current_preprocessing_result[0][1],
                    'team2_positions': current_preprocessing_result[0][2],
                    'team1_on_field': current_preprocessing_result[0][3],
                    'team2_on_field': current_preprocessing_result[0][4],
                    'ranks_team1': current_preprocessing_result[0][5],
                    'ranks_team2': current_preprocessing_result[0][6]
                })

                if current_result['ball_position'] != [-1, -1]:
                    predicted_ball_position = frame_postprocessing.predict_ball(ball_positions,
                                                                                current_game_results)
                    ball_positions.append(current_result['ball_position'])
                    game_flags["predicted_value_added"] = False
                if current_result['ball_position'] == [-1, -1]:
                    if not game_flags["predicted_value_added"]:
                        if predicted_ball_position[0] - 15 < ball_positions[-1][0] < predicted_ball_position[0] + 15:
                            ball_positions.append(predicted_ball_position)
                            current_result["predicted"] = predicted_ball_position
                            game_flags["predicted_value_added"] = True
                    else:
                        ball_positions.append(current_result['ball_position'])

                frame_postprocessing.count_game_score(ball_positions, game_config, current_game_results, game_flags)
                frame_postprocessing.detect_ball_reentering(ball_positions, game_config, game_flags)

                result_queue.put((current_frame, current_result, expect_id))
                del frame_dict[expect_id]
                expect_id += 1


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
    game_flags.update({
        'results': True,
        'show_objects': True,
        'show_kicker': False,
        'new_game': False,
        'manual_mode': False,
        '_ball_reenters_game': True,
        '_goal1_detected': False,
        '_goal2_detected': False,
        'predicted_value_added': False,
        'ball_was_found': False,
        "goalInCurrentFrame": False,
        'one_iteration': True
    })

    manager5 = multiprocessing.Manager()
    current_game_results = manager5.dict()
    current_game_results.update({
        'counter_team1': 0,
        'counter_team2': 0
    })

    print("initialize GUI")
    window = calibrate_before_first_image()

    print("Start Video Capture")
    cap = cv2.VideoCapture(configs.source)  # open camera/videostream
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # height of the frame
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # width of the frame
    cap.set(cv2.CAP_PROP_FPS, 240)  # set camera frame rate to 240 fps


    # start 4 worker processes to process the frames
    num_workers = 4
    for i in range(num_workers):
        preprocessing_worker = multiprocessing.Process(target=preprocess_frame,
                                                       args=(frame_queue, preprocessed_queue, user_input, game_config,
                                                             game_flags))
        preprocessing_worker.start()
    postprocessing_worker = multiprocessing.Process(target=update_game,
                                                    args=(preprocessed_queue, result_queue, user_input,
                                                          game_config, ball_positions, game_flags, current_game_results))
    postprocessing_worker.start()
    gui_worker = multiprocessing.Process(target=gui_handle,
                                         args=(window, result_queue, user_input, game_config, total_game_results,
                                               game_flags, current_game_results))
    gui_worker.start()

    frame_id = 0

    print("Start processing")

    calibrated = False

    while True:
        # read a frame from the camera
        ret, frame = cap.read()
        # add the frame to the queue for processing
        if not calibrated:
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

        if calibrated:
            frame_queue.put((frame_id, frame))
            frame_id += 1
            if configs.source != 0 or configs.source != 1:
                cv2.waitKey(1)

        if user_input.value == ord('q'):
            for i in range(num_workers):  # put last frames for workers in queue
                frame_queue.put((frame_id, frame))
                frame_id += 1
            print("main stopped")
            break
