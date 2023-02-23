import cv2
import keyboard
import os.path

import configs
import frame_postprocessing
import frame_rendering


def gui_handle(window, result_queue, user_input, game_config, total_game_results, game_flags,
               current_game_results):
    """
    update variables and gui related output data
    Parameters:
        window(obj): gui window object
        result_queue(queue): frame, frame id and frame results after postprocessing
        user_input(string): break criteria for loop
        game_config(dict): calibration values for current game
        total_game_results(list): time related total game results per game
        game_flags(dict): flag values for current game
        current_game_results(dict): time related interpretation results for each game
    Returns:
    """
    while True:
        if game_flags['manual_mode']:
            check_variables(user_input, game_flags)
            if game_flags['one_iteration']:
                frame, current_result, expect_id = result_queue.get()
                check_variables(user_input, game_flags)
                frame_postprocessing.reset_game(current_game_results, total_game_results, game_flags)

                out_frame = render_game(frame, current_result, game_config, game_flags)

                event, values = window.read(timeout=1)
                if current_result is not None:
                    window = update_gui(window, game_config, event, total_game_results, current_game_results,
                                        current_result, out_frame, expect_id)
                window.Refresh()

                if keyboard.is_pressed("q"):  # quit the program
                    user_input.value = ord('q')
                    cv2.destroyAllWindows()
                    print("Gui stopped")
                    break

                if keyboard.is_pressed("s"):  # safe configuration image
                    cv2.imwrite("./calibration_image.JPG", frame)

                game_flags['one_iteration'] = False

        elif not game_flags['manual_mode']:
            frame, current_result, expect_id = result_queue.get()
            check_variables(user_input, game_flags)
            frame_postprocessing.reset_game(current_game_results, total_game_results, game_flags)

            out_frame = render_game(frame, current_result, game_config, game_flags)

            event, values = window.read(timeout=1)

            if current_result is not None:
                window = update_gui(window, game_config, event, total_game_results, current_game_results,
                                    current_result, out_frame, expect_id)
            window.Refresh()

            if keyboard.is_pressed("q"):  # quit the program
                user_input.value = ord('q')
                cv2.destroyAllWindows()
                print("Gui stopped")
                break

            if keyboard.is_pressed("s"):  # safe configuration image
                if not os.path.exists(r"./calibration_image.JPG"):
                    cv2.imwrite("./calibration_image.JPG", frame)


def render_game(frame, results, game_config, game_flags):
    """
    draw all tracked objects on frame
    Parameters:
        frame(np.ndarray):frame from interpretations
        results(dict): frame results after interpretation
        game_config(dict): calibration values for current game
        game_flags(dict): flag values for current game
    Return:
        frame(np.ndarray):frame with renderings
    """
    if results is None:
        return frame_rendering.draw_calibration_marker(frame)
    else:
        if game_flags['show_kicker']:
            frame = frame_rendering.draw_field_calibrations(frame, game_config)
        if game_flags['show_objects']:
            frame = frame_rendering.draw_ball(frame, results)
            frame = frame_rendering.draw_predicted_ball(frame, results)
            frame = frame_rendering.draw_figures(frame, results, 'team1_positions', 'team1_on_field', 1,
                                                 'ranks_team1')
            frame = frame_rendering.draw_figures(frame, results, 'team2_positions', 'team2_on_field', 2,
                                                 'ranks_team2')
        return frame


def check_variables(user_input, game_flags):
    """
    check key bindings for user input
    Parameters:
         user_input(string): break criteria for loop
         game_flags(dict): flag values for current game
    Returns:
    """
    variables = {
        'c': ('show_kicker', True),
        'f': ('show_kicker', False),
        'a': ('show_objects', True),
        'd': ('show_objects', False),
        'n': ('new_game', True),
        'm': ('manual_mode', True),
        'l': ('manual_mode', False),
        'k': ('one_iteration', True)
    }
    for key, (flag, value) in variables.items():
        if keyboard.is_pressed(key):
            user_input.value = ord(key)
            if user_input.value == ord(key):
                game_flags[flag] = value


def update_gui(window, game_config, event, total_game_results, current_game_results, current_result, frame, expect_id):
    """
    update values on gui window
    Parameters:
        window(obj): gui window object
        game_config(dict): calibration values for current game
        event()
        total_game_results(list): time related total game results per game
        current_game_results(dict): time related interpretation results for each game
        current_result(dict): frame results after interpretation
        frame(np.ndarray):frame with renderings
        expect_id(int): frame id
    Returns:
        window(obj): gui window object
    """
    if expect_id % 2 == 0:
        window["-frame-"].update(data=cv2.imencode('.ppm', frame)[1].tobytes())

    window["-team_1-"].update(background_color=game_config['gui_team1_color'])
    window["-team_2-"].update(background_color=game_config['gui_team2_color'])
    window["-ball-"].update(background_color=game_config['gui_ball_color'])
    window["-score_team_1-"].update(current_game_results['counter_team1'])
    window["-score_team_2-"].update(current_game_results['counter_team2'])
    window["-fps-"].update(int(current_result['fps']))
    window["-last_game_team1-"].update(total_game_results[-1][0])
    window["-last_game_team2-"].update(total_game_results[-1][1])

    if len(total_game_results) > 1:
        window["-second_last_game_team1-"].update(total_game_results[-2][0])
        window["-second_last_game_team2-"].update(total_game_results[-2][1])
    if len(total_game_results) > 2:
        window["-third_last_game_team1-"].update(total_game_results[-3][0])
        window["-third_last_game_team2-"].update(total_game_results[-3][1])

    if os.path.exists("calibration_image.JPG"):
        window["-config_img-"].update("configuration image saved!")

    if event in ["-manual_game_counter_team_1_up-", "-manual_game_counter_team_2_up-"]:
        if event == "-manual_game_counter_team_1_up-":
            team = "counter_team1"
        else:
            team = "counter_team2"
        current_game_results[team] += 1
        if team == "counter_team1":
            window["-score_team_1-"].update(current_game_results[team])
        else:
            window["-score_team_2-"].update(current_game_results[team])

    if event in ["-manual_game_counter_team_1_down-", "-manual_game_counter_team_2_down-"]:
        if event == "-manual_game_counter_team_1_down-":
            team = "counter_team1"
        else:
            team = "counter_team2"
        if current_game_results[team] > 0:
            current_game_results[team] -= 1
        if team == "counter_team1":
            window["-score_team_1-"].update(current_game_results[team])
        else:
            window["-score_team_2-"].update(current_game_results[team])

    return window
