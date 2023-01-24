import cv2
import keyboard
import os.path

import configs
import frame_postprocessing
import frame_rendering
import time

def gui_handle(window, result_queue, user_input, game_config, total_game_results, ball_positions, game_flags, current_game_results):
    while True:
        start_time = time.time()
        new_game = False
        frame, current_result, expect_id = result_queue.get()
        # print(current_result)
        check_variables(user_input, game_flags)
        frame_postprocessing.reset_game(current_game_results, total_game_results, game_flags)

        out_frame = render_game(frame, current_result, game_config, total_game_results, game_flags)

        event, values = window.read(timeout=1)
        if current_result is not None:
            window = update_gui(window, game_config, event, total_game_results, current_game_results)
        window.Refresh()

        # if expect_id % 8 == 0:
        cv2.imshow("Camera", out_frame)
        # cv2.waitKey(1)

        if keyboard.is_pressed("q"):  # quit the program
            user_input.value = ord('q')
            cv2.destroyAllWindows()
            print("Gui stopped")
            break

        if keyboard.is_pressed("s"):  # safe configuration image
            cv2.imwrite("./calibration_image.JPG", frame)

        # print("total time render gui", time.time()-start_time)
        # print("")


def render_game(frame, results, game_configs, game_results, game_flags):
    if results is None:
        return frame_rendering.draw_calibration_marker(frame)
    else:
        # print(game_results)
        frame = frame_rendering.draw_fps(frame, results)
        if game_flags['show_kicker']:
            frame = frame_rendering.draw_field_calibrations(frame, game_configs)
        if game_flags['show_objects']:
            frame = frame_rendering.draw_ball(frame, results)
            frame = frame_rendering.draw_predicted_ball(frame, results, game_flags)
            frame = frame_rendering.draw_figures(frame, results, 'team1_positions', 'team1_on_field', 1,
                                                      'ranks_team1')
            frame = frame_rendering.draw_figures(frame, results, 'team2_positions', 'team2_on_field', 2,
                                                      'ranks_team2')
        return frame


def check_variables(user_input, game_flags):
    if keyboard.is_pressed("c"):
        user_input.value = ord('c')
        if user_input.value == ord('c'):
            game_flags['show_kicker'] = True
    if keyboard.is_pressed("f"):
        user_input.value = ord('f')
        if user_input.value == ord('f'):
            game_flags['show_kicker'] = False
    if keyboard.is_pressed("a"):
        user_input.value = ord('a')
        if user_input.value == ord('a'):
            game_flags['show_objects'] = True
    if keyboard.is_pressed("d"):
        user_input.value = ord('d')
        if user_input.value == ord('d'):
            game_flags['show_objects'] = False
    if keyboard.is_pressed("n"):  # new game
        user_input.value = ord('n')
        if user_input.value == ord('d'):
            game_flags['new_game'] = True
            game_flags["results"] = True


def update_gui(window, game_config, event, total_game_results, current_game_results):
    """
    update values on gui window
    """
    # self.window["-frame-"].update(data=cv2.imencode('.ppm', self.frame)[1].tobytes())
    window["-team_1-"].update(background_color=game_config['gui_team1_color'])
    window["-team_2-"].update(background_color=game_config['gui_team2_color'])
    window["-ball-"].update(background_color=game_config['gui_ball_color'])
    window["-score_team_1-"].update(current_game_results['counter_team1'])
    window["-score_team_2-"].update(current_game_results['counter_team2'])
    # self.window["-ball_speed-"].update(self.game.check_game_var("-ball_speed-"))
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
    if event == "-manual_game_counter_team_1_up-":
        current_game_results['counter_team1'] += 1
        window["-score_team_1-"].update(current_game_results['counter_team1'])
    if event == "-manual_game_counter_team_1_down-":
        if current_game_results['counter_team1'] > 0:
            current_game_results['counter_team1'] -= 1
        window["-score_team_1-"].update(current_game_results['counter_team1'])
    if event == "-manual_game_counter_team_2_up-":
        current_game_results['counter_team2'] += 1
        window["-score_team_2-"].update(current_game_results['counter_team2'])
    if event == "-manual_game_counter_team_2_down-":
        if current_game_results['counter_team2'] > 0:
            current_game_results['counter_team2'] -= 1
        window["-score_team_2-"].update(current_game_results['counter_team2'])
    # self.window["-heat_map-"].update(data=cv2.imencode(".ppm", self.analysis.check_analysis_var("-heat_map-")[1].tobytes()))

    return window
