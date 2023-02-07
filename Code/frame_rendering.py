import cv2
import configs


def draw_fps(frame, results):
    """
    draw FPS on frame
    """
    cv2.putText(frame, str(results["fps"]), (int(frame.shape[0] / 2), int(frame.shape[1] / 2)),
                cv2.FONT_HERSHEY_PLAIN, 1, (30, 144, 255), 2)
    return frame


def draw_field_calibrations(frame, game_configs):
    """
    show football field contour for calibration on frame if a certain button was pressed
    """
    cv2.line(frame, (int(game_configs['field'][0][0]), int(game_configs['field'][0][1])),
             (int(game_configs['field'][1][0]), int(game_configs['field'][1][1])),
             (0, 255, 0), 2)
    cv2.line(frame, (int(game_configs['field'][2][0]), int(game_configs['field'][2][1])),
             (int(game_configs['field'][3][0]), int(game_configs['field'][3][1])),
             (0, 255, 0), 2)
    cv2.line(frame, (int(game_configs['field'][0][0]), int(game_configs['field'][0][1])),
             (int(game_configs['field'][3][0]), int(game_configs['field'][3][1])),
             (0, 255, 0), 2)
    cv2.line(frame, (int(game_configs['field'][1][0]), int(game_configs['field'][1][1])),
             (int(game_configs['field'][2][0]), int(game_configs['field'][2][1])),
             (0, 255, 0), 2)
    cv2.rectangle(frame, (int(game_configs['goal1'][0][0]), int(game_configs['goal1'][0][1])),  # tor von Team1, orange
                  (int(game_configs['goal1'][1][0]), int(game_configs['goal1'][1][1])),
                  (0, 255, 0), 2)
    cv2.rectangle(frame, (int(game_configs['goal2'][0][0]), int(game_configs['goal2'][0][1])),  # tor von Team2, blau
                  (int(game_configs['goal2'][1][0]), int(game_configs['goal2'][1][1])),
                  (0, 255, 0), 2)
    cv2.rectangle(frame, (int(game_configs['throw_in_zone'][0][0]), int(game_configs['throw_in_zone'][0][1])),
                  (int(game_configs['throw_in_zone'][1][0]), int(game_configs['throw_in_zone'][1][1])),
                  (0, 255, 0), 2)
    for rod in game_configs['players_rods']:
        cv2.rectangle(frame, (int(rod[0][0]), int(rod[0][1])),
                      (int(rod[1][0]), int(rod[1][1])), (0, 255, 255), 2)
    return frame


def draw_ball(frame, results):
    """
    Draw a circle at the balls position and name the Object "ball"
    """
    # draw a circle for the ball
    if results['ball_position'] != [-1, -1]:
        cv2.circle(frame, (results['ball_position'][0], results['ball_position'][1]), int(16 * configs.SCALE_FACTOR),
                   (0, 255, 0), 2)
        cv2.putText(frame, "Ball", (results['ball_position'][0], results['ball_position'][1]),
                    cv2.FONT_HERSHEY_PLAIN, 1, (30, 144, 255), 2)
    return frame


def draw_predicted_ball(frame, results, game_flags):
    """
    Draw a circle at the predicted balls position if there is no ball
    detected in the frame and name the Object "ball"
    """
    if results["ball_position"] == [-1, -1]:
        cv2.circle(frame, (results["predicted"][0], results["predicted"][1]), int(16 * configs.SCALE_FACTOR),
                   (0, 255, 255), 2)
    return frame


def draw_figures(frame, results, team_dict_number, team_dict_flag, team_number, team_ranks):
    """
    Draw a rectangle at the players position and name it TeamX
    """
    if results[team_dict_flag]:
        for i, player_position in enumerate(results[team_dict_number]):
            cv2.rectangle(frame, (int(player_position[0][0]), int(player_position[0][1])),
                          (int(player_position[1][0]), int(player_position[1][1])),
                          (0, 255, 0), 2)
            cv2.putText(frame,
                        ("Team" + str(team_number) + ", " + str(results[team_ranks][i])),
                        (int(player_position[0][0]), int(player_position[0][1])), cv2.FONT_HERSHEY_PLAIN, 1,
                        (30, 144, 255), 2)
    return frame


def draw_calibration_marker(frame):
    cv2.circle(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)), int(18 * configs.SCALE_FACTOR),
               (30, 144, 255), 1)
    cv2.circle(frame, (int(frame.shape[1] / 2 - int(85 * configs.SCALE_FACTOR)), int(frame.shape[0] / 2)),
               int(18 * configs.SCALE_FACTOR), (30, 144, 255), 1)
    cv2.circle(frame, (int(frame.shape[1] / 2 + int(85 * configs.SCALE_FACTOR)), int(frame.shape[0] / 2)),
               int(18 * configs.SCALE_FACTOR), (30, 144, 255), 1)
    return frame
