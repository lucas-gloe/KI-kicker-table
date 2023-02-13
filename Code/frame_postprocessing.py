import cv2
import numpy as np


def count_game_score(ball_positions, game_config, current_game_results, game_flags):
    """
    Count game score +1  of a certain team if a goal was shot
    """
    if len(ball_positions) > 1 and 0 < ball_positions[-2][0] < game_config['goal1'][1][0] and game_config['goal1'][0][
        1] < ball_positions[-2][1] < game_config['goal1'][1][1] and ball_positions[-1] == [-1, -1] and game_flags[
        '_ball_reenters_game']:
        game_flags.update({
            '_goal1_detected': True,
            'goalInCurrentFrame': True
        })

    if len(ball_positions) > 1 and game_config['goal2'][0][0] < ball_positions[-2][0] < game_config['goal2'][1][0] and \
            game_config['goal2'][0][1] < \
            ball_positions[-2][1] < game_config['goal2'][1][1] and ball_positions[-1] == [-1, -1] and game_flags[
        '_ball_reenters_game']:
        game_flags.update({
            '_goal2_detected': True,
            'goalInCurrentFrame': True
        })

    if game_flags['_goal1_detected'] and game_flags['goalInCurrentFrame']:
        current_game_results['counter_team1'] += 1
        game_flags.update({
            'goalInCurrentFrame': False,
            '_ball_reenters_game': False,
            'recreate_shoot': True
        })

    if game_flags['_goal2_detected'] and game_flags['goalInCurrentFrame']:
        current_game_results['counter_team2'] += 1
        game_flags.update({
            'goalInCurrentFrame': False,
            '_ball_reenters_game': False,
            'recreate_shoot': True
        })


def detect_ball_reentering(ball_positions, game_config, game_flags):
    """
    Detect if the ball reenters the field in the middle section of the Kicker after a goal was shot
    """
    if game_flags['_goal1_detected'] or game_flags['_goal2_detected']:
        if len(ball_positions) >= 2:
            if game_config['throw_in_zone'][0][0] < ball_positions[-1][0] < game_config['throw_in_zone'][1][0] and \
                    ball_positions[-2] == [-1, -1]:
                game_flags.update({
                    '_goal1_detected': False,
                    '_goal2_detected': False,
                    '_ball_reenters_game': True,
                    'results': True
                })


def reset_game(current_game_results, total_game_results, game_flags):
    """
    Reset current game results to 0:0
    """
    if game_flags["new_game"] and game_flags['results']:
        team1 = current_game_results['counter_team1']
        team2 = current_game_results['counter_team2']
        total_game_results.append([team1, team2])
        current_game_results.update({
            'counter_team1': 0,
            'counter_team2': 0
        })
        game_flags['results'] = False


def predict_ball(ball_positions, current_game_results):
    predicted = KalmanFilter().predict(ball_positions[-1][0], ball_positions[-1][1])
    current_game_results["predicted"] = (predicted[0], predicted[1])
    current_ball_position = current_game_results["predicted"]
    return current_ball_position


class KalmanFilter:
    """
    Load the kalman filter for predicition purposess and use it to predict the next position of the ball
    Source: VisualComputer, https://www.youtube.com/watch?v=67jwpv49ymA
    """
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    @classmethod
    def predict(cls, coordX, coordY):
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        cls.kf.correct(measured)
        predicted = cls.kf.predict()
        x, y = int(predicted[0]), int(predicted[1])
        tuple = (x, y)
        return tuple
