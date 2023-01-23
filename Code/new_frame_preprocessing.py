import cv2
import numpy as np
import configs


def define_players_position(hsv_img, game_config, team_dict_number, team_number):
    _ranked = []

    team_color = game_config[team_dict_number]
    lower_color = np.asarray(team_color)
    upper_color = np.asarray(team_color)
    lower_color = lower_color - [10, 50, 50]
    upper_color = upper_color + [10, 50, 50]
    lower_color[lower_color < 0] = 0
    lower_color[lower_color > 255] = 255
    upper_color[upper_color < 0] = 0
    upper_color[upper_color > 255] = 255

    lower_color = np.array(lower_color)
    upper_color = np.array(upper_color)

    players_mask = cv2.inRange(hsv_img, lower_color, upper_color)

    objects = __find_objects(players_mask, team_number, game_config)

    if len(objects) >= 1:
        players_on_field = True
        ranked = __load_players_names(objects, team_number)
        player_positions = objects
        return player_positions, players_on_field, ranked

    else:
        players_on_field = False
        print("Team " + str(team_number) + " not found")
        return [], players_on_field, []


def __find_objects(mask, team_number, game_config):
    """
    tracking algorithm to find the contours on the masks
    return: Contours on the masks
    source: https://www.computervision.zone/courses/learn-opencv-in-3-hours/
    """
    # outline the contours on the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    objects = []
    # looping over every contour which was found
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # saving contours properties in variables if a certain area is detected on the mask (to prevent blurring)
        if area > 20:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            white_contour = x, y, w, h
            objects.append(white_contour)

    objects = np.array(objects)

    if len(objects) >= 1:
        objects = np.delete(objects, np.where(
            ((game_config['match_field'][0][0] > objects[:, 0]) | (
                    objects[:, 0] > game_config['match_field'][1][0])) | (
                    (game_config['match_field'][0][1] > objects[:, 1]) | (
                    objects[:, 1] > game_config['match_field'][1][1]))), axis=0)

    if team_number == 1 or team_number == 2:
        objects = __remove_overlapping_bounding_boxes(objects, team_number, game_config)

    return objects


def __load_players_names(objects, team_number):
    """
    take the position of alle players of a team and rank them sorted from the position of the field
    Return: Array with sorted list of players ranks
    """
    if len(objects) > 0:
        position_matrix = np.array(objects)
        valued_matrix = position_matrix[:, 0, 0] * 10 + position_matrix[:, 0, 1]
        sorted_valued_matrix = valued_matrix.argsort()
        ranks = np.empty_like(sorted_valued_matrix)
        ranks[sorted_valued_matrix] = np.arange(len(valued_matrix))
        if team_number == 1:
            ranks = __reverse_ranks(ranks)
        return ranks


def __remove_overlapping_bounding_boxes(objects, team_number, game_config):
    """
    check if a player was detected with more than one bounding box. If so combine these boxes to one big box
    so every player is only detected once
    """
    _max_bounding_boxes = []

    for contour in objects:
        y_mid = contour[1] + contour[3] / 2

        left_corner_x = contour[0]
        left_corner_y = y_mid - configs.HALF_PLAYERS_WIDTH
        right_corner_x = (contour[0] + contour[2])
        right_corner_y = y_mid + configs.HALF_PLAYERS_WIDTH
        max_box = [[left_corner_x, left_corner_y], [right_corner_x, right_corner_y]]
        _max_bounding_boxes.append(max_box)

    np.zeros((len(_max_bounding_boxes), len(_max_bounding_boxes)))

    i = 0

    while i < len(_max_bounding_boxes):
        for j, max_box2 in enumerate(_max_bounding_boxes[i + 1:]):
            if (__get_rod((_max_bounding_boxes[i][0][0] + (
                    abs(_max_bounding_boxes[i][0][0] - _max_bounding_boxes[i][1][0]) / 2)), game_config) == __get_rod(
                (max_box2[0][0] + (abs(max_box2[0][0] - max_box2[1][0]) / 2)), game_config) and (
                    (max_box2[0][1] <= _max_bounding_boxes[i][0][1] <= max_box2[1][1]) or (
                    max_box2[0][1] <= _max_bounding_boxes[i][1][1] <= max_box2[1][1]))):
                _max_bounding_boxes[i] = [
                    [min(_max_bounding_boxes[i][0][0], max_box2[0][0]),
                     int(min(_max_bounding_boxes[i][0][1], max_box2[0][1], _max_bounding_boxes[i][1][1],
                             max_box2[1][1]))],
                    [max(_max_bounding_boxes[i][1][0], max_box2[1][0]),
                     int(max(_max_bounding_boxes[i][0][1], max_box2[0][1], _max_bounding_boxes[i][1][1],
                             max_box2[1][1]))]]
                _max_bounding_boxes.pop(i + 1 + j)
                i = i - 1
                break
        i = i + 1

    if team_number == 1:
        _max_bounding_boxes_team_1 = _max_bounding_boxes
        return _max_bounding_boxes_team_1
    if team_number == 2:
        _max_bounding_boxes_team_2 = _max_bounding_boxes
        return _max_bounding_boxes_team_2


def __get_rod(x, game_config):
    """
    check at which rod the detected shape is positioned
    """
    for i, rod in enumerate(game_config['players_rods']):
        if rod[0][0] - configs.RODWIDTH <= x <= rod[1][0] + configs.RODWIDTH:
            return i
    return -1


def __reverse_ranks(ranks):
    """
    Load players ranks for the opposite team, so the counter always starts at the goalkeeper
    """
    reversed_ranks = []

    for rank in ranks:
        place = (len(ranks) - 1) - rank
        reversed_ranks.append(place)

    return reversed_ranks


def define_balls_position(hsv_img, game_config):
    _predicted = (0, 0)
    center_x = 0
    center_y = 0
    ball_color = game_config["ball_color"]

    # if not self.results_from_calibration and self.ball_was_found:
    #     self.ball_color = [int((self._colors[0][0] + self.__recalibrated_ball_color[0]) / 2),
    #                         int((self._colors[0][1] + self.__recalibrated_ball_color[1]) / 2),
    #                         int((self._colors[0][2] + self.__recalibrated_ball_color[2]) / 2)]

    lower_color = np.asarray(ball_color)
    upper_color = np.asarray(ball_color)
    lower_color = lower_color - [10, 50, 50]
    upper_color = upper_color + [10, 50, 50]
    lower_color[lower_color < 0] = 0
    lower_color[lower_color > 255] = 255
    upper_color[upper_color < 0] = 0
    upper_color[upper_color > 255] = 255

    lower_color = np.array(lower_color)
    upper_color = np.array(upper_color)

    # blurring image to prevent false object detection
    hsv_img = cv2.GaussianBlur(hsv_img, (3, 3), cv2.BORDER_DEFAULT)

    mask = cv2.inRange(hsv_img, lower_color, upper_color)
    ball_mask = __smooth_mask(mask)

    objects = __find_objects(ball_mask, -1, game_config)
    objects.flatten()

    if len(objects) == 1:

        # game_flags['predicted_value_added'] = False

        x = objects[0][0]
        y = objects[0][1]
        w = objects[0][2]
        h = objects[0][3]

        # defining the center points for the case the detected contour is the ball
        center_x = int((x + (w / 2)))
        center_y = int((y + (h / 2)))

        # save the current position of the ball into an array
        _current_ball_position = [center_x, center_y]

        # recalibrate ball color in current frame
        # self.__recalibrated_ball_color = self._dc.recalibrate_ball_color(hsv_img, center_x, center_y,
        #                                                                  self._team1_figures, self._team2_figures,
        #                                                                  self.players_rods)
        # game_flags['ball_was_found'] = True

        # _predicted = KalmanFilter().predict(center_x, center_y)

    elif len(objects) == 0:
        # if not game_flags['predicted_value_added']:
        #     # _current_ball_position = (_predicted[0], _predicted[1])
        #     _current_ball_position = [-1, -1]
        #     game_flags['predicted_value_added'] = True
        #     game_flags['ball_was_found'] = False
        # else:
        print("ball not found")
        _current_ball_position = [-1, -1]
        # game_flags['ball_was_found'] = False
    else:
        #     #self.__calculate_balls_position(objects)
        # if center_x != 0 & center_y != 0:
        #     _current_ball_position = [center_x, center_y]
        # else:
        print("ball not found")
        _current_ball_position = [-1, -1]

    return _current_ball_position


def __smooth_mask(mask):
    """
    The mask created inDetectBallPosition might be noisy.
    :param mask: The mask to smooth (Image with bit depth 1)
    :return: The smoothed mask
    """
    KERNELS = 3
    # create the disk-shaped kernel for the following image processing,
    kernel = np.ones((2 * KERNELS, 2 * KERNELS), np.uint8)
    for x in range(0, 2 * KERNELS):
        for y in range(0, 2 * KERNELS):
            if (x - KERNELS + 0.5) ** 2 + (y - KERNELS + 0.5) ** 2 > KERNELS ** 2:
                kernel[x, y] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

