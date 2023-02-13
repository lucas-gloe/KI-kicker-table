import cv2
import numpy as np
import configs


def define_players_position(hsv_img, game_config, team_dict_number, team_number):
    """
    create mask for player-colored object and search for contours on that mask
    """
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

    objects = _find_objects(players_mask, team_number, game_config)

    if len(objects) >= 1:
        players_on_field = True
        ranked = _load_players_names(objects, team_number)
        player_positions = objects
        return player_positions, players_on_field, ranked

    else:
        players_on_field = False
        print("Team " + str(team_number) + " not found")
        return [], players_on_field, []


def _find_objects(mask, team_number, game_config):
    """
    tracking algorithm to find the contours on the masks
    parts of code related to source : https://www.computervision.zone/courses/learn-opencv-in-3-hours/
    """
    # outline the contours on the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    objects = []
    # looping over every contour which was found
    for contour in contours:
        area_from_contours = cv2.contourArea(contour)
        # saving contours properties in variables if a certain area is detected on the mask (to prevent blurring)
        if area_from_contours > 20:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
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
        objects = _remove_overlapping_bounding_boxes(objects, team_number, game_config)

    return objects


def _load_players_names(objects, team_number):
    """
    take the position of alle players of a team and rank them sorted from the position of the field
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


def _remove_overlapping_bounding_boxes(objects, team_number, game_config):
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
            if (get_rod((_max_bounding_boxes[i][0][0] + (
                    abs(_max_bounding_boxes[i][0][0] - _max_bounding_boxes[i][1][0]) / 2)), game_config) ==
                    get_rod((max_box2[0][0] + (abs(max_box2[0][0] - max_box2[1][0]) / 2)), game_config) and
                    (_max_bounding_boxes[i][0][1] <= max_box2[0][1] <= _max_bounding_boxes[i][1][1] or
                     _max_bounding_boxes[i][0][1] <= max_box2[1][1] <= _max_bounding_boxes[i][1][1])):
                _max_bounding_boxes[i] = [[min(_max_bounding_boxes[i][0][0], max_box2[0][0]),
                                           min(_max_bounding_boxes[i][0][1], max_box2[0][1],
                                               _max_bounding_boxes[i][1][1], max_box2[1][1])],
                                          [max(_max_bounding_boxes[i][1][0], max_box2[1][0]),
                                           max(_max_bounding_boxes[i][0][1], max_box2[0][1],
                                               _max_bounding_boxes[i][1][1], max_box2[1][1])]
                                          ]
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


def get_rod(x, game_config):
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


def define_balls_position(hsv_img, game_config, game_flags):
    """
    create mask for ball-colored object and search for contours on that mask
    """
    _predicted = (0, 0)
    center_x = 0
    center_y = 0
    ball_color = game_config["ball_color"]

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
    ball_mask = _smooth_mask(mask)

    objects = _find_objects(ball_mask, -1, game_config)
    objects.flatten()

    if len(objects) == 1:
        game_flags['predicted_value_added'] = False

        x = objects[0][0]
        y = objects[0][1]
        w = objects[0][2]
        h = objects[0][3]

        # defining the center points for the case the detected contour is the ball
        center_x = int((x + (w / 2)))
        center_y = int((y + (h / 2)))

        # save the current position of the ball into an array
        _current_ball_position = [center_x, center_y]

        game_flags['ball_was_found'] = True

    elif len(objects) == 0:
        print("ball not found")
        _current_ball_position = [-1, -1]
        game_flags['ball_was_found'] = False
    else:
        print("ball not found")
        _current_ball_position = [-1, -1]
        game_flags['ball_was_found'] = False

    return _current_ball_position


def _smooth_mask(mask):
    """
    The mask created inDetectBallPosition might be noisy.
    source: https://www.computervision.zone/courses/learn-opencv-in-3-hours/
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

