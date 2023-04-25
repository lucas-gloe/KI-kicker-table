import cv2
import numpy as np
import configs


def define_players_position(hsv_img, game_config, team_dict_number, team_number):
    """
    create mask for player-colored object and search for contours on that mask
    Parameters:
        hsv_img(np.ndarray): HSV colored calibration image
        game_config(dict): calibration values for current game
        team_dict_number(string): team keyword in dict
        team_number(int): team number in field
    Returns:
        player_positions(list): list of every players position
        players_on_field(bool): boolean if players where found on mask
        ranked(list): list with ranks for every players postions based on their position
    """
    _ranked = []
    # read colors
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

    # create mask on frame based on the upper and lower border from calibrated colors
    players_mask = cv2.inRange(hsv_img, lower_color, upper_color)

    # find objects on calibrated mask
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
    Parameters:
        mask(np.ndarray): black-white image of the filtered positions of the objects
        team_number(int): team number in field
        game_config(dict): calibration values for current game
    Returns:
        objects(list): list of all tracked objects
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

    # compare tracked objects to match field to prevent wrong detection of outer scope objects
    if len(objects) >= 1:
        objects = np.delete(objects, np.where(
            ((game_config['match_field'][0][0] > objects[:, 0]) | (
                    objects[:, 0] > game_config['match_field'][1][0])) | (
                    (game_config['match_field'][0][1] > objects[:, 1]) | (
                    objects[:, 1] > game_config['match_field'][1][1]))), axis=0)

    # recalculate bounding boxes of players if a player is tracked twice
    if team_number == 1 or team_number == 2:
        objects = _remove_overlapping_bounding_boxes(objects, team_number, game_config)

    return objects


def _load_players_names(objects, team_number):
    """
    take the position of alle players of a team and rank them sorted from the position of the field
    Parameters:
        objects(list): list of all tracked objects
    Returns:
        ranks(list): list with ranks for every players postions based on their position
    """
    if len(objects) > 0:
        # take players positions
        position_matrix = np.array(objects)
        # put weight on players coords
        valued_matrix = position_matrix[:, 0, 0] * 10 + position_matrix[:, 0, 1]
        # sort players position after weighting
        sorted_valued_matrix = valued_matrix.argsort()
        # create empty array
        ranks = np.empty_like(sorted_valued_matrix)
        # fill array with evenly spaced values based on the number of tracked objects
        ranks[sorted_valued_matrix] = np.arange(len(valued_matrix))
        # reverse ranks for one team because of the opposite play direction
        if team_number == 1:
            ranks = __reverse_ranks(ranks)
        return ranks


def _remove_overlapping_bounding_boxes(objects, team_number, game_config):
    """
    check if a player was detected with more than one bounding box. If so combine these boxes to one big box
    so every player is only detected once
    Parameters:
        objects(np.array): list of all tracked objects
        team_number(int): team number in field
        game_config(dict): calibration values for current game
    Returns:
        max_bounding_boxes_team_X: list of all filtered objects
    """
    max_bounding_boxes = []
    # loop over every bounding box to check if two or more bounding boxes belong to the same player
    for contour in objects:
        y_mid = contour[1] + contour[3] / 2

        left_corner_x = contour[0]
        left_corner_y = y_mid - configs.HALF_PLAYERS_WIDTH
        right_corner_x = (contour[0] + contour[2])
        right_corner_y = y_mid + configs.HALF_PLAYERS_WIDTH
        max_box = [[left_corner_x, left_corner_y], [right_corner_x, right_corner_y]]
        max_bounding_boxes.append(max_box)

    np.zeros((len(max_bounding_boxes), len(max_bounding_boxes)))

    i = 0

    while i < len(max_bounding_boxes):
        for j, max_box2 in enumerate(max_bounding_boxes[i + 1:]):
            if (get_rod((max_bounding_boxes[i][0][0] + (
                    abs(max_bounding_boxes[i][0][0] - max_bounding_boxes[i][1][0]) / 2)), game_config) ==
                    get_rod((max_box2[0][0] + (abs(max_box2[0][0] - max_box2[1][0]) / 2)), game_config) and
                    (max_bounding_boxes[i][0][1] <= max_box2[0][1] <= max_bounding_boxes[i][1][1] or
                     max_bounding_boxes[i][0][1] <= max_box2[1][1] <= max_bounding_boxes[i][1][1])):
                max_bounding_boxes[i] = [[min(max_bounding_boxes[i][0][0], max_box2[0][0]),
                                           min(max_bounding_boxes[i][0][1], max_box2[0][1],
                                               max_bounding_boxes[i][1][1], max_box2[1][1])],
                                          [max(max_bounding_boxes[i][1][0], max_box2[1][0]),
                                           max(max_bounding_boxes[i][0][1], max_box2[0][1],
                                               max_bounding_boxes[i][1][1], max_box2[1][1])]
                                          ]
                max_bounding_boxes.pop(i + 1 + j)
                i = i - 1
                break
        i = i + 1

    if team_number == 1:
        max_bounding_boxes_team_1 = max_bounding_boxes
        return max_bounding_boxes_team_1
    if team_number == 2:
        max_bounding_boxes_team_2 = max_bounding_boxes
        return max_bounding_boxes_team_2


def get_rod(x, game_config):
    """
    check at which rod the detected shape is positioned
    Parameters:
        x(int): X coordinate from player
        game_config(dict): calibration values for current game
    Returns:
        i(int): position of rod in dict
    """
    # loop over every rod to compare x values of rod with given x value
    for i, rod in enumerate(game_config['players_rods']):
        if rod[0][0] - configs.RODWIDTH <= x <= rod[1][0] + configs.RODWIDTH:
            return i
    return -1


def __reverse_ranks(ranks):
    """
    Load players ranks for the opposite team, so the counter always starts at the goalkeeper
    Parameters:
        ranks(np.array): list with ranks for every players postions based on their position
    Returns:
        reversed_ranks(list): list with ranks for every players postions based on their position
    """
    reversed_ranks = []
    # reverse ranks to second team on match field, so both teams are starting at goal Keepers position
    for rank in ranks:
        place = (len(ranks) - 1) - rank
        reversed_ranks.append(place)

    return reversed_ranks


def define_balls_position(hsv_img, game_config, game_flags):
    """
    create mask for ball-colored object and search for contours on that mask
    Parameters:
        hsv_img(np.ndarray): HSV colored calibration image
        game_config(dict): calibration values for current game
        game_flags(dict): flag values for current game
    Returns:
        current_ball_position(list): current position of the ball
    """
    _predicted = (0, 0)
    ball_color = game_config["ball_color"]
    # read colors
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

    # blurring image to smooth out objects
    hsv_img = cv2.GaussianBlur(hsv_img, (3, 3), cv2.BORDER_DEFAULT)

    # create mask on frame based on the upper and lower border from calibrated colors
    mask = cv2.inRange(hsv_img, lower_color, upper_color)
    # find objects on calibrated mask
    ball_mask = _smooth_mask(mask)

    # tracking object on map
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
        current_ball_position = [center_x, center_y]

        game_flags['ball_was_found'] = True

    elif len(objects) == 0:
        print("ball not found")
        current_ball_position = [-1, -1]
        game_flags['ball_was_found'] = False
    else:
        print("ball not found")
        current_ball_position = [-1, -1]
        game_flags['ball_was_found'] = False

    return current_ball_position


def _smooth_mask(mask):
    """
    The mask created inDetectBallPosition might be noisy.
    source: https://www.computervision.zone/courses/learn-opencv-in-3-hours/
    Parameters:
        mask(np.ndarray): black-white image of the filtered positions of the objects
    Returns:
        mask(np.ndarray): filtered black-white image of the filtered positions of the objects
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

