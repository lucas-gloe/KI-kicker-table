import cv2
import numpy as np
import PySimpleGUI as sg
import configs


def calibrate_before_first_image():
    window = initialize_GUI_layout(configs.FONT)

    return window


def calibrate(calibration_image, game_config):
    angle = get_angle(calibration_image)
    center, ratio_pxcm = get_center_scale(calibration_image)
    field = calc_field(angle, center, ratio_pxcm)
    ball_color, team1_color, team2_color = calibrate_color(calibration_image, center)
    match_field, goal1, goal2, throw_in_zone, players_rods = load_game_field_properties(field)
    gui_ball_color, gui_team1_color, gui_team2_color = convert_tracked_hsv_colors(ball_color, team1_color, team2_color)

    game_config['goal1'] = goal1
    game_config['goal2'] = goal2
    game_config['throw_in_zone'] = throw_in_zone
    game_config['players_rods'] = players_rods
    game_config['angle'] = angle
    game_config['field'] = field
    game_config['match_field'] = match_field
    game_config['center'] = center
    game_config['ratio_pxcm'] = ratio_pxcm
    game_config['ball_color'] = ball_color
    game_config['team1_color'] = team1_color
    game_config['team2_color'] = team2_color
    game_config['gui_ball_color'] = gui_ball_color
    game_config['gui_team1_color'] = gui_team1_color
    game_config['gui_team2_color'] = gui_team2_color

    return game_config


def get_angle(calibration_image):
    """
    :param calibration_image: The HSV-image to use for calculation
    :return: Rotation angle of the field in image
    """
    rgb = cv2.cvtColor(calibration_image, cv2.COLOR_HSV2BGR)
    angle = 0
    count = 0

    gray = cv2.cvtColor(cv2.cvtColor(calibration_image, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 110)

    if lines.shape[0]:
        line_count = lines.shape[0]
    else:
        raise Exception('field not detected')

    for x in range(line_count):

        for rho, theta in lines[x]:
            a = np.cos(theta)
            b = np.sin(theta)
            # print(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)

            corr_angle = np.degrees(b)
            if corr_angle < 5:
                # print(CorrAngle)
                angle = angle + corr_angle
                count = count + 1
                cv2.line(rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if isinstance(angle, int) and isinstance(count, int):
        angle = angle / count

    else:
        angle = 0.1

    return angle


def get_center_scale(calibration_image):
    """
    :param calibration_image: The HSV-image to use for calculation
    :return: Position of center point in image (tuple), ratio px per cm (reproduction scale)
    """
    gray = cv2.cvtColor(calibration_image, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if configs.SCALE_FACTOR >= 0.4:
        gray = cv2.GaussianBlur(gray, (5, 5), 1)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=50, param2=30, minRadius=30, maxRadius=100)
    else:
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=int(50 * configs.SCALE_FACTOR),
                                   param2=int(30 * configs.SCALE_FACTOR), minRadius=int(30 * configs.SCALE_FACTOR),
                                   maxRadius=int(100 * configs.SCALE_FACTOR))
    center_circle = (0, 0, 0)  # centerx, centery, radius
    min_dist = 0xFFFFFFFFFFF
    for circle in circles[0]:
        dist_x = abs(circle[0] - calibration_image.shape[1] / 2)
        dist_y = abs(circle[1] - calibration_image.shape[0] / 2)

        if (dist_x + dist_y) < min_dist:
            min_dist = dist_x + dist_y
            center_circle = circle

    if center_circle[2] < 42.7:
        center_circle[0] = calibration_image.shape[1] / 2
        center_circle[1] = calibration_image.shape[0] / 2
        center_circle[2] = (114 * configs.SCALE_FACTOR)

    cv2.circle(gray, (int(center_circle[0]), int(center_circle[1])), int(center_circle[2]), (0, 255, 0), 2)

    cv2.imwrite("gray.JPG", gray)

    center = int(center_circle[0]), int(center_circle[1])
    radius = center_circle[2]
    ratio_pxcm = radius / 9.4

    return [center, ratio_pxcm]


def calc_field(angle, center, ratio_pxcm):
    """
    This method needs some class variables. get_angle and get_center_scale
    have to be called beforehand.
    :return: field edges [Top left, top right, bottom right and bottom left corner] (list)
    """

    half_field_width = configs.HALF_FIELD_WIDTH
    half_field_height = configs.HALF_FIELD_HEIGHT

    angle_radial_scale = np.radians(angle)

    x2 = int(center[0] - (half_field_width * ratio_pxcm) + np.tan(angle_radial_scale) *
             (half_field_height * ratio_pxcm))
    y2 = int(center[1] - np.tan(angle_radial_scale) * (half_field_width * ratio_pxcm) -
             (half_field_height * ratio_pxcm))
    top_left = [x2, y2]

    x2 = int(center[0] + (half_field_width * ratio_pxcm) + np.tan(angle_radial_scale) *
             (half_field_height * ratio_pxcm))
    y2 = int(center[1] + np.tan(angle_radial_scale) * (half_field_width * ratio_pxcm) -
             (half_field_height * ratio_pxcm))
    top_right = [x2, y2]

    x2 = int(center[0] - (half_field_width * ratio_pxcm) - np.tan(angle_radial_scale) *
             (half_field_height * ratio_pxcm))
    y2 = int(center[1] - np.tan(angle_radial_scale) * (half_field_width * ratio_pxcm) +
             (half_field_height * ratio_pxcm))
    bottom_left = [x2, y2]

    x2 = int(center[0] + (half_field_width * ratio_pxcm) - np.tan(angle_radial_scale) *
             (half_field_height * ratio_pxcm))
    y2 = int(center[1] + np.tan(angle_radial_scale) * (half_field_width * ratio_pxcm) +
             (half_field_height * ratio_pxcm))
    bottom_right = [x2, y2]

    field = [top_left, top_right, bottom_right, bottom_left]
    return field


def calibrate_color(calibration_image, center):
    """
    The user has to put the ball onto the center spot for calibration. the taken image will be used to read the colors from the marked positions.
    :param_type: calibration img, array
    :return: ball color, team colors
    """

    calibration_image = cv2.cvtColor(calibration_image, cv2.COLOR_BGR2HSV)

    # The initialization is done with only a small part of the image around the center spot.
    x1 = int(round(center[0] - (calibration_image.shape[1] / 10), 0))
    x2 = int(round(center[0] + (calibration_image.shape[1] / 10), 0))
    y1 = int(round(center[1] - (calibration_image.shape[0] / 10), 0))
    y2 = int(round(center[1] + (calibration_image.shape[0] / 10), 0))

    cropped_hsv_img = calibration_image[y1:y2, x1:x2]

    cv2.imwrite("cropped_calibration_img.JPG", cropped_hsv_img)

    ball_color = calibrate_ball_color(cropped_hsv_img)
    team1_color = calibrate_team_color(cropped_hsv_img, 1)
    team2_color = calibrate_team_color(cropped_hsv_img, 2)

    return [ball_color, team1_color, team2_color]


def calibrate_ball_color(cropped_hsv_img):
    """
    Calibration routine.
    Measures the color of the ball and stores it in the class.
    :param cropped_hsv_img: HSV-image to use for calculation.
    The ball has to be positioned in the center
    :return: None
    """
    x_center = int(round(cropped_hsv_img.shape[1] / 2))
    y_center = int(round(cropped_hsv_img.shape[0] / 2))

    # Get the color of the pixel around the image center
    colors = cropped_hsv_img[y_center - int(5 * configs.SCALE_FACTOR):y_center + int(6 * configs.SCALE_FACTOR),
             x_center - int(5 * configs.SCALE_FACTOR):x_center + int(6 * configs.SCALE_FACTOR)]
    lower_border_arr = [np.min(colors[:, :, 0]), np.min(colors[:, :, 1]), np.min(colors[:, :, 2])]
    upper_border_arr = [np.max(colors[:, :, 0]), np.max(colors[:, :, 1]), np.max(colors[:, :, 2])]

    # Create a mask for the areas with a color similar to the center pixel
    lower_border_arr = np.array(lower_border_arr)
    upper_border_arr = np.array(upper_border_arr)

    lower_border = tuple(lower_border_arr.tolist())
    upper_border = tuple(upper_border_arr.tolist())

    mask = cv2.inRange(cropped_hsv_img, lower_border, upper_border)

    # Average the color values of the masked area
    colors = cropped_hsv_img[mask == 255]
    h_mean = int(round(np.mean(colors[:, 0])))
    s_mean = int(round(np.mean(colors[:, 1])))
    v_mean = int(round(np.mean(colors[:, 2])))

    av = [h_mean, s_mean, v_mean]
    ball_color = tuple(av)

    return ball_color

def calibrate_team_color(cropped_hsv_img, team_number):
    """
    Calibration routine.
    Measures the color of the ball and stores it in the class.
    :param cropped_hsv_img: HSV-image to use for calculation.
    The ball has to be positioned in the center
    :return: None
    """
    # Get the exact point for measuring
    x_center = int(round(cropped_hsv_img.shape[1] / 2))
    y_center = int(round(cropped_hsv_img.shape[0] / 2))

    if team_number == 1:
        x_player = x_center + int(85 * configs.SCALE_FACTOR)
        y_player = y_center

    if team_number == 2:
        x_player = x_center - int(85 * configs.SCALE_FACTOR)
        y_player = y_center

    # Get the color of the pixel in the image center
    color = cropped_hsv_img[y_player, x_player]
    colors = cropped_hsv_img[y_player - int(6 * configs.SCALE_FACTOR):y_player + int(5 * configs.SCALE_FACTOR),
             x_player - int(6 * configs.SCALE_FACTOR):x_player + int(5 * configs.SCALE_FACTOR)]
    lower_border_arr = [np.min(colors[:, :, 0]), np.min(colors[:, :, 1]), np.min(colors[:, :, 2])]
    upper_border_arr = [np.max(colors[:, :, 0]), np.max(colors[:, :, 1]), np.max(colors[:, :, 2])]

    # Create a mask for the areas with a color similar to the center pixel
    lower_border_arr = np.array(lower_border_arr)
    upper_border_arr = np.array(upper_border_arr)

    lower_border = tuple(lower_border_arr.tolist())
    upper_border = tuple(upper_border_arr.tolist())

    mask = cv2.inRange(cropped_hsv_img, lower_border, upper_border)

    # Average the color values of the masked area
    colors = cropped_hsv_img[mask == 255]
    h_mean = int(round(np.mean(colors[:, 0])))
    s_mean = int(round(np.mean(colors[:, 1])))
    v_mean = int(round(np.mean(colors[:, 2])))

    av = [h_mean, s_mean, v_mean]
    if team_number == 1:
        team1_color = tuple(av)
        return team1_color

    if team_number == 2:
        team2_color = tuple(av)
        return team2_color


def load_game_field_properties(field):
    """
    calculate the position of the field, the goals, the throw-in-zone and the rods.
    :return: position parameters
    """
    match_field = np.array([[int(field[0][0]), int(field[0][1])],
                            [int(field[2][0]), int(field[2][1])]])
    goal1 = np.array([[int(match_field[0][0]), int((np.linalg.norm(
        (match_field[1][1] - match_field[0][1]) / 2) + match_field[0][1]) - int(
        configs.HALF_WIDTH_GOAL * configs.SCALE_FACTOR))],
                      [int(match_field[0][0] + int(100 * configs.SCALE_FACTOR)), int((np.linalg.norm(
                          (match_field[1][1] - match_field[0][1]) / 2) + match_field[0][
                                                                                          1]) + int(
                          configs.HALF_WIDTH_GOAL * configs.SCALE_FACTOR))]])
    goal2 = np.array([[int(match_field[1][0] - int(100 * configs.SCALE_FACTOR)), int((np.linalg.norm(
        (match_field[1][1] - match_field[0][1]) / 2) + match_field[0][1]) - int(
        configs.HALF_WIDTH_GOAL * configs.SCALE_FACTOR))],
                      [int(match_field[1][0]), int((np.linalg.norm(
                          (match_field[1][1] - match_field[0][1]) / 2) + match_field[0][
                                                        1]) + int(configs.HALF_WIDTH_GOAL * configs.SCALE_FACTOR))]])
    throw_in_zone = np.array([[int(match_field[0][0] + int(400 * configs.SCALE_FACTOR)), int(match_field[0][1])],
                              [int(match_field[1][0] - int(400 * configs.SCALE_FACTOR)), int(match_field[1][1])]])

    distance_between_rods = (np.linalg.norm(match_field[1][0] - match_field[0][0])) / 8

    players_rods = np.array([[[int(
        match_field[0][0] + (0.5 * distance_between_rods - int(16 * configs.SCALE_FACTOR)) + 12),
        int(match_field[0][1])],
        [int(match_field[0][0] + (
                0.5 * distance_between_rods + int(16 * configs.SCALE_FACTOR)) + 12),
         int(match_field[1][1])]],
        [[int(match_field[0][0] + (
                1.5 * distance_between_rods - int(16 * configs.SCALE_FACTOR)) + 9),
          int(match_field[0][1])],
         [int(match_field[0][0] + (
                 1.5 * distance_between_rods + int(16 * configs.SCALE_FACTOR)) + 9),
          int(match_field[1][1])]],
        [[int(match_field[0][0] + (
                2.5 * distance_between_rods - int(16 * configs.SCALE_FACTOR)) + 9),
          int(match_field[0][1])],
         [int(match_field[0][0] + (
                 2.5 * distance_between_rods + int(16 * configs.SCALE_FACTOR)) + 9),
          int(match_field[1][1])]],
        [[int(match_field[0][0] + (
                3.5 * distance_between_rods - int(16 * configs.SCALE_FACTOR)) + 3),
          int(match_field[0][1])],
         [int(match_field[0][0] + (
                 3.5 * distance_between_rods + int(16 * configs.SCALE_FACTOR)) + 3),
          int(match_field[1][1])]],
        [[int(match_field[0][0] + (
                4.5 * distance_between_rods - int(16 * configs.SCALE_FACTOR)) - 3),
          int(match_field[0][1])],
         [int(match_field[0][0] + (
                 4.5 * distance_between_rods + int(16 * configs.SCALE_FACTOR)) - 3),
          int(match_field[1][1])]],
        [[int(match_field[0][0] + (
                5.5 * distance_between_rods - int(16 * configs.SCALE_FACTOR)) - 9),
          int(match_field[0][1])],
         [int(match_field[0][0] + (
                 5.5 * distance_between_rods + int(16 * configs.SCALE_FACTOR)) - 9),
          int(match_field[1][1])]],
        [[int(match_field[0][0] + (
                6.5 * distance_between_rods - int(16 * configs.SCALE_FACTOR)) - 9),
          int(match_field[0][1])],
         [int(match_field[0][0] + (
                 6.5 * distance_between_rods + int(16 * configs.SCALE_FACTOR)) - 9),
          int(match_field[1][1])]],
        [[int(match_field[0][0] + (
                7.5 * distance_between_rods - int(16 * configs.SCALE_FACTOR)) - 12),
          int(match_field[0][1])],
         [int(match_field[0][0] + (
                 7.5 * distance_between_rods + int(16 * configs.SCALE_FACTOR)) - 12),
          int(match_field[1][1])]]])

    return [match_field, goal1, goal2, throw_in_zone, players_rods]


def convert_tracked_hsv_colors(ball_color, team1_color, team2_color):
    _tracked_ball_color_hsv = np.uint8([[ball_color]])
    _tracked_team1_color_hsv = np.uint8([[team1_color]])
    _tracked_team2_color_hsv = np.uint8([[team2_color]])

    _tracked_ball_color_rgb = cv2.cvtColor(_tracked_ball_color_hsv, cv2.COLOR_HSV2RGB)
    _tracked_team1_color_rgb = cv2.cvtColor(_tracked_team1_color_hsv, cv2.COLOR_HSV2RGB)
    _tracked_team2_color_rgb = cv2.cvtColor(_tracked_team2_color_hsv, cv2.COLOR_HSV2RGB)

    gui_ball_color = __rgb2hex(_tracked_ball_color_rgb)
    gui_team1_color = __rgb2hex(_tracked_team1_color_rgb)
    gui_team2_color = __rgb2hex(_tracked_team2_color_rgb)

    return gui_ball_color, gui_team1_color, gui_team2_color


def __rgb2hex(color):
    return '#%02X%02X%02X' % (color[0][0][0], color[0][0][1], color[0][0][2])


def initialize_GUI_layout(FONT):

    # layout of the GUI
    sg.theme('Reddit')

    ################# Game frame #######################################

    game_frame = [

        [sg.Image(filename="", key="-frame-")]
    ]
    ################# left frame with basic game infos #################

    # inner frame 1

    game_score_and_speed = [
        [sg.Text('SIT Smart Kicker', text_color='orange', font=(FONT, 30))],
        [sg.Button('goal+1', key="-manual_game_counter_team_1_up-", button_color='grey', font=(FONT, 8)),
         sg.Button('goal-1', key="-manual_game_counter_team_1_down-", button_color='grey', font=(FONT, 8)),
         sg.Button('goal+1', key="-manual_game_counter_team_2_up-", button_color='grey', font=(FONT, 8)),
         sg.Button('goal-1', key="-manual_game_counter_team_2_down-", button_color='grey', font=(FONT, 8))],
        [sg.Text("", key='-score_team_1-', font=(FONT, 45)), sg.Text(" : ", font=(FONT, 20)),
         sg.Text("", key='-score_team_2-', font=(FONT, 45))],
        [sg.Text("Team 1", font=(FONT, 15)),
         sg.Text("Team 2", font=(FONT, 15))],
        [sg.Text('Ball Speed:', font=(FONT, 10)),
         sg.Text("NOT SET YET", key='-ball_speed-', font=(FONT, 10))],
        [sg.Text('FPS:', font=(FONT, 10)),
         sg.Text("0", key='-fps-', font=(FONT, 10))],
        [sg.Text('Press S to save configuration image', key='-config_img-', font=(FONT, 10))]
    ]

    # inner frame 2

    game_configuration = [
        [sg.Text("Game Config", text_color='orange', font=(FONT, 15))],
        [sg.Text("Team 1", font=(FONT, 10), text_color='white', background_color="orange", key="-team_1-",
                 expand_x=True, justification='c')],
        [sg.Text("Team 2", font=(FONT, 10), text_color='white', background_color="orange", key="-team_2-",
                 expand_x=True, justification='c')],
        [sg.Text("Ball", font=(FONT, 10), text_color='white', background_color="orange", key="-ball-",
                 expand_x=True, justification='c')]
    ]

    last_games = [
        [sg.Text("Last Games", text_color='orange', font=(FONT, 15))],
        [sg.Text("", key='-last_game_team1-', font=(FONT, 10)), sg.Text(" : ", font=(FONT, 10)),
         sg.Text("", key='-last_game_team2-', font=(FONT, 10))],
        [sg.Text("", key='-second_last_game_team1-', font=(FONT, 10)), sg.Text(" : ", font=(FONT, 10)),
         sg.Text("", key='-second_last_game_team2-', font=(FONT, 10))],
        [sg.Text("", key='-third_last_game_team1-', font=(FONT, 10)), sg.Text(" : ", font=(FONT, 10)),
         sg.Text("", key='-third_last_game_team2-', font=(FONT, 10))]
    ]

    configurations = [
        [sg.Frame("", game_configuration, expand_x=True, expand_y=True, element_justification='c'),
         sg.Frame("", last_games, expand_x=True, expand_y=True, element_justification='c')]
    ]

    # inner frame 3

    key_bindings = [
        [sg.Text("Key Bindings", text_color='orange', font=(FONT, 15))],
        [sg.Text('Press N to start new game', font=(FONT, 10))],
        [sg.Text("Press C to show kicker, press F to hide kicker", font=(FONT, 10))],
        [sg.Text("Press A to show contours, press D to hide contours", font=(FONT, 10))],
        [sg.Text("Press M to switch to manual mode, l for automatic", font=(FONT, 10))],
        [sg.Text("Press K to loop through the frames", font=(FONT, 10))]
    ]

    # left frame

    basic_information = [
        [sg.Frame("", game_score_and_speed, expand_x=True, expand_y=True, element_justification='c')],
        [sg.Frame("", configurations, border_width=0, expand_x=True, expand_y=True)]
    ]

    game_stats = [
        [sg.Frame("", layout=basic_information, border_width=0, expand_x=True, expand_y=True)]
    ]

    ################# right frame with advanced infos #################

    # frame pattern

    heat_map = [
        [sg.Text("Place Holder", text_color='orange', font=(FONT, 15))]
    ]

    blank_frame = [
        [sg.Text("Place Holder", text_color='orange', font=(FONT, 15))]
    ]

    blank_frame2 = [
        [sg.Text("Place Holder", text_color='orange', font=(FONT, 15))]
    ]

    deep_information = [
        [sg.Frame("", key_bindings, expand_x=True, expand_y=True, element_justification='c'),
         sg.Frame("", heat_map, expand_x=True, expand_y=True, element_justification='c'),
         sg.Frame("", blank_frame, expand_x=True, expand_y=True, element_justification='c'),
         sg.Frame("", blank_frame2, expand_x=True, expand_y=True, element_justification='c')]
    ]

    # right frame

    game_analysis = [
        [sg.Frame("", layout=deep_information, border_width=0, expand_x=True, expand_y=True)]
    ]

    ################# final gui layout #################

    _layout = [
        [sg.Frame("", game_frame, border_width=0, expand_x=True, expand_y=True,
                  size=(int(1920 * configs.SCALE_FACTOR), int(1080 * configs.SCALE_FACTOR))),
         sg.Frame("Game Information", game_stats, border_width=0, size=(350, int(1080 * configs.SCALE_FACTOR)))],
        [sg.Frame("Game Statistics", game_analysis, border_width=0, size=(int(1920 * configs.SCALE_FACTOR) + 350, 200))]
    ]

    window = sg.Window('Kicker Game', _layout)

    return window
