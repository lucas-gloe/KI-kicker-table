import cv2
import numpy as np
import PySimpleGUI as sg
import configs


def calibrate_before_first_image():
    """
    initialize the gui to show data output
    Parameters:
    Returns:
        window(obj): gui window object
    """
    window = initialize_gui_layout(configs.FONT)

    return window


def calibrate(calibration_image, game_config):
    """
    get calibration information of the current game
    Parameters:
        calibration_image(np.ndarray): calibration image taken from the foosball
        game_config(dict): calibration values for current game
    Returns:
    """
    # read angle of soccer table rotation from the outer edges of the table
    angle = get_angle(calibration_image)
    # read center point and ratio from middle circle of match field
    center, ratio_pxcm = get_center_scale(calibration_image)
    # read total table field
    field = calc_field(angle, center, ratio_pxcm)
    # read color from tracked objects
    ball_color, team1_color, team2_color = calibrate_color(calibration_image, center)
    # read match field and other game properties from total table field
    match_field, goal1, goal2, throw_in_zone, players_rods = load_game_field_properties(field)
    # convert tracked hsv colors to rgb colors on gui
    gui_ball_color, gui_team1_color, gui_team2_color = convert_tracked_hsv_colors(ball_color, team1_color, team2_color)

    # update all calibrated values to shared dict
    game_config.update({
        'goal1': goal1,
        'goal2': goal2,
        'throw_in_zone': throw_in_zone,
        'players_rods': players_rods,
        'angle': angle,
        'field': field,
        'match_field': match_field,
        'center': center,
        'ratio_pxcm': ratio_pxcm,
        'ball_color': ball_color,
        'team1_color': team1_color,
        'team2_color': team2_color,
        'gui_ball_color': gui_ball_color,
        'gui_team1_color': gui_team1_color,
        'gui_team2_color': gui_team2_color
    })


def get_angle(calibration_image):
    """
    define angle of table soccer for playground definition
    source: https://github.com/StudentCV/TableSoccerCV
    Parameters:
        calibration_image(np.ndarray): calibration image taken from the foosball
    Returns:
        angle(int): angle of foosball table rotation
    """
    # convert color schema of calibration image
    rgb = cv2.cvtColor(calibration_image, cv2.COLOR_HSV2BGR)
    angle = 0
    count = 0
    # convert color schema of calibration image
    gray = cv2.cvtColor(cv2.cvtColor(calibration_image, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
    # convert display type of image by certain filter
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # count outer edges of table
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 110)

    if lines.shape[0]:
        line_count = lines.shape[0]
    else:
        raise Exception('field not detected')

    # check angle of rotation of outer edges
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
    read the center circle of table soccer and compare center size to given camera resolution
    source: https://github.com/StudentCV/TableSoccerCV
    Parameters:
        calibration_image(np.ndarray): calibration image taken from the foosball
    Returns:
        center(int): center point from foosball table
        ratio_pxcm(float): ratio of px to cm
    """
    # convert color schema of calibration image
    gray = cv2.cvtColor(calibration_image, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    # check for middle circle with a certain size based on the scale factor from configs
    if configs.SCALE_FACTOR >= 0.4:
        gray = cv2.GaussianBlur(gray, (5, 5), 1)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=50, param2=30, minRadius=30, maxRadius=100)
    else:
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=int(50 * configs.SCALE_FACTOR),
                                   param2=int(30 * configs.SCALE_FACTOR), minRadius=int(30 * configs.SCALE_FACTOR),
                                   maxRadius=int(100 * configs.SCALE_FACTOR))
    center_circle = (0, 0, 0)  # centerx, centery, radius
    min_dist = 0xFFFFFFFFFFF
    # check all found circle for that one that is in the middle of the field the most
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

    # create another calibration with the found center circle
    cv2.circle(gray, (int(center_circle[0]), int(center_circle[1])), int(center_circle[2]), (0, 255, 0), 2)

    cv2.imwrite("gray.JPG", gray)

    #calculate the ratio
    center = int(center_circle[0]), int(center_circle[1])
    radius = center_circle[2]
    ratio_pxcm = radius / 9.4

    return [center, ratio_pxcm]


def calc_field(angle, center, ratio_pxcm):
    """
    take calculate arguments to create table soccer playground
    part of code from source: https://github.com/StudentCV/TableSoccerCV
    Parameters:
        angle(int): angle of foosball table rotation
        center(list): list with x and y of center point from foosball table
        ratio_pxcm(float): ratio of px to cm
    Returns:
        field(list): contains all 4 corners from calibrated field with the following sequence: top left, top right, bottom left, bottom right
    """

    field = []
    half_field_width = configs.HALF_FIELD_WIDTH
    half_field_height = configs.HALF_FIELD_HEIGHT

    angle_radial_scale = np.radians(angle)

    corners = {
        0: ["-", "+", "-", "-"],
        1: ["+", "+", "+", "-"],
        2: ["+", "-", "+", "+"],
        3: ["-", "-", "-", "+"]
    }

    # save every corner of table field to render lines between all of the corners. these lines a form the field
    for corner in range(4):
        x = eval(str(center[0]) + corners[corner][0] + str(half_field_width * ratio_pxcm) + corners[corner][1] + str(
            np.tan(angle_radial_scale) * (half_field_height * ratio_pxcm)))
        y = eval(
            str(center[1]) + corners[corner][2] + str(np.tan(angle_radial_scale) * (half_field_width * ratio_pxcm)) +
            corners[corner][3] + str(half_field_height * ratio_pxcm))

        current_corner = [int(x), int(y)]
        field.append(current_corner)

    return field


def calibrate_color(calibration_image, center):
    """
    The taken image will be used to read the colors from the defined calibration positions.
    source: https://github.com/StudentCV/TableSoccerCV
    Parameters:
        calibration_image(np.ndarray): calibration image taken from the foosball
        center(list): list with x and y of center point from foosball table
    Returns:
        ball_color(tuple): calibrated ball color
        team1_color(tuple): calibrated team 1 color
        team2_color(tuple): calibrated team 2 color
    """

    calibration_image = cv2.cvtColor(calibration_image, cv2.COLOR_BGR2HSV)

    # The initialization is done with only a small part of the image around the center spot.
    x1 = int(round(center[0] - (calibration_image.shape[1] / 10), 0))
    x2 = int(round(center[0] + (calibration_image.shape[1] / 10), 0))
    y1 = int(round(center[1] - (calibration_image.shape[0] / 10), 0))
    y2 = int(round(center[1] + (calibration_image.shape[0] / 10), 0))

    # crop the calibration image by given values
    cropped_hsv_img = calibration_image[y1:y2, x1:x2]

    # save the area around the middle circle
    cv2.imwrite("cropped_calibration_img.JPG", cropped_hsv_img)

    # calibrate colors based on the cropped image
    ball_color = _calibrate_ball_color(cropped_hsv_img)
    team1_color = _calibrate_team_color(cropped_hsv_img, 1)
    team2_color = _calibrate_team_color(cropped_hsv_img, 2)

    return [ball_color, team1_color, team2_color]


def _calibrate_ball_color(cropped_hsv_img):
    """
    Measures the color around the balls position.
    part of code from source: https://github.com/StudentCV/TableSoccerCV
    Parameters:
        cropped_hsv_img(np.ndarray): cropped calibration image taken from calibration image
    Returns:
        ball_color(tuple): calibrated ball color
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

    # Create a mask for the areas with a color similar to the center pixel
    mask = cv2.inRange(cropped_hsv_img, lower_border, upper_border)

    # Average the color values of the masked area
    colors = cropped_hsv_img[mask == 255]
    h_mean = int(round(np.mean(colors[:, 0])))
    s_mean = int(round(np.mean(colors[:, 1])))
    v_mean = int(round(np.mean(colors[:, 2])))

    av = [h_mean, s_mean, v_mean]
    ball_color = tuple(av)

    return ball_color


def _calibrate_team_color(cropped_hsv_img, team_number):
    """
    Measures the color around the teams positions
    part of code from source: https://github.com/StudentCV/TableSoccerCV
    Parameters:
        cropped_hsv_img(np.ndarray): cropped calibration image taken from calibration image
        team_number(int): number for team for defining calibration area
    Returns:
        team_color(tuple(list)): calibrated teams color
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
    colors = cropped_hsv_img[y_player - int(6 * configs.SCALE_FACTOR):y_player + int(5 * configs.SCALE_FACTOR),
             x_player - int(6 * configs.SCALE_FACTOR):x_player + int(5 * configs.SCALE_FACTOR)]
    lower_border_arr = [np.min(colors[:, :, 0]), np.min(colors[:, :, 1]), np.min(colors[:, :, 2])]
    upper_border_arr = [np.max(colors[:, :, 0]), np.max(colors[:, :, 1]), np.max(colors[:, :, 2])]

    lower_border_arr = np.array(lower_border_arr)
    upper_border_arr = np.array(upper_border_arr)

    lower_border = tuple(lower_border_arr.tolist())
    upper_border = tuple(upper_border_arr.tolist())

    # Create a mask for the areas with a color similar to the center pixel
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
    Parameters:
        field(list): contains all 4 corners from calibrated field with the following sequence: top left, top right, bottom left, bottom right
    Returns:
        match_field(list): contains all 4 corners from calibrated match field with the following sequence: top left, top right, bottom left, bottom right
        goal1(list): contains all 4 corners from goal1 field with the following sequence: top left, top right, bottom left, bottom right
        goal2(list): contains all 4 corners from goal2 field with the following sequence: top left, top right, bottom left, bottom right
        throw_in_zone(list): contains all 4 corners from throw in zone field with the following sequence: top left, top right, bottom left, bottom right
        players_rods(lst): contains all 4 corners from players rods field with the following sequence: top left, top right, bottom left, bottom right
    """
    # calculate match_field
    match_field = np.array([[int(field[0][0]), int(field[0][1])],
                            [int(field[2][0]), int(field[2][1])]])

    # calculate goal zones
    goal_middle_y = int((match_field[1][1] - match_field[0][1]) / 2) + match_field[0][1]

    goal1 = np.array([[int(match_field[0][0]), goal_middle_y - int(configs.HALF_WIDTH_GOAL * configs.SCALE_FACTOR)],
                      [int(match_field[0][0] + int(100 * configs.SCALE_FACTOR)),
                       goal_middle_y + int(configs.HALF_WIDTH_GOAL * configs.SCALE_FACTOR)]])

    goal2 = np.array([[int(match_field[1][0] - int(100 * configs.SCALE_FACTOR)),
                       goal_middle_y - int(configs.HALF_WIDTH_GOAL * configs.SCALE_FACTOR)],
                      [int(match_field[1][0]), goal_middle_y + int(configs.HALF_WIDTH_GOAL * configs.SCALE_FACTOR)]])

    # calculate throw in zone
    throw_in_zone = np.array([[int(match_field[0][0] + int(400 * configs.SCALE_FACTOR)), int(match_field[0][1])],
                              [int(match_field[1][0] - int(400 * configs.SCALE_FACTOR)), int(match_field[1][1])]])

    # calculate rods
    number_of_rods = 8
    warp_reduction = [12, 9, 9, 3, -3, -9, -9, -12]
    players_rods = []

    distance_between_rods = (np.linalg.norm(match_field[1][0] - match_field[0][0])) / number_of_rods

    for rod in range(number_of_rods):
        current_rod = [[int(
            match_field[0][0] + ((rod + 0.5) * distance_between_rods - int(16 * configs.SCALE_FACTOR)) + warp_reduction[
                rod]), int(match_field[0][1])],
                       [int(match_field[0][0] + ((rod + 0.5) * distance_between_rods + int(16 * configs.SCALE_FACTOR)) +
                            warp_reduction[rod]), int(match_field[1][1])]]
        players_rods.append(current_rod)

    return match_field, goal1, goal2, throw_in_zone, players_rods


def convert_tracked_hsv_colors(ball_color, team1_color, team2_color):
    """
    convert hsv colors to showable rgb colors fot the gui
    Parameters:
        ball_color(tuple(list)): calibrated ball color in HSV
        team1_color(tuple(list)): calibrated team 1 color in HSV
        team2_color(tuple(list)): calibrated team 2 color in HSV
    Returns:
        ball_color(string): calibrated ball color in RGB
        team1_color(string): calibrated team 1 color in RGB
        team2_color(string): calibrated team 2 color in RGB
    """
    # read calibrated colors
    _tracked_ball_color_hsv = np.uint8([[ball_color]])
    _tracked_team1_color_hsv = np.uint8([[team1_color]])
    _tracked_team2_color_hsv = np.uint8([[team2_color]])

    # convert color into right color type
    _tracked_ball_color_rgb = cv2.cvtColor(_tracked_ball_color_hsv, cv2.COLOR_HSV2RGB)
    _tracked_team1_color_rgb = cv2.cvtColor(_tracked_team1_color_hsv, cv2.COLOR_HSV2RGB)
    _tracked_team2_color_rgb = cv2.cvtColor(_tracked_team2_color_hsv, cv2.COLOR_HSV2RGB)

    # convert color into readable values for gui
    gui_ball_color = _rgb2hex(_tracked_ball_color_rgb)
    gui_team1_color = _rgb2hex(_tracked_team1_color_rgb)
    gui_team2_color = _rgb2hex(_tracked_team2_color_rgb)

    return gui_ball_color, gui_team1_color, gui_team2_color


def _rgb2hex(color):
    """
    converts RGB colors to HEX code
    Parameters:
        color(np.array): RGB color value
    Returns:
        color(string): HEX color value
    """
    return '#%02X%02X%02X' % (color[0][0][0], color[0][0][1], color[0][0][2])


def initialize_gui_layout(FONT):
    """
    define design and properties from gui
    Parameters:
        FONT(string): font value for gui
    Returns:
        window(obj): gui window object
    """
    # layout of the GUI
    sg.theme('Reddit')

    ################# whole Game frame #######################################

    game_frame = [

        [sg.Image(filename="", key="-frame-")]
    ]
    ################# right frame with basic game infos #################

    # part from upper right frame, scoring, ball speed, fps,

    game_score_and_speed = [
        [sg.Text('SIT Smart Kicker', text_color='orange', font=(FONT, 30))],
        [sg.Button('Tor+1', key="-manual_game_counter_team_1_up-", button_color='grey', font=(FONT, 8)),
         sg.Button('Tor-1', key="-manual_game_counter_team_1_down-", button_color='grey', font=(FONT, 8)),
         sg.Button('Tor+1', key="-manual_game_counter_team_2_up-", button_color='grey', font=(FONT, 8)),
         sg.Button('Tor-1', key="-manual_game_counter_team_2_down-", button_color='grey', font=(FONT, 8))],
        [sg.Text("", key='-score_team_1-', font=(FONT, 45)), sg.Text(" : ", font=(FONT, 20)),
         sg.Text("", key='-score_team_2-', font=(FONT, 45))],
        [sg.Text("Team 1", font=(FONT, 15)),
         sg.Text("Team 2", font=(FONT, 15))],
        [sg.Text('Geschätzte Ballgeschwindigkeit:', font=(FONT, 10)),
         sg.Text("Nicht festgelegt", key='-ball_speed-', font=(FONT, 10))],
        [sg.Text('FPS:', font=(FONT, 10)),
         sg.Text("0", key='-fps-', font=(FONT, 10))],
        [sg.Text('Drücke S, um das Konfigurationsbild zu speichern', key='-config_img-', font=(FONT, 10))]
    ]

    # part from upper right frame, team colors

    game_configuration = [
        [sg.Text("Spielkonfiguration", text_color='orange', font=(FONT, 15))],
        [sg.Text("Team 1", font=(FONT, 10), text_color='white', background_color="orange", key="-team_1-",
                 expand_x=True, justification='c')],
        [sg.Text("Team 2", font=(FONT, 10), text_color='white', background_color="orange", key="-team_2-",
                 expand_x=True, justification='c')],
        [sg.Text("Ball", font=(FONT, 10), text_color='white', background_color="orange", key="-ball-",
                 expand_x=True, justification='c')]
    ]

    # part from upper right frame, game history

    last_games = [
        [sg.Text("Letzte Spiele", text_color='orange', font=(FONT, 15))],
        [sg.Text("", key='-last_game_team1-', font=(FONT, 10)), sg.Text(" : ", font=(FONT, 10)),
         sg.Text("", key='-last_game_team2-', font=(FONT, 10))],
        [sg.Text("", key='-second_last_game_team1-', font=(FONT, 10)), sg.Text(" : ", font=(FONT, 10)),
         sg.Text("", key='-second_last_game_team2-', font=(FONT, 10))],
        [sg.Text("", key='-third_last_game_team1-', font=(FONT, 10)), sg.Text(" : ", font=(FONT, 10)),
         sg.Text("", key='-third_last_game_team2-', font=(FONT, 10))]
    ]

    # whole upper right

    configurations = [
        [sg.Frame("", game_configuration, expand_x=True, expand_y=True, element_justification='c'),
         sg.Frame("", last_games, expand_x=True, expand_y=True, element_justification='c')]
    ]

    basic_information = [
        [sg.Frame("", game_score_and_speed, expand_x=True, expand_y=True, element_justification='c')],
        [sg.Frame("", configurations, border_width=0, expand_x=True, expand_y=True)]
    ]

    game_stats = [
        [sg.Frame("", layout=basic_information, border_width=0, expand_x=True, expand_y=True)]
    ]

    ################# down frame with advanced infos #################

    # part from bottom frame, key bindings

    key_bindings = [
        [sg.Text("Tastenbelegungen", text_color='orange', font=(FONT, 15))],
        [sg.Text('Drücke N, um ein neues Spiel zu starten', font=(FONT, 10))],
        [sg.Text("Drücke C, um Kicker anzuzeigen, F um auszublenden", font=(FONT, 10))],
        [sg.Text("Drücke A, um Konturen anzuzeigen, D zum auszublenden", font=(FONT, 10))],
        [sg.Text("Drücke M, um in den manuellen Modus zu wechseln, l für Automatik", font=(FONT, 10))],
        [sg.Text("Drücke K, um die Frames zu durchlaufen", font=(FONT, 10))]
    ]

    # part from bottom frame, heat_map

    heat_map = [
        [sg.Text("Platzhalter", text_color='orange', font=(FONT, 15))]
    ]

    # part from bottom frame

    blank_frame = [
        [sg.Text("Platzhalter", text_color='orange', font=(FONT, 15))]
    ]

    # part from bottom frame

    blank_frame2 = [
        [sg.Text("Platzhalter", text_color='orange', font=(FONT, 15))]
    ]

    # whole bottom frame

    deep_information = [
        [sg.Frame("", key_bindings, expand_x=True, expand_y=True, element_justification='c'),
         sg.Frame("", heat_map, expand_x=True, expand_y=True, element_justification='c'),
         sg.Frame("", blank_frame, expand_x=True, expand_y=True, element_justification='c'),
         sg.Frame("", blank_frame2, expand_x=True, expand_y=True, element_justification='c')]
    ]

    game_analysis = [
        [sg.Frame("", layout=deep_information, border_width=0, expand_x=True, expand_y=True)]
    ]

    ################# final gui layout #################

    _layout = [
        [sg.Frame("", game_frame, border_width=0, expand_x=True, expand_y=True,
                  size=(int(1920 * configs.SCALE_FACTOR), int(1080 * configs.SCALE_FACTOR))),
         sg.Frame("Spielinformationen", game_stats, border_width=0, size=(350, int(1080 * configs.SCALE_FACTOR)))],
        [sg.Frame("Spielstatistiken", game_analysis, border_width=0, size=(int(1920 * configs.SCALE_FACTOR) + 350, 200))]
    ]

    window = sg.Window('Tischfußballspiel', _layout)

    return window
