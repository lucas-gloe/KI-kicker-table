from datetime import datetime

import cv2
import keyboard
import numpy as np

from kalman_filter import KalmanFilter
from detect_field import FieldDetector
from detect_color import ColorTracker


class Game:
    """
    Class that tracks the game and his properties
    """
    ################################# INITIALIZE GAME CLASS ####################################

    def __init__(self, scale_percent):
        """
        Initialize game variables and properties
        """

        # constances
        self.SCALE_FACTOR = scale_percent
        self.RODWIDTH = 70 * self.SCALE_FACTOR / 100
        self.HALF_PLAYERS_WIDTH = 20 * self.SCALE_FACTOR / 100

        # variables
        self._start_time = None
        self._num_occurrences = 0
        self.results_from_calibration = True
        self._first_frame = True
        self._current_ball_position = None
        self._ball_positions = []
        self.predicted_value_added = False
        self._colors = None
        # self._colors = [[0, 116, 182, 7, 175, 255], [0, 167, 165, 23, 255, 255], [102, 66, 111, 120, 255, 255]]  # HSV
        self.__ball_color_from_calibration = None
        self._ball_color = None
        self.__team1_color_from_calibration = None
        self.__team2_color_from_calibration = None
        self.tracked_ball_color_for_GUI = "orange"
        self.tracked_team1_color_for_GUI = "orange"
        self.tracked_team2_color_for_GUI = "orange"
        self.__recalibrated_ball_color = None
        self.ball_was_found = False
        self._kf = KalmanFilter()
        self._df = FieldDetector(self.SCALE_FACTOR)
        self._dc = ColorTracker(self.SCALE_FACTOR)
        self._players_on_field = [False, False]
        self.field = None
        self.match_field = None
        self.goal1 = None
        self.goal2 = None
        self.throw_in_zone = None
        self.players_rods = None
        self._ranked = [[], []]
        self._team1_figures = None
        self._team2_figures = None
        self._predicted = (0, 0)
        self.counter_team1 = 0
        self.counter_team2 = 0
        self._ball_reenters_game = True
        self._goal1_detected = False
        self._goal2_detected = False
        self._results = True
        self.game_results = [[0, 0]]
        self.ratio_pxcm = None
        self._new_game = False
        self._show_contour = False
        self.last_speed = [0.0]

    def start(self):
        """
        start time counter for FPS tracking
        """
        self._start_time = datetime.now()
        return self

    def __counts_per_sec(self):
        """
        calculate average FPS output while videotracking
        """
        elapsed_time = (datetime.now() - self._start_time).total_seconds()
        return self._num_occurrences / elapsed_time if elapsed_time > 0 else 0

    ################################ INTERPRETATION OF THE FRAME ###############################

    def interpret_frame(self, frame, ball_color, field, team1_color, team2_color, ratio_pxcm):
        """
        interpret, track and draw game properties on the frame
        """
        # define colors from calibration
        self._num_occurrences += 1
        if self.results_from_calibration:
            self.__ball_color_from_calibration = ball_color
            self.__team1_color_from_calibration = team1_color
            self.__team2_color_from_calibration = team2_color
            self.field = field
            self.ratio_pxcm = ratio_pxcm
            self.results_from_calibration = False

        self._colors = [self.__ball_color_from_calibration, self.__team2_color_from_calibration, self.__team1_color_from_calibration]

        # Frame interpretation
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.match_field, self.goal1, self.goal2, self.throw_in_zone, self.players_rods = self._df.load_game_field_properties(
            field)
        self._team1_figures = self._track_players(1, hsv_img)
        self._team2_figures = self._track_players(2, hsv_img)
        self._track_ball(hsv_img)
        self._load_objects_colors()

        self._check_keybindings()
        self._check_variables()

        # track game stats
        self._count_game_score()
        self._detect_ball_reentering()
        self._reset_game()
        self._ball_speed_tracking()

        out_frame = frame.copy()

        # Draw results in frame
        #self._put_iterations_per_sec(out_frame, self.__counts_per_sec())
        self._draw_field_calibrations(out_frame)
        self._draw_ball(out_frame)
        self._draw_predicted_ball(out_frame)
        self._draw_figures(out_frame, self._team1_figures, 1)
        self._draw_figures(out_frame, self._team2_figures, 2)

        return out_frame

    #################################### FUNCTIONS FOR INTERPRETATION ###########################

    ####################################  TRACKING ##############################################

    def _track_ball(self, hsv_img):
        """
        look for objects in the dedicated mask, save the center position of the balls position
        """
        self._ball_color = self._colors[0]

        # if not self.results_from_calibration and self.ball_was_found:
        #     self._ball_color = [int((self._colors[0][0] + self.__recalibrated_ball_color[0]) / 2),
        #                         int((self._colors[0][1] + self.__recalibrated_ball_color[1]) / 2),
        #                         int((self._colors[0][2] + self.__recalibrated_ball_color[2]) / 2)]

        lower_color = np.asarray(self._ball_color)
        upper_color = np.asarray(self._ball_color)
        lower_color = lower_color - [10, 50, 50]
        upper_color = upper_color + [10, 50, 50]
        lower_color[lower_color < 0] = 0
        lower_color[lower_color > 255] = 255
        upper_color[upper_color < 0] = 0
        upper_color[upper_color > 255] = 255

        lower_color = np.array(lower_color)
        upper_color = np.array(upper_color)

        mask = cv2.inRange(hsv_img, lower_color, upper_color)
        mask = self.__smooth_ball_mask(mask)

        objects = self.__find_objects(mask, -1)
        objects.flatten()

        if len(objects) == 1:

            self.predicted_value_added = False

            x = objects[0][0]
            y = objects[0][1]
            w = objects[0][2]
            h = objects[0][3]

            # defining the center points for the case the detected contour is the ball
            center_x = int((x + (w / 2)))
            center_y = int((y + (h / 2)))

            # save the current position of the ball into an array
            self._current_ball_position = [center_x, center_y]

            # recalibrate ball color in current frame
            self.__recalibrated_ball_color = self._dc.recalibrate_ball_color(hsv_img, center_x, center_y,
                                                                             self._team1_figures, self._team2_figures,
                                                                             self.players_rods)
            self.ball_was_found = True

            self._predicted = self._kf.predict(center_x, center_y)

        elif len(objects) == 0:
            if not self.predicted_value_added:
                self._current_ball_position = (self._predicted[0], self._predicted[1])
                self.predicted_value_added = True
                self.ball_was_found = False
            else:
                print("Ball nicht erkannt")
                self._current_ball_position = [-1, -1]
                self.ball_was_found = False
        else:
            # self.__calculate_balls_position(objects)
            self._current_ball_position = [-1, -1]

        self._ball_positions.append(self._current_ball_position)

    def _track_players(self, team_number, hsv_img):
        """
        look for objects on the dedicated mask, sort them for position ranking and save them on players_positions
        """
        lower_color = np.asarray(self._colors[team_number])
        upper_color = np.asarray(self._colors[team_number])
        lower_color = lower_color - [10, 50, 50]  # good values (for test video are 10,50,50)
        upper_color = upper_color + [10, 50, 50]  # good values (for test video are 10,50,50)
        lower_color[lower_color < 0] = 0
        lower_color[lower_color > 255] = 255
        upper_color[upper_color < 0] = 0
        upper_color[upper_color > 255] = 255

        lower_color = np.array(lower_color)
        upper_color = np.array(upper_color)

        mask = cv2.inRange(hsv_img, lower_color, upper_color)

        objects = self.__find_objects(mask, team_number)
        if len(objects) >= 1:
            self._players_on_field[team_number - 1] = True
            self._ranked[team_number - 1] = self.__load_players_names(objects, team_number)
            player_positions = objects
            return player_positions

        elif len(objects) == 0:
            self._players_on_field[team_number - 1] = False
            print("Team " + str(team_number) + " nicht erkannt")
            return []

    def __find_objects(self, mask, team_number):
        """
        tracking algorithm to find the contours on the masks
        return: Contours on the masks
        source: https://www.computervision.zone/courses/learn-opencv-in-3-hours/
        """
        # outline the contours on the mask
        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        objects = []
        # looping over every contour which was found
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # saving contours properties in variables if a certain area is detected on the mask (to prevent blurring)
            if area > 100:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                white_contour = x, y, w, h
                objects.append(white_contour)

        objects = np.array(objects)
        if len(objects) > 1:
            objects = np.delete(objects, np.where(
                ((self.match_field[0][0] > objects[:, 0]) | (objects[:, 0] > self.match_field[1][0])) | (
                        (self.match_field[0][1] > objects[:, 1]) | (
                        objects[:, 1] > self.match_field[1][1]))), axis=0)

        if team_number == 1 or team_number == 2:
            objects = self.__remove_overlapping_bounding_boxes(objects, team_number)

        return objects

    def __smooth_ball_mask(self, mask):
        """
        The mask created inDetectBallPosition might be noisy.
        :param mask: The mask to smooth (Image with bit depth 1)
        :return: The smoothed mask
        """
        # create the disk-shaped kernel for the following image processing,
        r = 3
        kernel = np.ones((2 * r, 2 * r), np.uint8)
        for x in range(0, 2 * r):
            for y in range(0, 2 * r):
                if (x - r + 0.5) ** 2 + (y - r + 0.5) ** 2 > r ** 2:
                    kernel[x, y] = 0

        # remove noise
        # see http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def __load_players_names(self, objects, team_number):
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
            if team_number == 2:
                ranks = self.__reverse_ranks(ranks)
            return ranks

    def __get_rod(self, x):
        """
        check at which rod the detected shape is positioned
        """
        for i, rod in enumerate(self.players_rods):
            if rod[0][0] - self.RODWIDTH <= x <= rod[1][0] + self.RODWIDTH:
                return i
        return -1

    def __remove_overlapping_bounding_boxes(self, objects, team_number):
        """
        check if a player was detected with more than one bounding box. If so combine these boxes to one big box
        so every player is only detected once
        """
        _max_bounding_boxes = []
        for contour in objects:
            y_mid = contour[1] + contour[3] / 2

            left_corner_x = contour[0]
            left_corner_y = y_mid - self.HALF_PLAYERS_WIDTH
            right_corner_x = (contour[0] + contour[2])
            right_corner_y = y_mid + self.HALF_PLAYERS_WIDTH
            max_box = [[left_corner_x, left_corner_y], [right_corner_x, right_corner_y]]
            _max_bounding_boxes.append(max_box)

        np.zeros((len(_max_bounding_boxes), len(_max_bounding_boxes)))

        i = 0

        while i < len(_max_bounding_boxes):
            for j, max_box2 in enumerate(_max_bounding_boxes[i + 1:]):
                if (self.__get_rod((_max_bounding_boxes[i][0][0] + (
                        abs(_max_bounding_boxes[i][0][0] -_max_bounding_boxes[i][1][0]) / 2))) == self.__get_rod(
                        (max_box2[0][0] + (abs(max_box2[0][0] - max_box2[1][0]) / 2))) and (
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

    def __reverse_ranks(self, ranks):
        """
        Laod players ranks for the opposite team, so the counter always starts at the goalkeeper
        """
        reversed_ranks = []

        for rank in ranks:
            place = (len(ranks) - 1) - rank
            reversed_ranks.append(place)

        return reversed_ranks

    def _load_objects_colors(self):

        _tracked_ball_color_hsv = np.uint8([[self._colors[0]]])
        _tracked_team1_color_hsv = np.uint8([[self._colors[1]]])
        _tracked_team2_color_hsv = np.uint8([[self._colors[2]]])

        _tracked_ball_color_rgb = cv2.cvtColor(_tracked_ball_color_hsv, cv2.COLOR_HSV2RGB)
        _tracked_team1_color_rgb = cv2.cvtColor(_tracked_team1_color_hsv, cv2.COLOR_HSV2RGB)
        _tracked_team2_color_rgb = cv2.cvtColor(_tracked_team2_color_hsv, cv2.COLOR_HSV2RGB)

        self.tracked_ball_color_for_GUI = self.__rgb2hex(_tracked_ball_color_rgb)
        self.tracked_team1_color_for_GUI = self.__rgb2hex(_tracked_team1_color_rgb)
        self.tracked_team2_color_for_GUI = self.__rgb2hex(_tracked_team2_color_rgb)

    def __rgb2hex(self, color):
        return '#%02X%02X%02X' % (color[0][0][0], color[0][0][1], color[0][0][2])

    ##################################### GAME STATS ##############################################################

    def _check_keybindings(self):
        """
        Check the dedicated keys if their where pressed to set flag
        """
        if keyboard.is_pressed("c"):  # start calibration of kicker frame
            self._show_contour = True
        if keyboard.is_pressed("f"):  # end calibration of kicker frame
            self._show_contour = False
        if keyboard.is_pressed("n"):  # start new game
            self._new_game = True
            self._ball_positions = []
            self.last_speed = [0.0]
        if keyboard.is_pressed("p"):  # manual goal team1
            self.counter_team1 += 1
        if keyboard.is_pressed("l"):  # manual goal team2
            self.counter_team2 += 1


    def _check_variables(self):
        if len(self.last_speed) >= 200:
            self.last_speed.pop(0)
        if len(self._ball_positions) >= 1000:
            self._ball_positions.pop(0)

    def _count_game_score(self):
        """
        Count game score +1  of a certain team if a goal was shot
        """
        if len(self._ball_positions) > 1 and 0 < self._ball_positions[-2][0] < self.goal1[1][0] and self.goal1[0][1] < \
                self._ball_positions[-2][1] < self.goal1[1][1] and self._ball_positions[-1] == [-1, -1] and \
                self._ball_reenters_game:
            self._goal1_detected = True
            self.goalInCurrentFrame = True

        if len(self._ball_positions) > 1 and self._ball_positions[-2][0] > self.goal2[0][0] and self.goal2[0][1] < \
                self._ball_positions[-2][1] < self.goal2[1][1] and self._ball_positions[-1] == [-1, -1] and \
                self._ball_reenters_game:
            self._goal2_detected = True
            self.goalInCurrentFrame = True

        if self._goal1_detected and self.goalInCurrentFrame:
            self.counter_team1 += 1
            self.goalInCurrentFrame = False
        if self._goal2_detected and self.goalInCurrentFrame:
            self.counter_team2 += 1
            self.goalInCurrentFrame = False

    def _detect_ball_reentering(self):
        """
        Detect if the ball reenters the field in the middle section of the Kicker after a goal was shot
        """
        if self._goal1_detected or self._goal2_detected:
            if len(self._ball_positions) >= 2:
                if self.throw_in_zone[0][0] < self._ball_positions[-1][0] < self.throw_in_zone[1][0] and \
                        self._ball_positions[-2] == [-1, -1]:
                    self._goal1_detected = False
                    self._goal2_detected = False
                    self._results = True
                    self._ball_reenters_game = True

    def _reset_game(self):
        """
        Reset current game results to 0:0
        """
        if self._new_game and self._results:
            self.game_results.append([self.counter_team1, self.counter_team2])
            self.counter_team1 = 0
            self.counter_team2 = 0
            self._results = False

    def _ball_speed_tracking(self):
        """
        Measure the current speed of the ball
        """
        if len(self._ball_positions) >= 3 and self._ball_positions[-1] != [-1, -1]:
            # safe ball positions into an numpyArray
            current_position = np.array(self._ball_positions[-1])
            middle_position = np.array(self._ball_positions[-2])
            last_position = np.array(self._ball_positions[-3])

            # measure the distance between the positions of the ball
            distance1 = np.linalg.norm(current_position - middle_position)
            distance2 = np.linalg.norm(middle_position - last_position)
            distance = (distance1 + distance2) / 2

            # convert the travelled distance in pixel per frame into traveled distance km per hour
            cm_distance_per_frame = distance / self.ratio_pxcm
            m_distance_per_frame = cm_distance_per_frame / 100
            m_distance_per_second = m_distance_per_frame / self.__counts_per_sec()
            kmh = m_distance_per_second * 3.6

            kmh = round(kmh, 2)
            self.last_speed.append(kmh)

    #####################################  Drawing ON FRAME ######################################################

    # def _put_iterations_per_sec(self, tracked_frame, iterations_per_sec):
    #     """
    #     Add iterations per second text to lower-left corner of a frame.
    #     """
    #     cv2.putText(tracked_frame, "{:.0f} iterations/sec".format(iterations_per_sec), (50, 900),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))

    def _draw_field_calibrations(self, frame):
        """
        show football field contour for calibration on frame if a certain button was pressed
        """
        if self._show_contour:
            cv2.line(frame, (int(self.field[0][0]), int(self.field[0][1])),
                     (int(self.field[1][0]), int(self.field[1][1])),
                     (0, 255, 0), 2)
            cv2.line(frame, (int(self.field[2][0]), int(self.field[2][1])),
                     (int(self.field[3][0]), int(self.field[3][1])),
                     (0, 255, 0), 2)
            cv2.line(frame, (int(self.field[0][0]), int(self.field[0][1])),
                     (int(self.field[3][0]), int(self.field[3][1])),
                     (0, 255, 0), 2)
            cv2.line(frame, (int(self.field[1][0]), int(self.field[1][1])),
                     (int(self.field[2][0]), int(self.field[2][1])),
                     (0, 255, 0), 2)
            cv2.rectangle(frame, (int(self.goal1[0][0]), int(self.goal1[0][1])),  # Team1, orange
                          (int(self.goal1[1][0]), int(self.goal1[1][1])),
                          (0, 255, 0), 2)
            cv2.rectangle(frame, (int(self.goal2[0][0]), int(self.goal2[0][1])),  # Team2, blau
                          (int(self.goal2[1][0]), int(self.goal2[1][1])),
                          (0, 255, 0), 2)
            cv2.rectangle(frame, (int(self.throw_in_zone[0][0]), int(self.throw_in_zone[0][1])),
                          (int(self.throw_in_zone[1][0]), int(self.throw_in_zone[1][1])),
                          (0, 255, 0), 2)
            for rod in self.players_rods:
                cv2.rectangle(frame, (int(rod[0][0]), int(rod[0][1])),
                              (int(rod[1][0]), int(rod[1][1])), (0, 255, 255), 2)
            # for area in self._granted_players_areas_around_rods:
            #     cv2.rectangle(frame, (int(area[0][0]), int(area[0][1])),
            #                   (int(area[1][0]), int(area[1][1])), (0, 255, 255), 2)

    def _draw_ball(self, frame):
        """
        Draw a circle at the balls position and name the Object "ball"
        """
        # draw a circle for the ball
        if self._current_ball_position != [-1, -1]:
            cv2.circle(frame, (self._current_ball_position[0], self._current_ball_position[1]), int(16*self.SCALE_FACTOR/100), (0, 255, 0), 2)
            cv2.putText(frame, "Ball", (self._current_ball_position[0], self._current_ball_position[1]),
                        cv2.FONT_HERSHEY_PLAIN, 1, (30, 144, 255), 2)

    def _draw_predicted_ball(self, frame):
        """
        Draw a circle at the predicted balls position if there is no ball
        detected in the frame and name the Object "ball"
        """
        if self._current_ball_position == [-1, -1]:
            cv2.circle(frame, (self._predicted[0], self._predicted[1]), int(16*self.SCALE_FACTOR/100), (0, 255, 255), 2)

    def _draw_figures(self, frame, player_positions, team_number):
        """
        Draw a rectangle at the players position and name it TeamX
        """
        if self._players_on_field[team_number - 1]:
            for i, player_position in enumerate(player_positions):
                cv2.rectangle(frame, (int(player_position[0][0]), int(player_position[0][1])),
                              (int(player_position[1][0]), int(player_position[1][1])),
                              (0, 255, 0), 2)
                cv2.putText(frame,
                            ("Team" + str(team_number) + ", " + str(self._ranked[team_number - 1][i])),
                            (int(player_position[0][0]), int(player_position[0][1])), cv2.FONT_HERSHEY_PLAIN, 1,
                            (30, 144, 255), 2)

################# UPDATING GUI ###########################

    def check_game_var(self, _type):
        """
        Get the class variables
        :param _type: String to choose the variable
        :return: The requested variable, empty string if requested name is
        unavailable
        """
        if "-team_1-" == _type:
            return self.tracked_team1_color_for_GUI
        elif '-team_2-' == _type:
            return self.tracked_team2_color_for_GUI
        elif '-ball-' == _type:
            return self.tracked_ball_color_for_GUI
        elif "-counts_per_second-" == _type:
            return self.__counts_per_sec()
        elif '-score_team_1-' == _type:
            return self.counter_team1
        elif '-score_team_2-' == _type:
            return self.counter_team2
        elif '-ball_speed-' == _type:
            return max(self.last_speed)
        elif "-last_game_team1-" == _type:
            return self.game_results[-1][0]
        elif "-last_game_team2-" == _type:
            return self.game_results[-1][1]
        elif "-second_last_game_team1-" == _type:
                if len(self.game_results) >= 2:
                    return self.game_results[-2][0]
                else:
                    return ""
        elif "-second_last_game_team2-" == _type:
            if len(self.game_results) >= 2:
                return self.game_results[-2][1]
            else:
                return ""
        elif "-third_last_game_team1-" == _type:
            if len(self.game_results) >= 3:
                return self.game_results[-3][0]
            else:
                ""
        elif "-third_last_game_team2-" == _type:
            if len(self.game_results) >= 3:
                return self.game_results[-3][1]
            else:
                ""
        else:
            return ""  # False

