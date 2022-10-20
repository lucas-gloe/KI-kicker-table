from datetime import datetime

import cv2
import keyboard
import numpy as np

from kalman_filter import KalmanFilter


class Game:
    """
    Class that tracks the game and his properties
    """

    ################################# INITIALIZE GAME CLASS ####################################
    def __init__(self):
        """
        Initialize game variables and properties
        """
        self._start_time = None
        self._num_occurrences = 0
        self._first_frame = True
        self._values = [[], []]
        self._pixel = (0, 0, 0)
        self._current_ball_position = None
        self._ball_positions = []
        # self.ball_color = farbe aus calibrierung, daher erste stelle in colors unwichtig
        self._colors = [[0, 116, 182, 7, 175, 255], [0, 167, 165, 23, 255, 255], [102, 66, 111, 120, 255, 255]]  # HSV
        self.ball_color = []
        self._kf = KalmanFilter()
        self._players_on_field = False
        self.field = []
        self.match_field = []
        self.goal1 = []
        self.goal2 = []
        self.throw_in_zone = []
        self._ranked = [[], []]
        self._player1_figures = []
        self._player2_figures = []
        self._predicted = (0, 0)
        self._counter_team1 = 0
        self._counter_team2 = 0
        self._ball_out_of_game = True
        self._goal1_detected = False
        self._goal2_detected = False
        self._results = True
        self._game_results = [[0, 0]]
        self._new_game = False
        self._show_contour = False
        self._last_speed = [0.0]

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

    def interpret_frame(self, frame, ball_color, field):
        """
        interpret, track and draw game properties on the frame
        """
        # Frame interpretation
        self._num_occurrences += 1
        self.ball_color = ball_color
        self.field = field
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.load_game_field_properties()
        self._track_ball(hsv_img)
        self._player1_figures = self._track_players(1, 0, hsv_img)
        self._player2_figures = self._track_players(2, 1, hsv_img)

        self._check_keybindings()

        # track game stats
        self._count_game_score()
        self._detect_ball_reentering()
        self._reset_game()
        self._ball_speed_tracking()

        out_frame = frame.copy()

        # Draw results in frame
        self._put_iterations_per_sec(out_frame, self.__counts_per_sec())
        self._draw_contour_on_kicker(out_frame)
        self._draw_ball(out_frame)
        self._draw_predicted_ball(out_frame)
        self._draw_figures(out_frame, self._player1_figures, team=1)
        self._draw_figures(out_frame, self._player2_figures, team=2)
        self._show_game_score(out_frame)
        self._show_last_games(out_frame)
        self._show_ball_speed(out_frame)

        return out_frame

    #################################### FUNCTIONS FOR INTERPRETATION ###########################

    ####################################  TRACKING ##############################################

    def load_game_field_properties(self):
        """

        """
        self.match_field = np.array([[int(self.field[0][0]), int(self.field[0][1])],
                                     [int(self.field[2][0]), int(self.field[2][1])]])
        self.goal1 = np.array([[int(self.match_field[0][0] - 20), int((np.linalg.norm(
            (self.match_field[1][1] - self.match_field[0][1]) / 2) + self.match_field[0][1]) - 120)],
                               [int(self.match_field[0][0] + 20), int((np.linalg.norm(
                                   (self.match_field[1][1] - self.match_field[0][1]) / 2) + self.match_field[0][
                                                                           1]) + 120)]])
        self.goal2 = np.array([[int(self.match_field[1][0] - 20), int((np.linalg.norm(
            (self.match_field[1][1] - self.match_field[0][1]) / 2) + self.match_field[0][1]) - 120)],
                               [int(self.match_field[1][0] + 20), int((np.linalg.norm(
                                   (self.match_field[1][1] - self.match_field[0][1]) / 2) + self.match_field[0][
                                                                           1]) + 120)]])
        self.throw_in_zone = np.array([[int(self.match_field[0][0] + 400), int(self.match_field[0][1])],
                                       [int(self.match_field[1][0] - 400), int(self.match_field[1][1])]])

    def _track_ball(self, hsv_img):
        """
        look for objects in the dedicated mask, save the center position of the balls position
        """
        lower_color = np.asarray(self.ball_color)
        upper_color = np.asarray(self.ball_color)
        lower_color = lower_color - [10, 50, 50]  # good values (for test video are 10,50,50)
        upper_color = upper_color + [10, 50, 50]  # good values (for test video are 10,50,50)
        lower_color[lower_color < 0] = 0
        lower_color[lower_color > 255] = 255
        upper_color[upper_color < 0] = 0
        upper_color[upper_color > 255] = 255

        lower_color = np.array(lower_color)
        upper_color = np.array(upper_color)

        mask = cv2.inRange(hsv_img, lower_color, upper_color)
        mask = self.__smooth_ball_mask(mask)

        #mask = cv2.inRange(hsv_img, np.array(self._colors[0][0:3]), np.array(self._colors[0][3:6]))
        objects = self.__find_objects(mask)
        
        if len(objects) == 1:
            x = objects[0][0]
            y = objects[0][1]
            w = objects[0][2]
            h = objects[0][3]

            # defining the center points for the case the detected contour is the ball
            center_x = int((x + (w / 2)))
            center_y = int((y + (h / 2)))

            # save the current position of the ball into an array
            self._current_ball_position = [center_x, center_y]

            self._predicted = self._kf.predict(center_x, center_y)

        elif len(objects) == 0:
            print("Ball nicht erkannt")
            self._current_ball_position = [-1, -1]
        else:
            # self.__calculate_balls_position(objects)
            self._current_ball_position = [-1, -1]

        self._ball_positions.append(self._current_ball_position)

    def _track_players(self, team_number, team_rank, hsv_img):
        """
        look for objects on the dedicated mask, sort them for position ranking and save them on players_positions
        """
        player_positions = []
        mask = cv2.inRange(hsv_img, np.array(self._colors[team_number][0:3]), np.array(self._colors[team_number][3:6]))
        objects = self.__find_objects(mask)
        if len(objects) >= 1:
            self._players_on_field = True
            self._ranked[team_rank] = self.__load_players_names(objects, team_rank)
            player_positions = objects
            return player_positions

        elif len(objects) == 0:
            self._players_on_field = False
            print("Spieler nicht erkannt")
            return []

    def __find_objects(self, mask):
        """
        tracking algorithm to find the contours on the masks
        return: Contours on the masks
        source: https://www.computervision.zone/courses/learn-opencv-in-3-hours/
        """
        # outline the contours on the mask
        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        x, y, w, h = 0, 0, 0, 0
        white_contour = []
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

    def __load_players_names(self, objects, team_rank):
        """
        take the position of alle players of a team and rank them sorted from the position of the field
        Return: Array with sorted list of players ranks
        """
        if len(objects) > 0:
            position_matrix = np.array(objects)
            valued_matrix = position_matrix[:, 0] * 10 + position_matrix[:, 1]
            sorted_valued_matrix = valued_matrix.argsort()
            ranks = np.empty_like(sorted_valued_matrix)
            ranks[sorted_valued_matrix] = np.arange(len(valued_matrix))
            if team_rank == 1:
                ranks = self.__reverse_ranks(ranks)
            return ranks

    def __calculate_balls_position(self, objects):
        if len(self._ball_positions) > 0:
            if self._ball_positions[-1] == [-1, -1]:
                self._current_ball_position = [-1, -1]

            if self._ball_positions[-1] != [-1, -1]:
                position_matrix = np.array(objects)
                ball_positions = np.array(self._ball_positions)
                ball_positions = np.delete(ball_positions, np.where(ball_positions[:] == [-1, -1]), axis=0)
                potenziellebaelle = position_matrix[
                    np.where(np.abs(position_matrix[:, 2] - position_matrix[:, 3]) <= 5)]
                if len(potenziellebaelle) > 0:
                    naehezumletztenball = np.abs(potenziellebaelle[:, 0] - ball_positions[-1, 0])
                    wahrscheinlicheposition = potenziellebaelle[(np.argmin(naehezumletztenball))]
                    center_x = int((wahrscheinlicheposition[0] + (wahrscheinlicheposition[2] / 2)))
                    center_y = int((wahrscheinlicheposition[1] + (wahrscheinlicheposition[3] / 2)))

                    self.current_ball_position = [center_x, center_y]
        else:
            self._current_ball_position = [-1, -1]

    def __reverse_ranks(self, ranks):
        gesamtlaenge = len(ranks)
        reversed_ranks = []

        for rank in ranks:
            place = (gesamtlaenge - 1) - rank
            reversed_ranks.append(place)

        return reversed_ranks

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
            self._last_speed = []

    def _count_game_score(self):
        """
        Count game score +1  of a certain team if a goal was shot
        """
        if len(self._ball_positions) > 1 and 0 < self._ball_positions[-2][0] < 250 and 430 < self._ball_positions[-2][
            1] < 670 and self._ball_positions[-1] == [-1, -1] and self._ball_out_of_game:
            self._goal1_detected = True
            self.goalInCurrentFrame = True

        if len(self._ball_positions) > 1 and self._ball_positions[-2][0] > 1600 and 430 < self._ball_positions[-2][
            1] < 660 and self._ball_positions[-1] == [-1, -1] and self._ball_out_of_game:
            self._goal2_detected = True
            self.goalInCurrentFrame = True

        if self._goal1_detected and self.goalInCurrentFrame:
            self._counter_team1 += 1
            self._ball_out_of_game = False
        if self._goal2_detected and self.goalInCurrentFrame:
            self._counter_team2 += 1
            self._ball_out_of_game = False

    def _detect_ball_reentering(self):
        """
        Detect if the ball reenters the field in the middle section of the Kicker after a goal was shot
        """
        if self._goal1_detected or self._goal2_detected:
            if 700 < self._ball_positions[-1][0] < 1270 and self._ball_positions[-2] == [-1, -1]:
                self._goal1_detected = False
                self._goal2_detected = False
                self._results = True
                self._ball_out_of_game = True

    def _reset_game(self):
        """
        Reset current game results to 0:0
        """
        if self._new_game and self._results:
            self._game_results.append([self._counter_team1, self._counter_team2])
            self._counter_team1 = 0
            self._counter_team2 = 0
            self._results = False

    def _ball_speed_tracking(self):
        """
        Measure the current speed of the ball
        """
        if len(self._ball_positions) >= 3 and self._ball_positions[-1] != [-1, -1]:
            # safe the current ball position into an numpyArray
            current_position = np.array(self._ball_positions[-1])
            # safe the current-1 ball position into an numpyArray
            middle_position = np.array(self._ball_positions[-2])
            # safe the current-2 ball position into an numpyArray
            last_position = np.array(self._ball_positions[-3])
            # measure the distance between the last and last-1 point of the ball
            distance1 = np.linalg.norm(current_position - middle_position)
            # measure the distance between the last-1 and last-2 point of the ball
            distance2 = np.linalg.norm(middle_position - last_position)
            # calculate the travelled distance between 1 frame
            distance = (distance1 + distance2) / 2
            # convert the travelled distance into real speed measuring
            # ->Camera view on Kicker Table is 1300px x 740p at 120, Kicker Table is 1,20m x 0,68m relationship: ~1:1083
            real_distance_per_frame = distance / 1083
            real_distance_per_second = real_distance_per_frame * 60
            kmh = real_distance_per_second * 3.6
            kmh = round(kmh, 2)
            self._last_speed.append(kmh)

    #####################################  PRINTING ON FRAME ######################################################

    def _put_iterations_per_sec(self, tracked_frame, iterations_per_sec):
        """
        Add iterations per second text to lower-left corner of a frame.
        """
        cv2.putText(tracked_frame, "{:.0f} iterations/sec".format(iterations_per_sec), (50, 900),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))

    def _draw_contour_on_kicker(self, frame):
        """
        Add football field contour for calibration on frame
        """
        if self._show_contour:
            # cv2.rectangle(frame, (int(self.match_field[0][0]), int(self.match_field[0][1])),
            #               (int(self.match_field[1][0]), int(self.match_field[1][1])),
            #               (0, 255, 0), 2)
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
            cv2.rectangle(frame, (int(self.goal1[0][0]), int(self.goal1[0][1])),
                          (int(self.goal1[1][0]), int(self.goal1[1][1])),
                          (0, 255, 0), 2)
            cv2.rectangle(frame, (int(self.goal2[0][0]), int(self.goal2[0][1])),
                          (int(self.goal2[1][0]), int(self.goal2[1][1])),
                          (0, 255, 0), 2)
            cv2.rectangle(frame, (int(self.throw_in_zone[0][0]), int(self.throw_in_zone[0][1])),
                          (int(self.throw_in_zone[1][0]), int(self.throw_in_zone[1][1])),
                          (0, 255, 0), 2)

    def _draw_ball(self, frame):
        """
        Draw a circle at the balls position and name the Object "ball"
        """
        # draw a circle for the ball
        if self._current_ball_position != [-1, -1]:
            cv2.circle(frame, (self._current_ball_position[0], self._current_ball_position[1]), 16, (0, 255, 0), 2)
            cv2.putText(frame, "Ball", (self._current_ball_position[0], self._current_ball_position[1]),
                        cv2.FONT_HERSHEY_PLAIN, 1, (30, 144, 255), 2)

    def _draw_predicted_ball(self, frame):
        """
        Draw a circle at the predicted balls position if there is no ball
        detected in the frame and name the Object "ball"
        """
        if self._current_ball_position == [-1, -1]:
            cv2.circle(frame, (self._predicted[0], self._predicted[1]), 16, (0, 255, 255), 2)

    def _draw_figures(self, frame, player_positions, team):
        """
        Draw a rectangle at the players position and name it TeamX
        """
        if self._players_on_field:
            for i, player_position in enumerate(player_positions):
                cv2.rectangle(frame, (player_position[0], player_position[1]),
                              (player_position[0] + player_position[2], player_position[1] + player_position[3]),
                              (0, 255, 0), 2)
                cv2.putText(frame, ("Team" + str(team) + ", " + str(self._ranked[team - 1][i])),
                            (player_position[0], player_position[1]), cv2.FONT_HERSHEY_PLAIN, 1, (30, 144, 255), 2)
            self._players_on_field = False

    def _show_game_score(self, frame):
        """
        Draw game score in the bottom right corner
        """
        cv2.putText(frame, (str(self._counter_team1) + " : " + str(self._counter_team2)), (1700, 850),
                    cv2.FONT_HERSHEY_PLAIN, 2, (30, 144, 255), 2)

    def _show_last_games(self, frame):
        """
        Draw results of the last three games in the top right corner
        """
        cv2.putText(frame, "Last Games", (1700, 200), cv2.FONT_HERSHEY_PLAIN, 1, (30, 144, 255), 2)
        cv2.putText(frame, (str(self._game_results[-1][0]) + " : " + str(self._game_results[-1][1])), (1700, 220),
                    cv2.FONT_HERSHEY_PLAIN, 1, (30, 144, 255), 2)
        if len(self._game_results) > 2:
            cv2.putText(frame, (str(self._game_results[-2][0]) + " : " + str(self._game_results[-2][1])), (1700, 240),
                        cv2.FONT_HERSHEY_PLAIN, 1, (30, 144, 255), 2)
        if len(self._game_results) > 3:
            cv2.putText(frame, (str(self._game_results[-3][0]) + " : " + str(self._game_results[-3][1])), (1700, 260),
                        cv2.FONT_HERSHEY_PLAIN, 1, (30, 144, 255), 2)

    def _show_ball_speed(self, frame):
        """
        Draw the fastest ball speed of the last ~4sek in the bottom right corner
        """
        cv2.putText(frame, (str(max(self._last_speed)) + " Km/h"), (1700, 900), cv2.FONT_HERSHEY_PLAIN, 1,
                    (30, 144, 255), 2)
