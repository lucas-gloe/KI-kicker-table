import cv2
from datetime import datetime
from kalman_filter import KalmanFilter
import numpy as np
import keyboard


class Game:
    """
    Class that tracks the game and his properties
    """
################################# INITIALIZE GAME CLASS ####################################
    def __init__(self):
        self._start_time = None
        self._num_occurrences = 0
        self._current_ball_position = None
        self._ball_positions = []
        self._colors = [[0, 95, 134, 9, 144, 234], [0, 167, 165, 23, 255, 255], [102, 66, 111 ,120, 255, 255]] #HSV
        #self._colors = [[0, 69, 151, 9, 106, 201], [0, 114, 144, 57, 255, 255], [102, 66, 73,125, 255, 255]] #HSV
        self._kf = KalmanFilter()
        self._ranked = [[],[]]
        self._player1_figures = []
        self._player2_figures = []
        self._predicted = (0,0)
        self._counter_team1 = 0
        self._counter_team2 = 0
        self._ball_out_of_game = True
        self._goal1_detected = False
        self._goal2_detected = False
        self._results = True
        self._game_results= [[0,0]]
        self._new_game = False
        self._show_contour = False
        self._last_speed = [0.0]

    def start(self):
        self._start_time = datetime.now()
        return self

    def __counts_per_sec(self):
        elapsed_time = (datetime.now() - self._start_time).total_seconds()
        return self._num_occurrences / elapsed_time if elapsed_time > 0 else 0

################################ INTERPRETATION OF THE FRAME ###############################

    def interpret_frame(self, frame):
        # Frame interpretation
        self._num_occurrences += 1
        hsvimg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self._track_ball(hsvimg)
        self._player1_figures = self._track_players(1, 0, hsvimg)
        self._player2_figures = self._track_players(2, 1, hsvimg)
        
        self._check_keybindings()
        
        #track game stats
        self._count_gamescore() 
        self._detect_ball_reentering()
        self._reset_game()
        
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

        # check arrays
        self._check_length_of_arrays()

        return out_frame

#################################### FUNCTIONS FOR INTERPRETATION ###########################

####################################  TRACKING ##############################################

    def _track_ball(self, hsvimg):
        mask = cv2.inRange(hsvimg, np.array(self._colors[0][0:3]), np.array(self._colors[0][3:6]))
        objects = self.__find_objects(mask)
        if(len(objects) == 1):
            x = objects[0][0]
            y = objects[0][1]
            w = objects[0][2]
            h = objects[0][3]

            #defing the centerpoints for the case the detected contour is the ball
            centerX = int((x+(w/2)))
            centerY = int((y+(h/2)))
        
            # save the current position of the ball into an array
            self._current_ball_position = [centerX,centerY]

            self._predicted = self._kf.predict(centerX, centerY)

        elif(len(objects) == 0):
            print("Ball nicht erkannt")
            self._current_ball_position = [-1,-1]
        else:
            if len(self._ball_positions)>2:
                if self._ball_positions[-1] == [-1,-1]:
                    self._current_ball_position = [-1,-1]

                # if self.ball_positions[-1] != [-1,-1]:
                #     positionMatrix = np.array(objects)
                #     ball_positions = np.array(self.ball_positions)

                #     potenzielleBälle = positionMatrix[np.abs(positionMatrix[:,2]-positionMatrix[:,3]) <= 5]
                #     näheZumLetztenBall = np.abs(potenzielleBälle[:,0]-ball_positions[-1,0])
                #     wahrscheinlichePosition = potenzielleBälle[np.where(np.min(näheZumLetztenBall))]

                #     centerX = int((wahrscheinlichePosition[:,0]+(wahrscheinlichePosition[:,2]/2)))
                #     centerY = int((wahrscheinlichePosition[:,1]+(wahrscheinlichePosition[:,3]/2)))

                #     self.current_ball_position = [centerX,centerY]
            else:
                self._current_ball_position = [-1,-1]


        self._ball_positions.append(self._current_ball_position)

    def _track_players(self, team_number, teamrank, hsvimg):
        player_positions = []
        mask = cv2.inRange(hsvimg, np.array(self._colors[team_number][0:3]), np.array(self._colors[team_number][3:6]))
        objects = self.__find_objects(mask)
        if(len(objects) >= 1):
            self._ranked[teamrank] = self.loadPlayersNames(objects)
            # if teamrank == 1:
            #    self.ranked[teamrank] = self.ranked[teamrank][::-1]
            player_positions = objects
            return player_positions
            
        elif(len(objects) == 0):
            print("Spieler nicht erkannt")   

    def __find_objects(self, mask):
        """
        tracking algorithm to find the contours on the masks
        return: Contours on the masks
        source: https://www.computervision.zone/courses/learn-opencv-in-3-hours/
        """
        # outline the contours on the mask
        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        x , y , w , h = 0 ,0 ,0 ,0
        whiteContour = []
        objects = []
            #looping over every contour which was found
        for cnt in contours:
            area = cv2.contourArea(cnt)
                # saving countours properties in variables if a certain area is detected on the mask (to prevent blurring)
            if area>100:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                whiteContour = x,y,w,h
                objects.append(whiteContour)
        return objects

    def loadPlayersNames(self, objects):
        """
        take the position of alle players of a team and rank them sortet from the position of the field
        Return: Array with sorted list of players ranks
        """
        if len(objects)>0:
            positionMatrix = np.array(objects)
            valuetMatrix = positionMatrix[:,0]*10 + positionMatrix[:,1]
            sortedValuetMatrix = valuetMatrix.argsort()
            ranks = np.empty_like(sortedValuetMatrix)
            ranks[sortedValuetMatrix] = np.arange(len(valuetMatrix))
            return ranks

##################################### GAME STATS ##############################################################

    def _check_keybindings(self):
        if keyboard.is_pressed("c"): # start calibration of kicker frame
            self._show_contour = True
        if keyboard.is_pressed("f"): # end calibration of kicker frame
            self._show_contour = False
        if keyboard.is_pressed("n"): # start new game
            self._new_game = True

    def _count_gamescore(self):
        if len(self._ball_positions) > 1 and 0 < self._ball_positions[-2][0] < 250 and 430 < self._ball_positions[-2][1] < 670 and self._ball_positions[-1] == [-1,-1] and self._ball_out_of_game:
            self._goal1_detected = True
            self.goalInCurrentFrame = True

        if len(self._ball_positions) > 1 and self._ball_positions[-2][0] > 1600 and 430 < self._ball_positions[-2][1] < 660 and self._ball_positions[-1] == [-1,-1] and self._ball_out_of_game:
            self._goal2_detected = True
            self.goalInCurrentFrame = True

        if self._goal1_detected and self.goalInCurrentFrame:
            self._counter_team1 += 1
            self._ball_out_of_game = False
        if self._goal2_detected and self.goalInCurrentFrame:
            self._counter_team2 += 1
            self._ball_out_of_game = False

    def _detect_ball_reentering(self):
        if self._goal1_detected or self._goal2_detected:
            if 700 < self._ball_positions[-1][0] < 1270 and self._ball_positions[-2] == [-1,-1]:
                self._goal1_detected = False
                self._goal2_detected = False
                self._results = True
                self._ball_out_of_game = True
    
    def _reset_game(self):
        if self._new_game and self._results:
            self._game_results.append([self._counter_team1, self._counter_team2])
            self._counter_team1 = 0
            self._counter_team2 = 0
            self._results = False

    def ballSpeedTracking(self):
        """
        Measure the current speed of the ball on the frame
        """
        if len(self._ball_positions)>=3 and self._ball_positions[-1] != [-1,-1]:
            #safe the current ballposition into an numpyArray
            currentPosition = np.array(self._ball_positions[-1])
            #safe the current-1 ballposition into an numpyArray
            middlePosition = np.array(self._ball_positions[-2])
            #safe the current-2 ballposition into an numpyArray
            lastPosition = np.array(self._ball_positions[-3])
            #measure the distance between the last and last-1 point of the ball
            distance1 = np.linalg.norm(currentPosition-middlePosition)
            #measure the distance between the last-1 and last-2 point of the ball
            distance2 = np.linalg.norm(middlePosition-lastPosition)
            #calculate the travelled distance between 1 frame
            distance = (distance1+distance2)/2
            #convert the travelled distance into real speed messauring
            # -> Cameraview on Kicker Table is 1300px x 740p at 120, Kicker Table is 1,20m x 0,68m relationship: ~1:1.083
            realDistancePerFrame = distance / 1083
            realDistancePerSecond = realDistancePerFrame * 60
            kmh = realDistancePerSecond * 3.6
            kmh = round(kmh,2)
            self._last_speed.append(kmh)
            

    def _check_length_of_arrays(self):
        if len(self._ball_positions)>=1000:
            self._ball_positions.pop(0)
        if len(self._last_speed)>=100:
            self._last_speed.pop(0)


#####################################  PRINTING ON FRAME ######################################################

    def _put_iterations_per_sec(self, tracked_frame, iterations_per_sec):
        """
        Add iterations per second text to lower-left corner of a frame.
        """
        cv2.putText(tracked_frame, "{:.0f} iterations/sec".format(iterations_per_sec),(50, 900), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))

    def _draw_contour_on_kicker(self, frame):
        """
        Add football field contour for calibration on frame
        """
        if self._show_contour:
            cv2.rectangle(frame,(320,180),(1620,920),(0,255,0),2) #field
            cv2.rectangle(frame,(250,430),(340,670),(0,255,0),2) #goal1
            cv2.rectangle(frame,(1600,430),(1690,660),(0,255,0),2) #goal2
            cv2.rectangle(frame,(700,180),(1270,920),(0,255,0),2) #einwurf

    def _draw_ball(self, frame):
        """
        Draw a cicle at the balls position and name the Object "ball"
        """
        # draw a circle for the ball
        if(self._current_ball_position != [-1,-1]):
            cv2.circle(frame,(self._current_ball_position[0], self._current_ball_position[1]), 16 ,(0,255,0),2)
            cv2.putText(frame,"Ball", (self._current_ball_position[0], self._current_ball_position[1]) ,cv2.FONT_HERSHEY_PLAIN , 1 , (30, 144, 255), 2)

    def _draw_predicted_ball(self, frame):
        if(self._current_ball_position == [-1,-1]):
            cv2.circle(frame,(self._predicted[0], self._predicted[1]), 16 ,(0,255,255),2)

    def _draw_figures(self, frame, player_positions, team):
        """
        Draw a rectangle at the players position and name it TeamX
        """
        if player_positions:
            for i, player_position in enumerate(player_positions):
                cv2.rectangle(frame,(player_position[0],player_position[1]),(player_position[0]+player_position[2],player_position[1]+player_position[3]),(0,255,0),2)
                cv2.putText(frame,("Team" + str(team) + ", " + str(self._ranked[team-1][i])), (player_position[0],player_position[1]) ,cv2.FONT_HERSHEY_PLAIN , 1 , (30, 144, 255), 2)
    
    def _show_game_score(self, frame):
        cv2.putText(frame,(str(self._counter_team1) + " : " + str(self._counter_team2)), (1700, 850) ,cv2.FONT_HERSHEY_PLAIN , 2 , (30, 144, 255), 2)

    def _show_last_games(self, frame):
        cv2.putText(frame,("Last Games"), (1700, 200) ,cv2.FONT_HERSHEY_PLAIN , 1 , (30, 144, 255), 2)
        cv2.putText(frame,(str(self._game_results[-1][0]) + " : " + str(self._game_results[-1][1])), (1700, 220) ,cv2.FONT_HERSHEY_PLAIN , 1 , (30, 144, 255), 2)
        if len(self._game_results)>2:
            cv2.putText(frame,(str(self._game_results[-2][0]) + " : " + str(self._game_results[-2][1])), (1700, 240) ,cv2.FONT_HERSHEY_PLAIN , 1 , (30, 144, 255), 2)
        if len(self._game_results)>3:
            cv2.putText(frame,(str(self._game_results[-3][0]) + " : " + str(self._game_results[-3][1])), (1700, 260) ,cv2.FONT_HERSHEY_PLAIN , 1 , (30, 144, 255), 2)

    def _show_ball_speed(self, frame):
        cv2.putText(frame,(str(max(self._last_speed)) + " Km/h"), (1700, 900) ,cv2.FONT_HERSHEY_PLAIN , 1 , (30, 144, 255), 2)