import cv2
from datetime import datetime
from KalmanFilter import KalmanFilter
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
        self.current_ball_position = None
        self.ball_positions = []
        self.colors = [[0, 95, 134, 9, 144, 234], [0, 167, 165, 23, 255, 255], [102, 66, 111 ,120, 255, 255]] #HSV
        #self.colors = [[0, 69, 151, 9, 106, 201], [0, 114, 144, 57, 255, 255], [102, 66, 73,125, 255, 255]] #HSV
        self.kf = KalmanFilter()
        self.ranked = [[],[]]
        self.player1_figures = []
        self.player2_figures = []
        self.predicted = (0,0)
        self.counterTeam1 = 0
        self.counterTeam2 = 0
        self.ballOutOfGame = True
        self.goal1detected = False
        self.goal2detected = False
        self.results = True
        self.gameResults= [[0,0]]
        self.showContour = False

    def start(self):
        self._start_time = datetime.now()
        return self

    def countsPerSec(self):
        elapsed_time = (datetime.now() - self._start_time).total_seconds()
        return self._num_occurrences / elapsed_time if elapsed_time > 0 else 0

################################ INTERPRETATION OF THE FRAME ###############################

    def interpretFrame(self, frame):
        # Frame interpretation
        self._num_occurrences += 1
        hsvimg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.trackBall(hsvimg)
        self.player1_figures = self.trackPlayers(1, 0, hsvimg)
        self.player2_figures = self.trackPlayers(2, 1, hsvimg)
        
        #track game stats
        self.countGamescore() 
        self.detectBallReentering()
        self.resetGame()
        

        # Draw results in frame
        self.putIterationsPerSec(frame, self.countsPerSec())
        self.drawContourOnKicker(frame)
        self.drawBall(frame)
        self.drawPredictedBall(frame)
        self.drawFigures(frame, self.player1_figures, team=1)
        self.drawFigures(frame, self.player2_figures, team=2)
        self.showGameScore(frame)
        self.showLastGames(frame)

        return frame

#################################### FUNCTIONS FOR INTERPRETATION ###########################

####################################  TRACKING ##############################################

    def trackBall(self, hsvimg):
        mask = cv2.inRange(hsvimg, np.array(self.colors[0][0:3]), np.array(self.colors[0][3:6]))
        objects = self.findObjects(mask)
        if(len(objects) == 1):
            x = objects[0][0]
            y = objects[0][1]
            w = objects[0][2]
            h = objects[0][3]

            #defing the centerpoints for the case the detected contour is the ball
            centerX = int((x+(w/2)))
            centerY = int((y+(h/2)))
        
            # save the current position of the ball into an array
            self.current_ball_position = [centerX,centerY]

            self.predicted = self.kf.predict(centerX, centerY)

        elif(len(objects) == 0):
            print("Ball nicht erkannt")
            self.current_ball_position = [-1,-1]
        else:
            print("Mehr als ein Ball erkannt")
            self.current_ball_position = [-1,-1]

        self.ball_positions.append(self.current_ball_position)

    def trackPlayers(self, team_number, teamrank, hsvimg):
        player_positions = []
        mask = cv2.inRange(hsvimg, np.array(self.colors[team_number][0:3]), np.array(self.colors[team_number][3:6]))
        objects = self.findObjects(mask)
        if(len(objects) >= 1):
            self.ranked[teamrank] = self.loadPlayersNames(objects)
            # if teamrank == 1:
            #    self.ranked[teamrank] = self.ranked[teamrank][::-1]
            for contour in objects:
                x = contour[0]
                y = contour[1]
                w = contour[2]
                h = contour[3]

                current_player_position = [x,y,w,h]
                player_positions.append(current_player_position)
            return player_positions
            
        elif(len(objects) == 0):
            print("Spieler nicht erkannt")   

    def findObjects(self, mask):
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

    def countGamescore(self):
        if len(self.ball_positions) > 1 and 0 < self.ball_positions[-2][0] < 250 and 430 < self.ball_positions[-2][1] < 670 and self.ball_positions[-1] == [-1,-1] and self.ballOutOfGame:
            self.goal1detected = True
            self.goalInCurrentFrame = True

        if len(self.ball_positions) > 1 and self.ball_positions[-2][0] > 1600 and 430 < self.ball_positions[-2][1] < 660 and self.ball_positions[-1] == [-1,-1] and self.ballOutOfGame:
            self.goal2detected = True
            self.goalInCurrentFrame = True

        if self.goal1detected and self.goalInCurrentFrame:
            self.counterTeam1 += 1
            self.ballOutOfGame = False
        if self.goal2detected and self.goalInCurrentFrame:
            self.counterTeam2 += 1
            self.ballOutOfGame = False

    def detectBallReentering(self):
        if self.goal1detected or self.goal2detected:
            if 700 < self.ball_positions[-1][0] < 1270 and self.ball_positions[-2] == [-1,-1]:
                self.goal1detected = False
                self.goal2detected = False
                self.results = True
                self.ballOutOfGame = True
    
    def resetGame(self):
        if keyboard.is_pressed("n") and self.results:
            self.gameResults.append([self.counterTeam1, self.counterTeam2])
            self.counterTeam1 = 0
            self.counterTeam2 = 0
            self.results = False

#####################################  PRINTING ON FRAME ######################################################
    def putIterationsPerSec(self, tracked_frame, iterations_per_sec):
        """
        Add iterations per second text to lower-left corner of a frame.
        """
        cv2.putText(tracked_frame, "{:.0f} iterations/sec".format(iterations_per_sec),(50, 600), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))

    def drawContourOnKicker(self, frame):
            """
            Add football field contour for calibration on frame
            """
            if keyboard.is_pressed("c"):
                self.showContour = True
            if self.showContour:
                cv2.rectangle(frame,(320,180),(1620,920),(0,255,0),2) #field
                #cv2.rectangle(frame,(250,430),(340,670),(0,255,0),2) #goal1
                #cv2.rectangle(frame,(1600,430),(1690,660),(0,255,0),2) #goal2
                #cv2.rectangle(frame,(700,180),(1270,920),(0,255,0),2) #einwurf
                if keyboard.is_pressed("f"):
                    self.showContour = False

    def drawBall(self, frame):
        """
        Draw a cicle at the balls position and name the Object "ball"
        """
        # draw a circle for the ball
        if(self.current_ball_position != [-1,-1]):
            cv2.circle(frame,(self.current_ball_position[0], self.current_ball_position[1]), 16 ,(0,255,0),2)
            cv2.putText(frame,"Ball", (self.current_ball_position[0], self.current_ball_position[1]) ,cv2.FONT_HERSHEY_PLAIN , 1 , (30, 144, 255), 2)

    def drawPredictedBall(self, frame):
        if(self.current_ball_position == [-1,-1]):
            cv2.circle(frame,(self.predicted[0], self.predicted[1]), 16 ,(0,255,255),2)

    def drawFigures(self, frame, player_positions, team):
        """
        Draw a rectangle at the players position and name it TeamX
        """
        if player_positions:
            for i, player_position in enumerate(player_positions):
                cv2.rectangle(frame,(player_position[0],player_position[1]),(player_position[0]+player_position[2],player_position[1]+player_position[3]),(0,255,0),2)
                cv2.putText(frame,("Team" + str(team) + ", " + str(self.ranked[team-1][i])), (player_position[0],player_position[1]) ,cv2.FONT_HERSHEY_PLAIN , 1 , (30, 144, 255), 2)
    
    def showGameScore(self, frame):
        cv2.putText(frame,(str(self.counterTeam1) + " : " + str(self.counterTeam2)), (1700, 800) ,cv2.FONT_HERSHEY_PLAIN , 2 , (30, 144, 255), 2)

    def showLastGames(self, frame):
        cv2.putText(frame,("Last Games"), (1700, 200) ,cv2.FONT_HERSHEY_PLAIN , 1 , (30, 144, 255), 2)
        cv2.putText(frame,(str(self.gameResults[-1][0]) + " : " + str(self.gameResults[-1][1])), (1700, 220) ,cv2.FONT_HERSHEY_PLAIN , 1 , (30, 144, 255), 2)
        if len(self.gameResults)>2:
            cv2.putText(frame,(str(self.gameResults[-2][0]) + " : " + str(self.gameResults[-2][1])), (1700, 240) ,cv2.FONT_HERSHEY_PLAIN , 1 , (30, 144, 255), 2)
        if len(self.gameResults)>3:
            cv2.putText(frame,(str(self.gameResults[-3][0]) + " : " + str(self.gameResults[-3][1])), (1700, 260) ,cv2.FONT_HERSHEY_PLAIN , 1 , (30, 144, 255), 2) 