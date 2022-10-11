import cv2
from datetime import datetime
from KalmanFilter import KalmanFilter
import numpy as np


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
        self.colors = [[0, 69, 151, 9, 106, 201], [0, 114, 144, 57, 255, 255], [102, 66, 73,125, 255, 255]] #HSV
        self.kf = KalmanFilter()
        self.player1_figures = []
        self.player2_figures = []

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
        self.player1_figures = self.trackPlayers(1, hsvimg)
        self.player2_figures = self.trackPlayers(2, hsvimg)
        

        # Draw results in frame
        self.putIterationsPerSec(frame, self.countsPerSec())
        self.drawContourOnKicker(frame)
        self.drawBall(frame)
        self.drawFigures(frame, self.player1_figures, team=1)
        self.drawFigures(frame, self.player2_figures, team=2)

        return frame

#################################### FUNCTIONS FOR INTERPRETATION ###########################

    def putIterationsPerSec(self, tracked_frame, iterations_per_sec):
        """
        Add iterations per second text to lower-left corner of a frame.
        """

        cv2.putText(tracked_frame, "{:.0f} iterations/sec".format(iterations_per_sec),(50, 600), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))

            
    def drawContourOnKicker(self, frame):
        """
        Add football field contour for calibration on frame
        """
        cv2.rectangle(frame,(180,55),(1110,570),(0,255,0),2) #field


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

        elif(len(objects) == 0):
            print("Ball nicht erkannt")
            self.current_ball_position = [-1,-1]
        else:
            print("Mehr als ein Ball erkannt")
            self.current_ball_position = [-1,-1]

        self.ball_positions.append(self.current_ball_position)

    def trackPlayers(self, team_number, hsvimg):
        player_positions = []
        mask = cv2.inRange(hsvimg, np.array(self.colors[team_number][0:3]), np.array(self.colors[team_number][3:6]))
        objects = self.findObjects(mask)
        if(len(objects) >= 1):
            for contour in objects:
                x = contour[0]
                y = contour[1]
                w = contour[2]
                h = contour[3]

                current_player_position = [x,y,w,h]
                player_positions.append(current_player_position)

                ranked = self.loadPlayersNames(player_positions)
                return ranked
            
        elif(len(objects) == 0):
            print("Spieler nicht erkannt")   


    # def computeObjectTracking(self, frame):
    #     """
    #     mark every contour which should be tracked on the image based on their colour
    #     Return: marked image
    #     """
    #     # converting rgb to hsv shema
    #     hsvimg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #     # looping over every color to track to define the masks
    #     for color in self.colors:
    #         mask = cv2.inRange(hsvimg, np.array(color[0:3]), np.array(color[3:6]))

    #         # finding every object on the masks
    #         #cv2.imshow(str(color), mask)
    #         objects = self.findObjects(mask)
    #         #filter players for their positions

    #         if color == self.colors[1] or color == self.colors[2]:
    #             # looping over every contour witch where found on the masks	
    #             players_names = self.loadPlayersNames(objects)
    #         for i, contour in enumerate(objects):
    #             x = contour[0]
    #             y = contour[1]
    #             w = contour[2]
    #             h = contour[3]
    #             # name the given contour 
    #             if color == self.colors[0]:
    #                 #defing the centerpoints for the case the detected contour is the ball
    #                 centerX = int((x+(w/2)))
    #                 centerY = int((y+(h/2)))
                
    #                 # save the current position of the ball into an array
    #                 self.current_ball_position = [centerX,centerY]
                    
    #                     # save the current position of the ball together with his last view positions
    #                 ballPositions.append(currentBallPosition)
                    
    #                     # predict the next centerpoints of the ball in case the ball is under a figure in the next frame
    #                 predicted = kf.predict(centerX, centerY)

    #             if color == self.colors[1]:
    #                 # draw a rectangle for the players of team 1
    #                 ObjectTracking.drawPlayer("Team1", players_names[i] ,w ,h , x, y, trackedFrame)

    #             if color == self.colors[2]:
    #                 # draw a rectangle for the players of team 2
    #                 ObjectTracking.drawPlayer("Team2", players_names[i], w, h, x, y, trackedFrame)

    #         # if no circle was drawn on the videostream draw the estimated position of the ball from the last frame 
    #         if color == colors[0] and len(objects) == 0:
    #             # if no contour was found in the mask of the ball draw the estimated position of the ball from the last frame 
    #             ballPositions.append([-1,-1])
    #             cv2.circle(trackedFrame,(predicted[0], predicted[1]), 16 ,(0,255,0),2)

    #     return predicted, ballPositions

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

    def drawBall(self, frame):
        """
        Draw a cicle at the balls position and name the Object "ball"
        """
        # draw a circle for the ball
        if(self.current_ball_position != [-1,-1]):
            cv2.circle(frame,(self.current_ball_position[0], self.current_ball_position[1]), 16 ,(0,255,0),2)
            cv2.putText(frame,"Ball", (self.current_ball_position[0], self.current_ball_position[1]) ,cv2.FONT_HERSHEY_PLAIN , 1 , (30, 144, 255), 2)

    def drawFigures(self, frame, player_positions, team):
        """
        Draw a rectangle at the players posioion and name it TeamX
        """
        if player_positions:
            for i, player_position in enumerate(player_positions):
                cv2.rectangle(frame,(player_position[0],player_position[1]),(player_position[0]+player_position[2],player_position[1]+player_position[3]),(0,255,0),2)
                cv2.putText(frame,("Team" + team + ", " + i), (player_position[0],player_position[1]) ,cv2.FONT_HERSHEY_PLAIN , 1 , (30, 144, 255), 2)