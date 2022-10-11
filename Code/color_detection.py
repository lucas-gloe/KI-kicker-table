import numpy as np
import cv2
from filterpy.kalman import KalmanFilter
import keyboard
import time


# defining global variables 

# defining the colors to track, in this Case Ball-red, Team1- orange and Team2- blue
colors = [[0, 69, 151, 9, 106, 201], [0, 114, 144, 57, 255, 255], [102, 66, 73,125, 255, 255]] #HSV
#colors = [[0, 87, 175, 184, 153, 243]] #HSV
#colors = [[0, 87, 175, 184, 153, 243], [0, 135, 222, 66, 255, 255], [38, 102, 106, 139, 255, 255]] #HSV
#colors = [[0, 0, 156, 182, 88, 248], [0, 135, 222, 66, 255, 255], [38, 102, 106, 139, 255, 255]] #weiß



classes = ["Ball", "Team1", "Team2"] 

# defining the webcams properties
camera = cv2.VideoCapture(1)
#camera = cv2. VideoCapture("../OpenCV/Video/1080p/60fps/20220923_200827.MOV")
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) #heigth
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) #width
camera.set(cv2.CAP_PROP_FPS, 240) #FPS
fps = camera.get(cv2.CAP_PROP_FPS)

def drawContourOnKicker(showContour, frame):
	
	if keyboard.is_pressed("c"):
		showContour = True
	#720p
	#cv2.rectangle(newframe,(180,55),(1110,570),(0,255,0),2) #field
	#cv2.rectangle(newframe,(160,230),(190,395),(0,255,0),2) #goal1
	#cv2.rectangle(newframe,(1100,230),(1130,395),(0,255,0),2) #goal2
	#cv2.rectangle(newframe,(430,55),(890,570),(0,255,0),2) #einwurf
	#1080p
	if showContour:
		cv2.rectangle(frame,(320,180),(1620,920),(0,255,0),2) #field
		#cv2.rectangle(frame,(250,430),(340,670),(0,255,0),2) #goal1
		#cv2.rectangle(frame,(1600,430),(1690,660),(0,255,0),2) #goal2
		#cv2.rectangle(frame,(700,180),(1270,920),(0,255,0),2) #einwurf
		if keyboard.is_pressed("f"):
			showContour = False
	return showContour

class KalmanFilter:
	"""
	Load the kalman filter for predicition purposess and use it to predict the next position of the ball
	Source: VisualComputer, https://www.youtube.com/watch?v=67jwpv49ymA
	"""
	kf = cv2.KalmanFilter(4,2)
	kf.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]], np.float32)
	kf.transitionMatrix = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]], np.float32)

	def predict(self, coordX, coordY):
		measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
		self.kf.correct(measured)
		predicted = self.kf.predict()
		x,y = int(predicted[0]), int(predicted[1])
		tuple = (x,y)
		return tuple

def createColorMasks(color, hsvimg):
	"""
	create a black and white mask vor every Color which is in [color] for tracking purposess
	Return: the mask
	"""
	lower= np.array(color[0:3])
	upper= np.array(color[3:6])
	mask = cv2.inRange(hsvimg, lower, upper)
	return mask

def drawBall(classes, centerX, centerY, x, y, trackedFrame):
	"""
	Draw a cicle at the balls position and name the Object "ball"
	"""
	# draw a circle for the ball
	cv2.circle(trackedFrame,(centerX, centerY), 16 ,(0,255,0),2)
	cv2.putText(trackedFrame,str(classes), (int(x), int(y)) ,cv2.FONT_HERSHEY_PLAIN , 1 , (30, 144, 255), 2)

def drawPlayer(classes, name, w, h, x, y, trackedFrame):
	"""
	Draw a rectangle at the players posioion and name it TeamX
	"""
	cv2.rectangle(trackedFrame,(x,y),(x+w,y+h),(0,255,0),2)
	cv2.putText(trackedFrame,(str(classes) + ", " + str(name)), (int(x), int(y)) ,cv2.FONT_HERSHEY_PLAIN , 1 , (30, 144, 255), 2)

def findObjects(mask):
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

def loadPlayersNames(objects):
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

def computeObjectTracking(frame, trackedFrame, colors, predicted, ballPositions, kf):
	"""
	mark every contour which should be tracked on the image based on their colour
	Return: marked image
	"""
	# converting rgb to hsv shema
	hsvimg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	# looping over every color to track to define the masks
	for color in colors:
		mask = createColorMasks(color, hsvimg)
		# finding every object on the masks
		#cv2.imshow(str(color), mask)
		objects = findObjects(mask)
		#filter players for their positions
		if color == colors[1] or color == colors[2]:
			# looping over every contour witch where found on the masks	
			name = loadPlayersNames(objects)
		for i, contour in enumerate(objects):
			x = contour[0]
			y = contour[1]
			w = contour[2]
			h = contour[3]
			# name the given contour 
			if color == colors[0]:
				#defing the centerpoints for the case the detected contour is the ball
				centerX = int((x+(w/2)))
				centerY = int((y+(h/2)))
				
					#draw the balls posotion in the image
				drawBall(classes[0],centerX, centerY, x, y, trackedFrame)
				
					# save the current position of the ball into an array
				currentBallPosition = [x,y]
				
					# save the current position of the ball together with his last view positions
				ballPositions.append(currentBallPosition)
				
					# predict the next centerpoints of the ball in case the ball is under a figure in the next frame
				predicted = kf.predict(centerX, centerY)

			if color == colors[1]:
				# draw a rectangle for the players of team 1
				drawPlayer(classes[1], name[i] ,w ,h , x, y, trackedFrame)

			if color == colors[2]:
				# draw a rectangle for the players of team 2
				drawPlayer(classes[2], name[i], w, h, x, y, trackedFrame)

		# if no circle was drawn on the videostream draw the estimated position of the ball from the last frame 
		if color == colors[0] and len(objects) == 0:
			# if no contour was found in the mask of the ball draw the estimated position of the ball from the last frame 
			ballPositions.append([-1,-1])
			cv2.circle(trackedFrame,(predicted[0], predicted[1]), 16 ,(0,255,0),2)

	return predicted, ballPositions

def resetGame(trackedFrame, counterTeam1, counterTeam2, gameResults, results):
	if keyboard.is_pressed("n") and results:
		cv2.putText(trackedFrame,"Neues Spiel", (800,500) ,cv2.FONT_HERSHEY_PLAIN , 5 , (30, 144, 255), 2)
		gameResults.append([counterTeam1, counterTeam2])
		counterTeam1 = 0
		counterTeam2 = 0
		results = False
	return counterTeam1, counterTeam2, gameResults, results


def ballSpeedTracking(ballPositions, trackedFrame, lastSpeed):
    """
    Measure the current speed of the ball on the frame
    """
    if len(ballPositions)>=3:
        #safe the current ballposition into an numpyArray
        currentPosition = np.array(ballPositions[-1])
        #safe the current-1 ballposition into an numpyArray
        middlePosition = np.array(ballPositions[-2])
        #safe the current-2 ballposition into an numpyArray
        lastPosition = np.array(ballPositions[-3])
        #measure the distance between the last and last-1 point of the ball
        distance1 = np.linalg.norm(currentPosition-middlePosition)
        #measure the distance between the last-1 and last-2 point of the ball
        distance2 = np.linalg.norm(middlePosition-lastPosition)
        #calculate the travelled distance between 1 frame
        distance = (distance1+distance2)/2
        #convert the travelled distance into real speed messauring
        # -> Cameraview on Kicker Table is 1300px x 740p at 120, Kicker Table is 1,20m x 0,68m relationship: ~1:1.083
        realDistancePerFrame = distance / 1083
        realDistancePerSecond = realDistancePerFrame * 120
        kmh = realDistancePerSecond * 3.6
        kmh = round(kmh,2)
        lastSpeed.append(kmh)
        cv2.putText(trackedFrame,(str(max(lastSpeed)) + " Km/h"), (1700, 900) ,cv2.FONT_HERSHEY_PLAIN , 1 , (30, 144, 255), 2)
        if len(ballPositions)>=1000:
            ballPositions.pop(0)
        if len(lastSpeed)>=100:
            lastSpeed.pop(0)
    return lastSpeed, ballPositions

def showResults(trackedFrame,counterTeam1, counterTeam2, gameResults):
	cv2.putText(trackedFrame,(str(counterTeam1) + " : " + str(counterTeam2)), (1700, 850) ,cv2.FONT_HERSHEY_PLAIN , 2 , (30, 144, 255), 2)
	cv2.putText(trackedFrame,("Last Games"), (1700, 200) ,cv2.FONT_HERSHEY_PLAIN , 1 , (30, 144, 255), 2)
	cv2.putText(trackedFrame,(str(gameResults[-1][0]) + " : " + str(gameResults[-1][1])), (1700, 220) ,cv2.FONT_HERSHEY_PLAIN , 1 , (30, 144, 255), 2)
	if len(gameResults)>2:
		cv2.putText(trackedFrame,(str(gameResults[-2][0]) + " : " + str(gameResults[-2][1])), (1700, 240) ,cv2.FONT_HERSHEY_PLAIN , 1 , (30, 144, 255), 2)
	if len(gameResults)>3:
		cv2.putText(trackedFrame,(str(gameResults[-3][0]) + " : " + str(gameResults[-3][1])), (1700, 260) ,cv2.FONT_HERSHEY_PLAIN , 1 , (30, 144, 255), 2)

def detectGoal(ballPositions, trackedFrame, counterTeam1, counterTeam2, goal1detected, goal2detected, ballOutOfGame, gameResults):
	goalInCurrentFrame = False
	if len(ballPositions) > 1 and 0 < ballPositions[-2][0] < 250 and 430 < ballPositions[-2][1] < 670 and ballPositions[-1] == [-1,-1] and ballOutOfGame:
		goal1detected = True
		goalInCurrentFrame = True

	if len(ballPositions) > 1 and ballPositions[-2][0] > 1600 and 430 < ballPositions[-2][1] < 660 and ballPositions[-1] == [-1,-1] and ballOutOfGame:
		goal2detected = True
		goalInCurrentFrame = True

	if goal1detected and goalInCurrentFrame:
		counterTeam1 += 1
		ballOutOfGame = False
	if goal2detected and goalInCurrentFrame:
		counterTeam2 += 1
		ballOutOfGame = False

		
	showResults(trackedFrame,counterTeam1, counterTeam2, gameResults)

	return counterTeam1, counterTeam2, goal1detected, goal2detected, gameResults, ballOutOfGame

def detectBallReentering(goal1detected, goal2detected, ballPositions, results, ballOutOfGame):
	if goal1detected or goal2detected:
		if 700 < ballPositions[-1][0] < 1270 and ballPositions[-2] == [-1,-1]:
			goal1detected = False
			goal2detected = False
			results = True
			ballOutOfGame = True
	return goal1detected, goal2detected, results, ballOutOfGame


def main():
	"""
	using the functions to output the tracked image
	"""
	# defining calibration
	showContour = False
	# defining Arrays for BallSpeedTracking
	ballPositions = []
	lastSpeed = []

	# defining Arrays for Ballpredicition
	predicted = (0,0)

	# defining variables for goalCounting
	counterTeam1 = 0
	counterTeam2 = 0
	ballOutOfGame = True
	goal1detected = False
	goal2detected = False
	results = True
	gameResults= [[0,0]]

	# defining classes names
	kf = KalmanFilter()

	# loop over every frame from the camera and perform the tracking
	while True:
		#read the cameras frame
		new_frame_time = time.time()
		ret, frame = camera.read()
		
		#print(fps)
		#print(frame.shape)
	

		trackedFrame = frame.copy()

		showContour = drawContourOnKicker(showContour, frame)

		counterTeam1, counterTeam2, gameResults, results = resetGame(trackedFrame, counterTeam1, counterTeam2, gameResults, results)

		#track the searched object on the frames
		predicted, ballPositions = computeObjectTracking(frame, trackedFrame, colors, predicted, ballPositions, kf)

		#measure the speed of the ball
		lastSpeed, ballPositions = ballSpeedTracking(ballPositions, trackedFrame, lastSpeed) 

		#look if a goal was shot
		counterTeam1, counterTeam2, goal1detected, goal2detected, gameResults, ballOutOfGame = detectGoal(ballPositions, trackedFrame, counterTeam1, counterTeam2 ,goal1detected ,goal2detected, ballOutOfGame, gameResults)

		# detect if ball reenters the field
		goal1detected, goal2detected, results, ballOutOfGame = detectBallReentering(goal1detected, goal2detected, ballPositions, results, ballOutOfGame)        

		#fps = 1/(new_frame_time-prev_frame_time)
		#prev_frame_time = new_frame_time
		#print(fps)


		# cv2.putText(trackedFrame,str(fps), (1200, 200) ,cv2.FONT_HERSHEY_PLAIN , 1 , (30, 144, 255), 2)

		#show the tracked frame
		cv2.imshow("tracked",trackedFrame)
		#cv2.imshow("original",originalFrame)


		#break if q is pressed
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

		
	#release and stop the camera and code
	camera.release()
	cv2.destroyAllWindows()

main()