import numpy as np
import cv2
import glob
from filterpy.kalman import KalmanFilter
import keyboard

# defining global variables 

# defining the colors to track, in this Case Ball-red, Team1- orange and Team2- blue
colors = [[121, 38, 226, 213, 134, 255], [0, 135, 222, 66, 255, 255], [38, 102, 106, 139, 255, 255]] #HSV
classes = ["Ball", "Team1", "Team2"] 

# defining the webcams properties
camera = cv2.VideoCapture("../OpenCV/Video/720p/30fps/schuss_schnell.mp4")
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) #width
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) #height
#camera.set(cv2.CAP_PROP_FPS, 60) #FPS
fps = camera.get(cv2.CAP_PROP_FPS)

def loads_camera_matrix_and_distortion_coefficients():
	"""
	defining camera matrix and distortion coefficients.
	Source: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
	"""
	# termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 28, 0.001)
	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((8*6,3), np.float32)
	objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d point in real world space
	imgpoints = [] # 2d points in image plane.
	images = glob.glob('../OpenCV/Bilder/calibration/*.jpg')
	for fname in images:
		img = cv2.imread(fname)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   		# Find the chess board corners
		ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
   		# If found, add object points, image points (after refining them)
		if ret == True:
			objpoints.append(objp)
			corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
			imgpoints.append(corners)
   	    	# Draw and display the corners
			cv2.drawChessboardCorners(img, (8,6), corners2, ret)
		
			ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
		
			img = cv2.imread('../OpenCV/Bilder/ohne_gitter.jpg')
			h,  w = img.shape[:2]
			newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
		
			return [mtx, dist, roi, newcameramtx]

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
		# filter players for their positions
		if color == colors[1] or color == colors[2]:
		# looping over every contour witch where found on the masks	
			name = loadPlayersNames(objects)
			#print(name)	
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
		cv2.putText(trackedFrame,"Neues Spiel", (450,300) ,cv2.FONT_HERSHEY_PLAIN , 5 , (30, 144, 255), 2)
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
        # -> Cameraview on Kicker Table is 425p x 740p at 30fps, Kicker Table is 0,68m x 1,20m relationship: ~1:625
        realDistancePerFrame = distance / 625
        realDistancePerSecond = realDistancePerFrame * 30
        kmh = realDistancePerSecond * 3.6
        kmh = round(kmh,2)
        lastSpeed.append(kmh)
        cv2.putText(trackedFrame,(str(max(lastSpeed)) + " Km/h"), (1150, 550) ,cv2.FONT_HERSHEY_PLAIN , 1 , (30, 144, 255), 2)
        if len(ballPositions)>=1000:
            ballPositions.pop(0)
        if len(lastSpeed)>=100:
            lastSpeed.pop(0)
    return lastSpeed, ballPositions

def showResults(trackedFrame,counterTeam1, counterTeam2, gameResults):
	cv2.putText(trackedFrame,(str(counterTeam1) + " : " + str(counterTeam2)), (1150, 500) ,cv2.FONT_HERSHEY_PLAIN , 2 , (30, 144, 255), 2)
	cv2.putText(trackedFrame,("Last Games"), (1150, 70) ,cv2.FONT_HERSHEY_PLAIN , 1 , (30, 144, 255), 2)
	cv2.putText(trackedFrame,(str(gameResults[-1][0]) + " : " + str(gameResults[-1][1])), (1150, 90) ,cv2.FONT_HERSHEY_PLAIN , 1 , (30, 144, 255), 2)
	if len(gameResults)>2:
		cv2.putText(trackedFrame,(str(gameResults[-2][0]) + " : " + str(gameResults[-2][1])), (1150, 110) ,cv2.FONT_HERSHEY_PLAIN , 1 , (30, 144, 255), 2)
	if len(gameResults)>3:
		cv2.putText(trackedFrame,(str(gameResults[-3][0]) + " : " + str(gameResults[-3][1])), (1150, 130) ,cv2.FONT_HERSHEY_PLAIN , 1 , (30, 144, 255), 2)

def detectGoal(ballPositions, trackedFrame, counterTeam1, counterTeam2, goal1detected, goal2detected, ballOutOfGame, gameResults):
	goalInCurrentFrame = False
	if len(ballPositions) > 1 and 0 < ballPositions[-2][0] < 180 and 230 < ballPositions[-2][1] < 395 and ballPositions[-1] == [-1,-1] and ballOutOfGame:
		goal1detected = True
		goalInCurrentFrame = True

	if len(ballPositions) > 1 and ballPositions[-2][0] > 1110 and 230 < ballPositions[-2][1] < 395 and ballPositions[-1] == [-1,-1] and ballOutOfGame:
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
		if 430 < ballPositions[-1][0] < 890 and ballPositions[-2] == [-1,-1]:
			goal1detected = False
			goal2detected = False
			results = True
			ballOutOfGame = True
	return goal1detected, goal2detected, results, ballOutOfGame


def main():
	"""
	using the functions to output the tracked image
	"""
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

	#load the needed coeficients for the camera
	#mtx, dist, roi, newcameramtx = loads_camera_matrix_and_distortion_coefficients()

	cv2.namedWindow("cropped image", cv2.WINDOW_AUTOSIZE)

	cv2.resizeWindow("cropped image", 1024, 768)

	# loop over every frame from the camera and perform the tracking
	while True:
		#read the cameras frame
		ret, frame = camera.read()

		# undistort
		#frame = cv2.undistort(originalFrame, mtx, dist, None, newcameramtx)
		
		# crop the image
		##x, y, w, h = roi
		#frame = frame[y:y+h, x:x+w]
		#cv2.imwrite('test.png', frame)
		

		#perform the undistortion on the cameras frame
		#frame = cv2.undistort(originalFrame, mtx, dist, None, None)
		trackedFrame = frame.copy()

		counterTeam1, counterTeam2, gameResults, results = resetGame(trackedFrame, counterTeam1, counterTeam2, gameResults, results)

		#track the searched object on the frames
		predicted, ballPositions = computeObjectTracking(frame, trackedFrame, colors, predicted, ballPositions, kf)

		#measure the speed of the ball
		lastSpeed, ballPositions = ballSpeedTracking(ballPositions, trackedFrame, lastSpeed) 

		#look if a goal was shot
		counterTeam1, counterTeam2, goal1detected, goal2detected, gameResults, ballOutOfGame = detectGoal(ballPositions, trackedFrame, counterTeam1, counterTeam2 ,goal1detected ,goal2detected, ballOutOfGame, gameResults)

		# detect if ball reenters the field
		goal1detected, goal2detected, results, ballOutOfGame = detectBallReentering(goal1detected, goal2detected, ballPositions, results, ballOutOfGame)        

		#show the tracked frame
		cv2.imshow("cropped image",trackedFrame)
		#cv2.imshow("original",originalFrame)
		cv2.waitKey(0) & 0xFF == ord("q")

	#release and stop the camera and code
	camera.release()
	cv2.destroyAllWindows()

main()