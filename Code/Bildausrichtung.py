from cv2 import undistort
import numpy as np
import cv2
import glob

# kamera einlesen
camera = cv2.VideoCapture(1)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) #
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) #
camera.set(cv2.CAP_PROP_FPS, 120) #FPS
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

def drawContourOnKicker(newframe):
	#720p
	#cv2.rectangle(newframe,(180,55),(1110,570),(0,255,0),2) #field
	#cv2.rectangle(newframe,(160,230),(190,395),(0,255,0),2) #goal1
	#cv2.rectangle(newframe,(1100,230),(1130,395),(0,255,0),2) #goal2
	#cv2.rectangle(newframe,(430,55),(890,570),(0,255,0),2) #einwurf
	#1080p
	cv2.rectangle(newframe,(320,180),(1620,920),(0,255,0),2) #field
	cv2.rectangle(newframe,(250,430),(340,670),(0,255,0),2) #goal1
	cv2.rectangle(newframe,(1600,430),(1690,660),(0,255,0),2) #goal2
	cv2.rectangle(newframe,(700,180),(1270,920),(0,255,0),2) #einwurf


def main():
	#mtx, dist, roi, newcameramtx = loads_camera_matrix_and_distortion_coefficients()

	while True:
		ret, frame = camera.read()
		#frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

		# crop the image
		#x, y, w, h = roi
		#newframe = frame[y:y+h, x:x+w]

		#print(fps)
		#print(frame.shape)
		
		drawContourOnKicker(frame)

		cv2.imshow("Frame", frame)

		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

	camera.release()
	cv2.destroyAllWindows()

main()