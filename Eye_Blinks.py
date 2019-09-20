
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import time
import dlib
import cv2 
eyeLash = 0.3 
continousFrames = 3
counter = 0
total = 0
data="shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(data)
(l1, l2) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(r1, r2) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
#function for finding eye points ratio
def formula(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	eyepoints = (A + B) / (2.0 * C)
	return eyepoints
#For Video Capturing
cap = cv2.VideoCapture(0)
while True:
	
	video = cap.read()[1] 
	gray = cv2.cvtColor(video,cv2.COLOR_BGR2GRAY) 
	det = detector(gray) 
	for i in det:
		
		shape = predictor(gray, i)
		shape = face_utils.shape_to_np(shape)
		left = shape[l1:l2]
		right = shape[r1:r2]
		lear = formula(left)
		rear = formula(right)
		frameCounter = (lear + rear) / 2.0
		# if frame counter value is less than the certain eyeLash value the blink is counted
		if frameCounter<eyeLash:
			counter += 1

		else:
			#if eyes are continously closed , the automatically eye blinks are incremented
			if counter >= continousFrames :
				total += 1
			counter = 0
		cv2.putText(video, "Eye_Blinks: {}".format(total), (480, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
		
	
	cv2.imshow("OUTPUT", video)
	k=cv2.waitKey(1)
	if k==27:
		break


 		

cv2.destroyAllWindows()




