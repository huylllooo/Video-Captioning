import cv2
import math
import os

imagesFolder = 'Frames/'
items = os.listdir("./Caption_Video/YouTubeClips")
for videoFile in items:
	if videoFile.endswith('.avi'):
		print(videoFile)
		cap = cv2.VideoCapture(videoFile)
		frameRate = cap.get(5) #frame rate
		while(cap.isOpened()):
			frameId = cap.get(1) #current frame number
			ret, frame = cap.read()
			if (ret != True):
				break
			if (int(frameId) % 10 == 1):
				filename = imagesFolder + str(videoFile) + '_' + str(int(frameId)) + '.jpg'
				cv2.imwrite(filename, frame)
		cap.release()
