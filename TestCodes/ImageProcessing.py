#!/usr/bin/python
import cv2
import numpy as np


def PrintSize(frame):
	height = np.size(frame, 0)
	width = np.size(frame, 1)
	print(width,height)

def ConnectCamera():
	cap = cv2.VideoCapture(-1)
	while(True):
		ret, frame = cap.read()
		cv2.imshow('original',frame)
		frame = frame[220:720,100 :1100]
		# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
		gray = ImageProcessing(frame)
		cv2.imshow('frame',gray)
		if cv2.waitKey(40) & 0xFF == ord('q'):
			break

	cv2.release()
	cv2.destroyWindow('frame')

def ImageProcessing(frame):
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		filtered = cv2.GaussianBlur(gray,(5,5),0)
		#cv2.fastNlMeansDenoising(frame,None,10,7,21)
		ret,thresh = cv2.threshold(filtered,225,255,cv2.THRESH_BINARY)

		return thresh

def Destructor():
	cv2.release()
	cv2.destroyAllWindows()

def main():
	ConnectCamera()


def Destructor():
	cv2.release()
	cv2.destroyAllWindows()
	
if __name__ == '__main__':
	main()
