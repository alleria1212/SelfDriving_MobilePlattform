import cv2
import numpy as np
import road

def nothing(x):
    pass

def PrintImgSize(img):
	print(np.size(img,0),np.size(img,1))

def LaneTracking2Vid():
	video_capture = cv2.VideoCapture(0)
	lane_detector = road.LaneDetection(verbose=True)
	while True:
		ret, frame = video_capture.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		lane_detector.analyse_image(frame)
		if lane_detector.have_lane():
			Lane = lane_detector.get_lane()
			center = lane_detector.get_center_line()
			cv2.line(frame, (center.top.x, center.top.y),
				(center.bottom.x, center.bottom.y),
				(12, 128, 232), 2, cv2.CV_AA)
			cv2.line(frame, (center.left.x, center.left.y),
				(center.left.x, center.left.y),
				(12, 128, 232), 2, cv2.CV_AA)
			cv2.line(frame, (center.right.x, center.right.y),
				(center.right.x, center.right.y),
				(12, 128, 232), 2, cv2.CV_AA)

		cv2.imshow('Video', frame)
		if cv2.waitKey(100) & 0xFF == ord('q'):
			break
	video_capture.release()
	cv2.destroyAllWindows()

def LaneTracking2Img():
	img = cv2.imread('new2.jpg')
	#cv2.imshow('original',img)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)	
	lane_detector = road.LaneDetection(verbose=True)
	lane_detector.analyse_image(img)
	if lane_detector.have_lane():
            Lane = lane_detector.get_lane()
            center = lane_detector.get_center_line()

            cv2.line(img, (center.top.x, center.top.y),
                     (center.bottom.x, center.bottom.y),
                     (12, 128, 232), 2, cv2.CV_AA)

            cv2.line(img, (Lane.left.top.x, Lane.left.top.y),
                     (Lane.left.bottom.x, Lane.left.bottom.y),
                     (0, 0, 255), 2, cv2.CV_AA)
            cv2.line(img, (Lane.right.top.x, Lane.right.top.y),
                     (Lane.right.bottom.x, Lane.right.bottom.y),
                     (255, 0, 0), 2, cv2.CV_AA)
	cv2.imshow('original',img)
	cv2.waitKey(-1)

def LaneTracking():
	img = cv2.imread('street1.jpg')
	cv2.imshow('original',img)
	#cv2.resize(img,img,(640,320))
	#img = img [:1300,100:1200]
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray, 50, 500)
	cv2.imshow('canny',edges)
	##while True:
	#cv2.createTrackbar('MIN','tracker',400,1000,nothing)
	#	cv2.createTrackbar('MAX','tracker',1,1000,nothing)
	#	cv2.imshow('tracker',edges)
	#	mind = cv2.getTrackbarPos('MIN','tracker')
	#	maxd = cv2.getTrackbarPos('MAX','tracker')
	#	edges = cv2.Canny(gray, mind, maxd)
	#	if cv2.waitKey(5000) & 0xFF == ord('q'):
	#		break

	minLineLength = 100
	maxLineGap = 5
	lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
	for x1,y1,x2,y2 in lines[0]:
		cv2.line(img,(x1,y1),(x2,y2),(0,0,0),2)
	#lines = cv2.HoughLines(edges,1,np.pi/180,200)
	#print(lines)
	#length = 100;
	#for rho,theta	in lines[0]:
		#a = np.cos(theta)
		#b = np.sin(theta)
		#x0 = a * rho
		#y0 = b * rho
		#print(x0,y0)
		#x1 = int( x0 +  length * (-b))
		#y1 = int( y0 +	length * (a))
		#x2 = int( x0 -  length * (-b))
		#y2 = int( y0 -  length * (a))
		#cv2.line(img,(x1,y1),(x2,y2),(0,0,0),2 )
	cv2.imshow('results', img)
	cv2.waitKey(-1)
	cv2.destroyWindow('results')
	cv2.destroyWindow('original')

def main():
	print('ok')
	LaneTracking2Img()
	#LaneTracking2Vid()
if __name__ == "__main__":
	main()