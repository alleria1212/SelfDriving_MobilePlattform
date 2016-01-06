import cv2
import numpy as np

def nothing(x):
    pass

def PrintImgSize(img):
	print(np.size(img,0),np.size(img,1))

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
	LaneTracking()
if __name__ == "__main__":
	main()