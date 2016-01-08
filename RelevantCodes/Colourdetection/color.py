import cv2
import numpy as np
import ShadowRemoval
def nothing(x):
    return x

def trackFinal():
	video_capture = cv2.VideoCapture(0)

	#HSV Threshold
	lowH =(172,150,60)
	highH = (179,255,255)
	lowHH =(0,150,60)
	highHH =(5,255,255)	
	while True:

		ret, img = video_capture.read()
		img = cv2.resize(img, (960, 720))
		cv2.imshow('original', img)
		img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)


		####TODO get Region of interest
		#roi_ratio = 0.3
		#height, width = gray.shape
        #roi_offset = height // roi_ratio
        #threshed = im_gray[oi_offset:height, :width]

        #HSV filtering
		threshed = cv2.inRange(img,lowH,highH)
		threshed1 = cv2.inRange(img,lowHH,highHH)
		kernel = np.ones((5,5),np.uint8)

		#filtering for image 1
		threshed = cv2.erode(threshed,kernel,iterations =1)
		threshed = cv2.dilate(threshed,kernel,iterations =1)
		threshed = cv2.dilate(threshed,kernel,iterations=1)
		threshed = cv2.erode(threshed,kernel,iterations =1)

		#filtering for image 2
		threshed1 = cv2.erode(threshed1,kernel,iterations =1)
		threshed1 = cv2.dilate(threshed1,kernel,iterations =1)
		threshed1 = cv2.dilate(threshed1,kernel,iterations=1)
		threshed1 = cv2.erode(threshed1,kernel,iterations =1)

		#blending together
		threshed = cv2.addWeighted(threshed,0.5,threshed1,0.5,0)

		cv2.imshow('lane', threshed)
		if cv2.waitKey(100) & 0xFF == ord('q'):
			break

	video_capture.release()
	cv2.destroyWindow('original')
	cv2.destroyWindow('lane')

def track():
	img = cv2.imread('3.jpg')
	gray = cv2.resize(img, (960, 720))
	cv2.imshow('original', gray)
	res, val = ShadowRemoval.process(gray)
	ShadowRemoval.scripts.desplay("img",res)
	#gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	lowB =(0,0,150)
	highB = (100,100,255)
	mask = cv2.GaussianBlur(gray, (7, 7), 0)
	threshed = cv2.inRange(mask, lowB, highB)




	cv2.imshow('lane', threshed)
	cv2.waitKey(-1)

	cv2.destroyWindow('original')
	cv2.destroyWindow('lane')
def track2():
	video_capture = cv2.VideoCapture(0)
	lowB =(20,20,110)
	highB = (90,90,255)
	lowH =(172,150,60)
	highH = (179,255,255)
	lowHH =(0,150,60)
	highHH =(5,255,255)
	#cv2.namedWindow('image')
	#cv2.createTrackbar('LH','image',0,179,nothing)
	#cv2.createTrackbar('HH','image',0,179,nothing)
	
	#cv2.createTrackbar('LS','image',0,255,nothing)
	#cv2.createTrackbar('HS','image',0,255,nothing)

	#cv2.createTrackbar('LV','image',0,255,nothing)
	#cv2.createTrackbar('HV','image',0,255,nothing)
	while True:

		ret, img = video_capture.read()
		gray = cv2.resize(img, (960, 720))
		cv2.imshow('original', gray)
		#imshow('try',gray)
		#gray = ShadowRemoval.process(gray)
		#cv2.imwrite('test.jpg',res)
		#cv2.imshow('removed',res)
		#gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
		#height, width = gray.shape
		#mask = cv2.GaussianBlur(res, (5, 5), 0)
		#threshed = cv2.inRange(mask, lowB, highB)

#hsv approach
		gray = cv2.cvtColor(gray,cv2.COLOR_BGR2HSV)
		#LH = cv2.getTrackbarPos('LH','image')
		#HH = cv2.getTrackbarPos('HH','image')

		#LS = cv2.getTrackbarPos('LS','image')
		#HS = cv2.getTrackbarPos('HS','image')

		#LV = cv2.getTrackbarPos('LV','image')
		#HV = cv2.getTrackbarPos('HV','image')
		#height, width = im_gray.shape
        #self.__roi_offset = height // self.__roi_ratio
        #im_roi = im_gray[self.__roi_offset:height, :width]



		threshed = cv2.inRange(gray,lowH,highH)
		threshed1 = cv2.inRange(gray,lowHH,highHH)
		kernel = np.ones((5,5),np.uint8)
		threshed = cv2.erode(threshed,kernel,iterations =1)
		threshed = cv2.dilate(threshed,kernel,iterations =1)
		threshed = cv2.dilate(threshed,kernel,iterations=1)
		threshed = cv2.erode(threshed,kernel,iterations =1)
		threshed1 = cv2.erode(threshed1,kernel,iterations =1)
		threshed1 = cv2.dilate(threshed1,kernel,iterations =1)
		threshed1 = cv2.dilate(threshed1,kernel,iterations=1)
		threshed1 = cv2.erode(threshed1,kernel,iterations =1)
		threshed = cv2.addWeighted(threshed,0.5,threshed1,0.5,0)
		#threshed = cv2.inRange(gray, (LH,LS,LV), (HH,HS,HV))
		#mask = cv2.GaussianBlur(threshed,(3,3),0)
		cv2.imshow('lane', threshed)
		if cv2.waitKey(100) & 0xFF == ord('q'):
			break

	video_capture.release()
	cv2.destroyWindow('original')
	cv2.destroyWindow('lane')


def track3():
	video_capture = cv2.VideoCapture(0)
	lowB = (0,0,150)
	highB = (100,100,255)

	cv2.namedWindow('image')
	cv2.createTrackbar('LH','image',0,179,nothing)
	cv2.createTrackbar('HH','image',0,179,nothing)
	
	cv2.createTrackbar('LS','image',0,255,nothing)
	cv2.createTrackbar('HS','image',0,255,nothing)

	cv2.createTrackbar('LV','image',0,255,nothing)
	cv2.createTrackbar('HV','image',0,255,nothing)
	while True:
		ret, img = video_capture.read()
		img = cv2.resize(img, (960, 720))
		#cv2.imshow('original', img)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		#img =inRange(img,)


		LH = cv2.getTrackbarPos('LH','image')
		HH = cv2.getTrackbarPos('HH','image')

		LS = cv2.getTrackbarPos('LS','image')
		HS = cv2.getTrackbarPos('HS','image')

		LV = cv2.getTrackbarPos('LV','image')
		HV = cv2.getTrackbarPos('HV','image')


		img = cv2.inRange(img, (LH,LS,LV), (HH,HS,HV))
		#img = cv2.GaussianBlur(img, (7, 7), 0)
		cv2.imshow('image',img)
		#threshold, img = cv2.threshold(np.array(img, np.uint8), 0, 255, cv2.THRESH_BINARY + cv2. THRESH_OTSU)
		#edges = cv2.Canny(img,50,100) #threshold*0.5,threshold)
 		#lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50,50,40)
		#threshed = cv2.inRange(mask, lowB, highB)

		cv2.waitKey(100)

		#cv2.imshow('lane', img)
		#if cv2.waitKey(100) & 0xFF == ord('q'):
		#	break

	video_capture.release()
	cv2.destroyWindow('original')
	cv2.destroyWindow('lane')


       # height, width = im_gray.shape
       # self.__roi_offset = height // self.__roi_ratio

        #im_roi = im_gray[self.__roi_offset:height, :width]

        im_roi = cv2.GaussianBlur(im_roi, (3, 3), 0)


def main():
	print('ok')
	#i=2
	#for i in range(16):
	track2()
		#i=i+1
	#LaneTracking2Vid()
if __name__ == "__main__":
	main()