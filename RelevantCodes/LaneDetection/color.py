import cv2
import numpy as np

def getCentroid(img,height,width):
	threshold= 50
	edges = cv2.Canny(img,threshold,threshold*2)
	cnt, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	a=[0]
	b=[0]
	if cnt is not None:
		for c in cnt:
			for e in c:
				for x,y in e:
					a.append(x)
					b.append(y)
		a=int(np.mean(a))
		b=int(np.mean(b))
		if a == 0 and b == 0:
			return height,width
		return a,b
	return height,width


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

		#get the area of interest
		roi_ratio = 2
		#print(img.shape)
		height, width,x = img.shape
		roi_offset = height // roi_ratio
		img = img[roi_offset:height, :width]

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
		height, width = threshed.shape
		roi_w=width//2
		threshed = cv2.addWeighted(threshed,0.5,threshed1,0.5,0)
		cy,cx = getCentroid(threshed,roi_w,height)
		#print(cx,cy)
		cv2.line(threshed,(cy, cx),( roi_w,height),(255,0, 0), 2, cv2.CV_AA)
		if cy - roi_w != 0:
			#h=height-cv

			deltax=roi_w-cy
			deltay=height-cx
			gradient=deltax/float(deltay)
			#print(gradient)
			cv2.putText(threshed, 'Gradient:', (100,200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
			cv2.putText(threshed, str(gradient), (250,200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

			#return gradient  (leftturn is positive slope, rightturn is negative slope)


		#threshed = cv2.arrowedLine = (threshed,(480,720),(cx,cy),(255,0,0),3,8,3)
		cv2.imshow('lane', threshed)
		if cv2.waitKey(1000) & 0xFF == ord('q'):
			break

	video_capture.release()
	cv2.destroyWindow('original')
	cv2.destroyWindow('lane')


def main():
	print('ok')
	trackFinal()			#calculate gradient, dont have a return yet 
if __name__ == "__main__":
	main()