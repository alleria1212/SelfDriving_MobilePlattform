import cv2


def SaveData(img):
    cv2.imwrite('1.jpg',img)

def Recognition():
    StopCascade = cv2.CascadeClassifier('stop_sign.xml')
    LightCascade = cv2.CascadeClassifier('traffic_light.xml')
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        stopps = StopCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        lights = LightCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        for (x, y, w, h) in stopps:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, 'STOP', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #SaveData(frame)

        threshold = 150;
        for (x, y, w, h) in lights:
            roi = gray[y+10:y + h-10, x+10:x + w-10]      #Area of Interest
            mask = cv2.GaussianBlur(roi, (25, 25), 0)
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(mask)
            if maxVal - minVal > threshold:
                cv2.circle(roi, maxLoc, 5, (255, 0, 0), 2)

                if 1.0/8*(h-30) < maxLoc[1] < 4.0/8*(h-30):
                    cv2.putText(frame, 'Red', (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                elif 5.5/8*(h-30) < maxLoc[1] < h-30:
                    cv2.putText(frame, 'Green', (x+5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow('Video', frame)
        if cv2.waitKey(5000) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def main():
    Recognition()
    
if __name__ == "__main__":
    main()