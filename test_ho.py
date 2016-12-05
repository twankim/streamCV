import numpy as np
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detect(gray):
    faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30,30),
                flags=cv2.cv.CV_HAAR_SCALE_IMAGE
             )
    return faces

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    faces = face_detect(gray)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.imwrite('ex.png',frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
