import numpy as np
import cv2
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description=
                        'Real-time Face Detection')
    parser.add_argument('-fps', dest='set_fps',
                        help='frames per second',
                        default = 10, type=int)
    args = parser.parse_args()
    return args

def main(args):
    fps_input = args.set_fps
    cap = cv2.VideoCapture(0)
    cap.set(cv2.cv.CV_CAP_PROP_FPS,fps_input)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
        start = time.time()
        faces = face_cascade.detectMultiScale(
                                    gray,
                                    scaleFactor=1.1,
                                    minNeighbors=5,
                                    minSize=(30,30),
                                    flags=cv2.cv.CV_HAAR_SCALE_IMAGE
                                )
        print "Detection... {:.3f}s".format(time.time()-start)
    
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_args()
    print ("Called with args:")
    print (args)

    main(args)
