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
    parser.add_argument('-seth', dest='set_height',
                        help='resize image to certain height',
                        default = 480, type = int)
    parser.add_argument('-multi', dest='set_multi',
                        help='Use multiple camera sensors',
                        default = False, type = bool)
    args = parser.parse_args()
    return args

def main(args):
    fps_input = args.set_fps
    seth = args.set_height
    setm = args.set_multi
    
    cap1 = cv2.VideoCapture(0)
    cap1.set(cv2.cv.CV_CAP_PROP_FPS,fps_input)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    ret,frame = cap1.read()
    set_ratio = seth/float(frame.shape[0])
    setw = int(frame.shape[1]*set_ratio)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap1.read()
        
        newframe = cv2.resize(frame,(setw,seth))

        # Our operations on the frame come here
        gray = cv2.cvtColor(newframe, cv2.COLOR_BGR2GRAY)
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
            cv2.rectangle(frame,(int(x/set_ratio),int(y/set_ratio)),
                          (int((x+w)/set_ratio),int((y+h)/set_ratio)),
                          (255,0,0),2)

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    # When everything done, release the capture
    cap1.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_args()
    print ("Called with args:")
    print (args)

    main(args)
