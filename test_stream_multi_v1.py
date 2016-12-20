import numpy as np
import cv2
import time
import argparse
from sys import platform

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
                        default = 1, type = int)
    parser.add_argument('-verbose', dest='verbose',
                        help='verbose is True if >= 1',
                        default = 0, type = int)
    parser.add_argument('-record', dest='record_on',
                        help='Save recorded video if >0',
                        default = 0, type = int)
    parser.add_argument('-out', dest='out_name',
                        help='Output file name',
                        default = 'out', type = str)
    args = parser.parse_args()
    return args

def cam_init(cid,fps_input,seth):
    cap = cv2.VideoCapture(cid)
    cap.set(cv2.cv.CV_CAP_PROP_FPS,fps_input)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    ret,frame = cap.read()
    assert ret, "!!!!!!!! Cam ID {} is not working".format(cid)
    set_ratio = seth/float(frame.shape[0])
    setw = int(frame.shape[1]*set_ratio)

    return {"cap":cap, "ret":ret, "set_ratio":set_ratio,\
            "setw":setw, "detector": face_cascade,
            "w_org":frame.shape[1],"h_org":frame.shape[0]}

def vwriter_init(cid,out_name,fps_input,w,h):
    if platform == "linux" or platform == "linux2":
        # linux
        fourcc = cv2.cv.CV_FOURCC(*'XVID')
    elif platform == "darwin":
        # OS X
        fourcc = cv2.cv.CV_FOURCC(*'mp4v')
    elif platform == "win32":
        # Windows...
        fourcc = cv2.cv.CV_FOURCC(*'MJPG')
    else:
        fourcc = cv2.cv.CV_FOURCC(*'MJPG')
    vout = cv2.VideoWriter('{}_{}.mp4'.format(out_name,cid), fourcc, float(fps_input), (w,h))
    return vout

def main(args):
    fps_input = args.set_fps
    seth = args.set_height
    setm = args.set_multi
    out_name = args.out_name

    if args.verbose >= 1:
        def vprint( obj ):
            print obj
    else:   
        def vprint( obj ):
            pass

    if args.record_on > 0:
        record_on = True
    else:
        record_on = False

    if setm < 1:
        setm = 1

    caps = [cam_init(cid,fps_input,seth) for cid in range(setm)]

    if record_on:
        vwriters = [vwriter_init(cid,out_name,fps_input,caps[cid]["w_org"],caps[cid]["h_org"]) for cid in range(setm)]

    while(True):
        # Capture frame-by-frame
        ret_frames = [cap["cap"].read() for cap in caps]
        
        newframes = [cv2.resize(ret_frame[1],(cap["setw"],seth)) \
                     for ret_frame,cap in zip(ret_frames,caps)]

        # Our operations on the frame come here
        grays = [cv2.cvtColor(newframe, cv2.COLOR_BGR2GRAY) for newframe in newframes]
        start = time.time()
        faces_list = [cap["detector"].detectMultiScale(
                                                       grays[cid],
                                                       scaleFactor=1.1,
                                                       minNeighbors=5,
                                                       minSize=(30,30),
                                                       flags=cv2.cv.CV_HAAR_SCALE_IMAGE
                                                      ) for cid,cap in enumerate(caps)]
                                
        vprint("Detection... {:.3f}s ({} cameras)".format(time.time()-start,setm))

        frames = [ret_frame[1] for ret_frame in ret_frames]

        for cid,(cap,faces) in enumerate(zip(caps,faces_list)):
            set_ratio = cap["set_ratio"]
            frame = frames[cid]
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,
                              (int(x/set_ratio),int(y/set_ratio)),
                              (int((x+w)/set_ratio),int((y+h)/set_ratio)),
                              (255,0,0),2)

        # Display the resulting frame
        for cid, frame in enumerate(frames):
            if record_on:
                vwriters[cid].write(frame)
            cv2.imshow('Cam {}'.format(cid),frame)
        
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    # When everything done, release the capture
    for cap in caps:
        cap["cap"].release()
        if record_on:
            vwriters[cid].release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_args()
    print ("Called with args:")
    print (args)

    main(args)
