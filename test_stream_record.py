import cv2
import time

import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
## uncomment if ffmpeg is downloaded but not globally installed
#ffmpeg_location = 'ffmpeg/bin/ffmpeg.exe'
#plt.rcParams['animation.ffmpeg_path'] = os.path.realpath(ffmpeg_location)

if __name__ == "__main__":

    cap0 = cv2.VideoCapture(0)    
    cap0.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1920)
    cap0.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 1080)
    ret, frame = cap0.read()
    assert ret
    print frame.shape
    
    figwidth = 10 # inches, must be a factor of frame width and height
    fig = plt.figure(dpi = frame.shape[1]/figwidth,
            figsize=(figwidth, frame.shape[0]/float(frame.shape[1])*figwidth))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    aplt = plt.imshow(frame, aspect='equal',interpolation='none')
    writer = manimation.FFMpegWriter(fps=5., bitrate=250*5)
    start_time = time.time()
    with writer.saving(fig, "writer_test.mp4", fig.dpi):
        try:
            while True:
                time.sleep(.2)
                ret, frame = cap0.read()
                aplt.set_data(frame[:,:,::-1])
                writer.grab_frame()
        except KeyboardInterrupt: pass
    print time.time() - start_time
    cap0.release()