import argparse
import time


import numpy as np
import cv2

import classes
import config

# global x_, posy
posx, posy = 0,0


def _playsound():
    import pyaudio
    import wave

    #define stream chunk
    chunk = 1024

    #open a wav format music
    f = wave.open(r"./start.wav","rb")
    #instantiate PyAudio
    p = pyaudio.PyAudio()
    #open stream
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),
                    channels = f.getnchannels(),
                    rate = f.getframerate(),
                    output = True)
    #read data
    data = f.readframes(chunk)

    #play stream
    while data != '':
        stream.write(data)
        data = f.readframes(chunk)

    #stop stream
    stream.stop_stream()
    stream.close()

    #close PyAudio
    p.terminate()


def main(args):

    sr = config.sample_rate
    window = int(config.window*config.sample_rate)
    order = 4
    cutoff = config.lowpass
    mexp = classes.MotionExplorer(ndim = 2, window=window, 
                                order=order, sr=sr, filter_cutoff=cutoff)

    def draw_circle(event,x,y,flags,param):
        global posx, posy
        # print event, x,y
        posx, posy = x, y

    # Create a black image, a window and bind the function to window
    img = np.ones((512,512,3), np.uint8)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)

    while True:

        # global posx, posy
        ms = int(round(time.time() * 1000))

        # time.sleep(0.2)

        img[posy, posx] = [0,0,255]
        cv2.imshow('image', img)


        mexp.new_sample(ms, (posx, posy))
        score, added = mexp.knn()

        print ms, posx, posy, mexp.drops, score, added

        ## slow down the loop
        time.sleep(0.01)

        k = cv2.waitKey(1)
        key = chr(k & 255)
        if key == 'q':
            break
        if key == 'd':
            import ipdb; ipdb.set_trace()




if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Implementation of ROT with mouse input.')
    parser.add_argument('-d', metavar='X', help='number of derivatives', required=False, default=0)
    args = parser.parse_args()
    main(args)