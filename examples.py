import argparse
import time

import numpy as np
import cv2

import matplotlib
from matplotlib import pyplot as plt

import rto

# global for plt callback
posx, posy = 0,0


def main(args):

    order = args.d
    mexp = rto.MotionExplorer(inputdim = 2)

    def mouse_cb(event,x,y,flags,param):
        global posx, posy
        posx, posy = x, y

    # Create a black image, a window and bind the function to window
    img = np.ones((512,512,3), np.uint8)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_cb)

    lastms = int(round(time.time() * 1000))

    fig, ax = plt.subplots(1, 2)
    ax0 = ax[0]
    ax1 = ax[1]

    ax0.set_ylim([0,512])
    x = np.linspace(1,100,100)
    posx_arr = np.zeros(100)
    posx_ = ax0.plot(x, posx_arr, 'o')[0]

    f_posx = np.zeros(100)
    f_posx_ = ax0.plot(x, f_posx, 'o')[0]

    f_velx = np.zeros(100)
    f_velx_ = ax0.plot(x, f_velx, 'o')[0]

    scatter_data = 512*np.random.random((100,2))
    scxy = ax1.scatter(scatter_data[:,0], scatter_data[:,1])

    while True:

        newms = int(round(time.time() * 1000))

        img[posy, posx] = [0,0,255]
        cv2.imshow('image', img)

        posx_arr = np.roll(posx_arr,-1)
        posx_arr[-1] = posx
        posx_.set_data(x, posx_arr)

        f_posx = np.roll(f_posx,-1)
        f_posx[-1] = mexp.last_sample[0]
        f_posx_.set_data(x, f_posx)

        f_velx = np.roll(f_velx,-1)
        f_velx[-1] = 1e2*mexp.last_sample[1]
        f_velx_.set_data(x, f_velx)

        ## extract posx and posy
        scxy.set_offsets(mexp.observations[:, [0, order]])

        plt.pause(0.01)

        score, added = mexp.new_sample(newms, (posx, posy))

        print newms-lastms, posx, posy, score, added


        k = cv2.waitKey(1)
        key = chr(k & 255)
        if key == 'q':
            break
        if key == 'd':
            import ipdb; ipdb.set_trace()

        lastms = newms


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Implementation of ROT with mouse input.')
    parser.add_argument('-d', metavar='X', help='number of derivatives', required=False, default=4)
    args = parser.parse_args()
    main(args)