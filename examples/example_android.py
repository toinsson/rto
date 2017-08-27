import time

import socket, traceback

import numpy as np
import scipy
from scipy import constants

import cv2

import sounddevice as sd
import soundfile as sf
stop_wav, stop_fs = sf.read("./stop.wav", dtype='float32')
start_wav, start_fs = sf.read("./start.wav", dtype='float32')

import rto

host = ''
port = 5555

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
s.bind((host, port))

def get_time_ms(): 
    return int(round(time.time() * 1000))

def get_acc_mag_data(data):
    if data == None:
        return None
    data = data[0].split(',')

    # print data

    return_data = dict()

    return_data['t'] = float(data[0])

    ## there is acc AND mag
    return_data['acc'] = np.array([float(d) for d in data[2:5]])

    if len(data) == 9:
        return_data['mag'] = np.array([float(d) for d in data[6:9]])
    else:
        return_data['mag'] = np.zeros(3)

    return return_data


order = 4
mexp = rto.MotionExplorer(inputdim = 3, order = order, window = 100)



lastadded = get_time_ms()
lastreject = get_time_ms()

while True:
    newms = get_time_ms()

    try:
        android_data = s.recvfrom(8192)
    except (KeyboardInterrupt, SystemExit):
        android_data = None
        pass

    # print (get_time_ms() - newms), android_data

    data_parse = get_acc_mag_data(android_data)
    if data_parse == None:
        pass
    else:
        acc = data_parse['acc']

    acc_norm = np.linalg.norm(acc)/1000

    if acc_norm < 3*constants.g:
        score, added = mexp.new_sample(newms, acc)
        print score

    ## acc too high, play reject
    elif (newms - lastreject) > 500:
        lastreject = newms
        sd.play(stop_wav[:5000], stop_fs, device=sd.default.device, blocking=False)

    ## new vector, play start
    if added and (newms - lastadded) > 500:
        lastadded = newms
        sd.play(start_wav[:5000], start_fs, device=sd.default.device, blocking=False)

    ## for loop control only
    cv2.imshow('random', np.random.random((10,10,3)))

    ## 30 FPS sample on the JAKE
    time.sleep(1./30)

    k = cv2.waitKey(1)
    key = chr(k & 255)
    if key == 'q':
        break
    if key == 'd':
        import ipdb; ipdb.set_trace()

    lastms = newms


print 'number of saved original vectors: ', mexp.observations.shape[0]
np.save('jake_observations_'+time.strftime("%H:%M:%S"), mexp.observations)
