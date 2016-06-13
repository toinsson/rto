import numpy as np
import sounddevice as sd
import soundfile as sf
import jake
import sys
import time
import cv2

jd = jake.jake_device()
connected = jd.connect_rfcomm("00:50:C2:A1:D0:17")

print connected
if not connected:
    sys.exit(0)
time.sleep(2)

stop_wav, stop_fs = sf.read("./stop.wav", dtype='float32')
start_wav, start_fs = sf.read("./start.wav", dtype='float32')

import rto

order = 4
mexp = rto.MotionExplorer(inputdim = 3, order = order, window = 100)

lastms = int(round(time.time() * 1000))

lastadded = int(round(time.time() * 1000))
lastreject = int(round(time.time() * 1000))
while True:

    newms = int(round(time.time() * 1000))

    acc = jd.acc()
    acc_norm = np.linalg.norm(acc)/1000

    ## filter the max norm of the acc
    if acc_norm < 3:
        score, added = mexp.new_sample(newms, acc)

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
