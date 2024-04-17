import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import mlutil

def plotReceptiveField(image, layers, fmap):
    imga = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #imga = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

    fmin = np.min(fmap)
    fmax = np.max(fmap)
    scaled = (((fmap - fmin) / (fmax - fmin)) * 255).astype(np.uint8)
    mask = np.zeros((image.shape[0], image.shape[1], 4), np.uint8)

    # Create receptive field mask
    for y in range(scaled.shape[0]):
        for x in range(scaled.shape[1]):
            v = scaled[y, x]
            xfield, yfield = mlutil.calcReceptiveField(x, y, layers)
            for xf in range(xfield[0], xfield[1] + 1):
                for yf in range(yfield[0], yfield[1] + 1):
                    if v > 0:
                        mask[yf, xf] = (0, 255, 0, max(v, mask[yf, xf, 3]))



    #for y in range(scaled.shape[0]):
    #    for x in range(scaled.shape[1]):
    #        v = scaled[y, x]
    #        xfield, yfield = mlutil.calcReceptiveField(x, y, layers)
    #        mask = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    #        cv2.rectangle(mask, (xfield[0], yfield[0]), (xfield[1], yfield[1]), (0, 255, 0), -1)
    #        #vv = float(v) / 255.0
    #        vv = 0.7
    #        imga = cv2.addWeighted(imga, vv, mask, 1 - vv, 0)
    #        #break
    #    break
            #for xf in range(xfield[0], xfield[1] + 1):
            #    for yf in range(yfield[0], yfield[1] + 1):
            #        if v > 0:
            #            mask[yf, xf] = (0, 255, 0, max(v, mask[yf, xf, 3]))

    mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2RGB)
    combined = cv2.addWeighted(imga, 0.5, mask, 0.5, 0)

    cv2.imwrite('testa.png', imga)
    cv2.imwrite('test.png', mask)
    cv2.imwrite('test2.png', scaled)
    cv2.imwrite('test3.png', combined)
    print(1)
