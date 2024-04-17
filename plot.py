import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import mlutil

def plotReceptiveField(image, layers, fmap):
    imga = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fmin = np.min(fmap)
    fmax = np.max(fmap)
    scaled = (((fmap - fmin) / (fmax - fmin)) * 255).astype(np.uint8)
    mask = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)

    # Create receptive field mask
    for y in range(scaled.shape[0]):
        for x in range(scaled.shape[1]):
            v = scaled[y, x]
            xfield, yfield = mlutil.calcReceptiveField(x, y, layers)
            for xf in range(xfield[0], xfield[1] + 1):
                for yf in range(yfield[0], yfield[1] + 1):
                    if v > 0:
                        mask[yf, xf] = (0, max(v, mask[yf, xf, 1]), 0)

    combined = cv2.addWeighted(imga, 1.0, mask, 1.0, 0)

    cv2.imwrite('testa.png', imga)
    cv2.imwrite('test.png', mask)
    cv2.imwrite('test2.png', scaled)
    cv2.imwrite('test3.png', combined)
    return combined, mask
