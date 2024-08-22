import cv2
import numpy as np
import mlutil


def plot_feature_map(fmap, image, frange, layers):

    # Scale image to 0-255
    if frange is None or frange[1] == 0.0:
        scaled = fmap
    else:
        scaled = np.interp(fmap, frange, [0, 255]).astype(np.uint8)

    # Render receptive field mask
    mask = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    for y in range(scaled.shape[0]):
        for x in range(scaled.shape[1]):
            v = scaled[y, x]
            if v > 0:
                xfield, yfield = mlutil.receptive_field(x, y, layers)
                for xf in range(xfield[0], xfield[1] + 1):
                    for yf in range(yfield[0], yfield[1] + 1):
                        if mask[yf, xf][2] < v:
                            mask[yf, xf] = (0, 0, v)

    combined = cv2.addWeighted(image, 0.4, mask, 1.0, 0.0)
    return combined


def plot_features(model, image, fmaps, franges=None):

    # Load image
    if isinstance(image, str):
        imagedata = cv2.imread(image)
        imagedata = cv2.resize(imagedata, (model.input_shape[1], model.input_shape[2]))
    else:
        imagedata = image

    outputnames = [o.name for o in model.output]
    outs = model.predict(np.asarray([imagedata]))

    for f in fmaps:
        layername, findex = mlutil.parse_feature_ref(f)
        oindex = outputnames.index(layername)
        foutput = outs[oindex][0, :, :, findex]
        convinfo = model.receptive_subset(layername)
        r = None if franges is None else franges[f]
        canvas = plot_feature_map(foutput, imagedata, r, convinfo)
        cv2.imwrite('test.png', canvas)

        print(1)


