import cv2
import numpy as np
import mlutil


def plot_feature_map(fmap, image, frange, layers):
    mask = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)

    # Scale image to 0-255
    if frange is None:  # Scale by local feature only
        mn = fmap.min()
        mx = fmap.max()
        if mx > 0.0:
            scaled = np.interp(fmap, (mn, mx), [0, 255]).astype(np.uint8)
        else:
            return image, mask  # No output to mask
    elif frange[1] == 0.0:
        return image, mask  # No output to mask
    else:   # Scale from global feature range
        scaled = np.interp(fmap, frange, [0, 255]).astype(np.uint8)

    # Render receptive field mask
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
    return combined, mask


def plot_features(model, image, fmaps, franges=None):

    # Load image
    if isinstance(image, str):
        imagedata = cv2.imread(image)
        imagedata = cv2.resize(imagedata, (model.input_shape[1], model.input_shape[2]))
    else:
        imagedata = image

    outs = model.predict(np.asarray([imagedata]))

    plots = {}
    for f in fmaps:
        layername, findex = mlutil.parse_feature_ref(f)
        oindex = model.output_names.index(layername)
        foutput = outs[oindex][0, :, :, findex]
        convinfo = model.receptive_subset(layername)
        r = None if franges is None else franges[f]
        canvas, _ = plot_feature_map(foutput, imagedata, r, convinfo)
        plots[f] = canvas

    return plots


def output_features(model, image, fmaps, pathprefix, franges=None):
    plots = plot_features(model, image, fmaps, franges)

    for f in fmaps:
        path = pathprefix + '-' + f + '.png'
        cv2.imwrite(path, plots[f])
