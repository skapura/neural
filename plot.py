import cv2
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
import os
import shutil
import mlutil


def renderFeatureMaps(images, model, outputlayers, franges=None):
    layermodel = mlutil.makeLayerOutputModel(model, outputlayers)

    # Load images
    imagedata = []
    for index, imagepath in images.items():
        img = cv2.imread(imagepath)
        img = cv2.resize(img, (256, 256))
        imagedata.append(img)

    outs = layermodel.predict(np.asarray(imagedata))

    # Render features maps for each image
    for i in range(len(images)):
        index = images.index.values[i]

        # Initialize output directory
        imagedir = 'session/' + str(index).zfill(5)
        outpath = imagedir + '/features'
        if os.path.exists(outpath):
            shutil.rmtree(outpath)
        if not os.path.exists(imagedir):
            os.makedirs(imagedir)
        os.makedirs(outpath)

        # Iterate each layer/fmap
        for oi in range(len(outs) - 1):
            for fi in range(outs[oi].shape[-1]):
                fmap = outs[oi][i, :, :, fi]

                # Calculate min/max for scaling
                if franges is None:
                    f_min, f_max = fmap.min(), fmap.max()
                else:
                    f_min, f_max = franges['range'][oi][fi][0], franges['range'][oi][fi][1]

                # Scale image to 0-255
                if f_max == 0.0:
                    scaled = fmap
                else:
                    scaled = np.interp(fmap, [f_min, f_max], [0, 255]).astype(np.uint8)

                # Render feature map
                filename = 'feature-' + str(index).zfill(5) + '-' + outputlayers[oi] + '-' + str(fi).zfill(4) + '.png'
                cv2.imwrite(outpath + '/' + filename, scaled)


def renderHeatmaps(images, model, lastconvlayer, useimagedir=True):

    for index, row in images.iterrows():
        imagepath = row['imagepath']
        label = row.iloc[-1]

        # Initialize output directory
        if useimagedir:
            imagedir = 'session/' + str(index).zfill(5)
            if not os.path.exists(imagedir):
                os.makedirs(imagedir)
        else:
            imagedir = 'session/'

        # Render heatmap
        img = cv2.imread(imagepath)
        img = cv2.resize(img, (256, 256))
        h, _ = mlutil.heatmap(np.asarray([img]), model, lastconvlayer, label)
        heatout = mlutil.overlayHeatmap(np.asarray([img]), h)
        cv2.imwrite(imagedir + '/heat-' + str(index).zfill(5) + '.png', heatout)
        cv2.imwrite(imagedir + '/image-' + str(index).zfill(5) + '.png', img)



def renderFeatureActivation(image, model, fmapname, franges=None):

    # Get receptive layer info
    tokens = fmapname.split('-')
    layername = tokens[0]
    fmapindex = int(tokens[1])
    layerindex = mlutil.layerIndex(model, layername)
    convlayername = model.layers[layerindex - 1].name
    layerinfo = mlutil.getConvInfo(model)
    receptivelayers = []
    for l in layerinfo:
        receptivelayers.append(l)
        if l['name'] == convlayername:
            break

    layermodel = mlutil.makeLayerOutputModel(model, [layername])
    outs = layermodel.predict(np.asarray([image]))
    fmap = outs[0, :, :, fmapindex]

    # Calculate min/max for scaling
    if franges is None:
        f_min, f_max = fmap.min(), fmap.max()
    else:
        f_min, f_max = franges['name'][layername][fmapindex][0], franges['name'][layername][fmapindex][1]

    # Scale image to 0-255
    if f_max == 0.0:
        scaled = fmap
    else:
        scaled = np.interp(fmap, [f_min, f_max], [0, 255]).astype(np.uint8)

    # Create receptive field mask
    mask = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    for y in range(scaled.shape[0]):
        for x in range(scaled.shape[1]):
            v = scaled[y, x]
            xfield, yfield = mlutil.calcReceptiveField(x, y, receptivelayers)
            for xf in range(xfield[0], xfield[1] + 1):
                for yf in range(yfield[0], yfield[1] + 1):
                    if v > 0:
                        mask[yf, xf] = (0, 0, v)

    combined = cv2.addWeighted(image, 1.0, mask, 1.0, 0)
    return combined, mask, image


def renderPattern(patternid, pattern, model, trans, indexlist, franges=None):

    # Initialize output directory
    imagedir = 'session/'
    outpath = imagedir + '/patterns/' + str(patternid).zfill(4)
    if os.path.exists(outpath):
        shutil.rmtree(outpath)
    if not os.path.exists(imagedir):
        os.makedirs(imagedir)
    if not os.path.exists(imagedir + '/patterns'):
        os.makedirs(imagedir + '/patterns')
    os.makedirs(outpath)

    layers = list(set([t.split('-')[0] for t in pattern['pattern']]))
    layers.sort()
    layermodel = mlutil.makeLayerOutputModel(model, layers)

    for index in indexlist:
        path = trans.loc[index, 'imagepath']
        image = cv2.imread(path)
        image = cv2.resize(image, (256, 256))
        outs = layermodel.predict(np.asarray([image]))
        for p in pattern['pattern']:
            t = p.split('-')
            layername = t[0]
            t = t[1].split('_')
            fmapindex = int(t[0])
            if t[1] == '1':     # Only output if this is an activated feature
                layerindex = layers.index(layername)
                fmap = outs[layerindex][0, :, :, fmapindex]
                receptivelayers = mlutil.getConvLayerSubset(model, layername)

                # Calculate min/max for scaling
                if franges is None:
                    f_min, f_max = fmap.min(), fmap.max()
                else:
                    f_min, f_max = franges['name'][layername][fmapindex][0], franges['name'][layername][fmapindex][1]

                # Scale image to 0-255
                if f_max == 0.0:
                    scaled = fmap
                else:
                    scaled = np.interp(fmap, [f_min, f_max], [0, 255]).astype(np.uint8)

                # Create receptive field mask
                mask = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
                for y in range(scaled.shape[0]):
                    for x in range(scaled.shape[1]):
                        v = scaled[y, x]
                        xfield, yfield = mlutil.calcReceptiveField(x, y, receptivelayers)
                        for xf in range(xfield[0], xfield[1] + 1):
                            for yf in range(yfield[0], yfield[1] + 1):
                                if v > 0:
                                    mask[yf, xf] = (0, 0, v)

                combined = cv2.addWeighted(image, 1.0, mask, 1.0, 0)
                filepath = 'pattern-' + str(index).zfill(5) + '-' + layername + '-' + str(fmapindex)
                cv2.imwrite(outpath + '/' + filepath + '-activation.png', combined)
                cv2.imwrite(outpath + '/' + filepath + '-mask.png', mask)


def renderFeatureMaps2(mapinfo, pattern=None):
    fh = 30#maps.shape[0]
    fw = 30#maps.shape[1]
    maps = mapinfo['output']
    patitems = []
    for item in pattern:
        i = item.rfind('_')
        layername = item[:i]
        if layername == mapinfo['name']:
            patitems.append(item)
    xmaps = 30
    numfilters = maps.shape[-1]
    ymaps = math.ceil(numfilters / xmaps)
    f_min, f_max = maps.min(), maps.max()
    scaled = np.interp(maps, [f_min, f_max], [0, 255]).astype(np.uint8)
    (_, lh), _ = cv2.getTextSize('0', cv2.FONT_HERSHEY_PLAIN, 1, 1)
    img = np.full(((fh + lh + 2) * ymaps + 1, (fw + 1) * xmaps + 1), 255, np.uint8)
    x = 1
    y = 0
    for i in range(0, numfilters):
        filter = scaled[:, :, i]
        #r = np.rollaxis(filter, 1)  # swap HxW->WxH for cv2
        if mapinfo['name'] + '_' + str(i) in patitems:
            thickness = 2
        else:
            thickness = 1
        (_, lh), _ = cv2.getTextSize(str(i), cv2.FONT_HERSHEY_PLAIN, 1, thickness)
        img = cv2.putText(img, str(i), (x, y + lh + 1), cv2.FONT_HERSHEY_PLAIN, 1, 0, thickness)
        r = cv2.resize(filter, (fh, fw), interpolation=cv2.INTER_NEAREST_EXACT)
        img[y + lh + 2:y + r.shape[0] + lh + 2, x:x + r.shape[1]] = r

        #if mapinfo['name'] + '_' + str(i) in patitems:
        #    cv2.rectangle(img, (x, y), (x + r.shape[1], y + r.shape[0] + lh + 2), (0, 255, 0), 2)

        if (i + 1) % xmaps == 0:
            x = 1
            y += fh + lh + 2
        else:
            x += fw + 1

    return img


def rotate3(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (
    width / 2, height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return rotated_mat


def renderClassScores(classes, output):
    scaler = MinMaxScaler(feature_range=(-100, 100))
    rout = scaler.fit_transform(output.reshape(-1, 1)).flatten()
    maxindex = max(enumerate(output), key=lambda x: x[1])[0]

    # Get max text rendered length
    th = 0
    for c in classes:
        (lw, _), _ = cv2.getTextSize(c, cv2.FONT_HERSHEY_PLAIN, 1, 1)
        th = max(th, lw)

    canvas = np.full((th + 200, len(classes) * 20 - 5, 3), 255, np.uint8)

    # Render bar plot
    x = 0
    i = 0
    for o in rout:
        y = 100 - int(o)
        y1 = min(y, 100) - 1
        y2 = max(y, 100) - 1
        if i == maxindex:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)
        cv2.rectangle(canvas, (x, y1), (x + 15, y2), color, -1)
        x += 20
        i += 1
    cv2.line(canvas, (0, 99), (canvas.shape[0] - 1, 99), (0, 0, 0), 2)

    # Render class labels
    y = 200
    x = 0
    for c in classes:
        (lw, lh), _ = cv2.getTextSize(c, cv2.FONT_HERSHEY_PLAIN, 1, 1)
        tc = np.full((lh, lw, 3), 255, np.uint8)
        tc = cv2.putText(tc, c, (0, lh), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        tc = rotate3(tc, 90)
        canvas[y:y + tc.shape[0], x:x + tc.shape[1]] = tc
        x += 20

    return canvas


def renderSummary(image, title, label, maps, classes, output, pattern=None):
    rmaps = [renderFeatureMaps(m, pattern) for m in maps]
    #rmaps = [renderFeatureMaps(m['output'], pattern) for m in maps]
    rclasses = renderClassScores(classes, output)

    (_, lh), _ = cv2.getTextSize('0', cv2.FONT_HERSHEY_PLAIN, 1, 1)
    h = sum(m.shape[0] for m in rmaps) + rclasses.shape[0] + image.shape[0] + len(rmaps) * (lh + 2) + 6
    w = max(m.shape[1] for m in rmaps)
    canvas = np.full((h, w, 3), 255, np.uint8)

    x = 0
    y = 1
    canvas[y:y + image.shape[0], x + 1:x + image.shape[1] + 1] = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    canvas = cv2.putText(canvas, title + ' - ' + label, (x + image.shape[1] + 5, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

    y += image.shape[0] + 5
    for i in range(len(rmaps)):
        rgbmap = cv2.cvtColor(rmaps[i], cv2.COLOR_GRAY2RGB)
        title = maps[i]['name'] + ' - ' + str(maps[i]['output'].shape[0]) + 'x' + str(maps[i]['output'].shape[1]) + 'x' + str(maps[i]['output'].shape[2])
        canvas = cv2.putText(canvas, title, (1, y + lh), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        y += lh + 1
        canvas[y:y + rgbmap.shape[0], x:x + rgbmap.shape[1]] = rgbmap
        y += rgbmap.shape[0] + 1

    canvas[y:y + rclasses.shape[0], 1:1 + rclasses.shape[1]] = rclasses
    return canvas


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

    #cv2.imwrite('testa.png', imga)
    #cv2.imwrite('test.png', mask)
    #cv2.imwrite('test2.png', scaled)
    #cv2.imwrite('test3.png', combined)
    return combined, mask
