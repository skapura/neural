import tensorflow as tf
import keras
from keras import layers
from keras.src.models import Model
import cv2
import numpy as np
import os
from pathlib import Path
import mlutil


class ReceptiveFieldLayer(layers.Layer):

    def __init__(self, model_input_shape, num_feature_maps, layer_info, **kwargs):
        super().__init__(**kwargs)
        self.model_input_shape = model_input_shape
        self.num_feature_maps = num_feature_maps
        self.layer_info = layer_info

    def compute_output_shape(self, input_shape):
        return [None, self.model_input_shape[0], self.model_input_shape[1], self.num_feature_maps]

    #@tf.function
    #@tf.py_function(Tout=[tf.float32, tf.float32, tf.float32, tf.float32])
    #@tf.py_function(Tout=tf.float32)
    def transform_feature_map(self, fmap):
        mask = np.zeros((self.model_input_shape[0], self.model_input_shape[1]), np.uint8)
        npfmap = fmap.numpy()
        for y in range(npfmap.shape[0]):
            for x in range(npfmap.shape[1]):
                v = npfmap[y, x]
                if v > 0:
                    xfield, yfield = mlutil.receptive_field(x, y, self.layer_info)
                    for xf in range(xfield[0], xfield[1]):
                        for yf in range(yfield[0], yfield[1]):
                            if mask[yf, xf] < v:
                                mask[yf, xf] = v
        return mask
        #return 1.2

    def transform_map(self, fmap):
        return tf.py_function(self.transform_feature_map, [fmap], tf.float32)

    #@tf.function
    def transform_feature_map_set(self, fmaps):
        f = tf.transpose(fmaps, perm=[2, 0, 1])
        fr = tf.map_fn(self.transform_feature_map, f)
        return fr

    #@tf.function
    def call(self, inputs):
        selectedfeats = tf.gather(inputs, indices=[0, 1, 2], axis=-1)
        #b = tf.transpose(inputs[0], perm=[2, 0, 1])
        #a = self.transform_feature_map(b[0])
        #a = tf.py_function(self.transform_feature_map, [b[0]], tf.float32)
        receptivemasks = tf.map_fn(self.transform_feature_map_set, selectedfeats)
        #receptivemasks = tf.map_fn(self.transform_map, selectedfeats)
        return receptivemasks


def plot_receptive_field(image, receptive_field, output_path):
    #img = cv2.imread(image_path)
    #img = cv2.resize(img, (256, 256))
    if receptive_field.max() == 0.0:
        cv2.imwrite(output_path, image)
        return
    scaled = np.interp(receptive_field, (0.0, receptive_field.max()), [0, 255]).astype(np.uint8)
    k = np.zeros_like(scaled)
    #srgbraw = cv2.cvtColor(scaled, cv2.COLOR_GRAY2RGB)
    #cv2.imwrite('session/test4.png', srgbraw)
    srgb = cv2.merge([k, k, scaled])
    combined = cv2.addWeighted(image, 0.4, srgb, 1.0, 0.0)
    #cv2.imwrite('session/test.png', srgb)
    #cv2.imwrite('session/test2.png', image)
    #cv2.imwrite('session/test3.png', combined)
    cv2.imwrite(output_path, combined)


def plot_feature(fmap, output_path):
    if fmap.max() == 0.0:
        scaled = np.zeros((fmap.shape[0], fmap.shape[1]), np.uint8)
    else:
        scaled = np.interp(fmap, (0.0, fmap.max()), [0, 255]).astype(np.uint8)
    cv2.imwrite(output_path, scaled)


def plot_receptive_field_set(model, medians, features, image_paths, output_dir='session'):
    layername = mlutil.parse_feature_ref(features[0])[0]
    layerinfo = mlutil.conv_layer_info(model, layername)
    featidx = [mlutil.parse_feature_ref(f)[1] for f in features]
    layermodel = mlutil.make_output_nodes(model, [layername])
    for imgpath in image_paths:
        name = Path(imgpath).stem
        print(name)
        img = cv2.imread(imgpath)
        img = cv2.resize(img, model.input_shape[1:3])
        outs = layermodel(np.asarray([img]))
        feats = tf.gather(outs[0][0], indices=featidx, axis=-1)
        feats = tf.transpose(feats, perm=[2, 0, 1])
        for idx, f in zip(featidx, feats):
            print(layername + '-' + str(idx))
            fmedian = medians[idx].numpy()
            fnp = f.numpy()
            fnp[fnp < fmedian] = 0.0
            outpath = os.path.join(output_dir, name + '-' + layername + '-' + str(idx) + '.png')
            #plot_feature(fnp, outpath)
            rmap = mlutil.map_receptive_field(fnp, model.input_shape[1:], layerinfo)
            plot_receptive_field(img, rmap, outpath)
    print(1)


def build_feature_mapper(model, layers):
    layerinfo = mlutil.conv_layer_info(model)
    layermodel = mlutil.make_output_nodes(model, layers)
    inputs = keras.Input(model.input_shape[1:])
    xb = layermodel(inputs)
    outs = []

    for lidx, l in enumerate(layers[:-1]):
        mlayer = model.get_layer(l)
        idx = model.layers.index(mlayer) - 1
        for i in range(len(layerinfo)):
            if layerinfo[i]['name'] == model.layers[idx].name:
                layerindex = i
                break
        linfo = layerinfo[:layerindex+1]
        fcount = mlayer.output.shape[-1]
        rlayer = ReceptiveFieldLayer(model.input_shape[1:], fcount, linfo)
        outs.append(rlayer(xb[lidx]))
    rmodel = Model(inputs=inputs, outputs=outs)
    return rmodel


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
                for xf in range(xfield[0], xfield[1]):
                    for yf in range(yfield[0], yfield[1]):
                        if mask[yf, xf][2] < v:
                            mask[yf, xf] = (0, 0, v)

    combined = cv2.addWeighted(image, 0.4, mask, 1.0, 0.0)
    return combined, mask


def plot_features(model, image, fmaps, franges=None):

    # Load image
    if isinstance(image, str):
        imagedata = model.load_image(image)
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


def overlay_heatmap(image, heatmap, alpha=0.4, cutoff=128):
    if heatmap.max() == 0.0:
        heatmap = heatmap.astype('uint8')
    else:
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + 0
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype('uint8')
    heatmap[heatmap < cutoff] = 0
    #colormap = cv2.COLORMAP_VIRIDIS
    colormap = cv2.COLORMAP_TURBO
    heatmap = cv2.applyColorMap(heatmap, colormap)
    output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
    return output
