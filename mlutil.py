import numpy as np
from tensorflow.keras import models, callbacks
import tensorflow as tf
import cv2
import pandas as pd


def makeDebugModel(model, onlyconv=False):
    if onlyconv:
        outputs = [layer.output for layer in model.layers if type(layer) is keras.layers.Conv2D]
    else:
        outputs = [layer.output for layer in model.layers]
    debugmodel = models.Model(inputs=model.inputs, outputs=outputs)
    return debugmodel


def makeLayerOutputModel(model, layernames):
    outputs = [layer.output for layer in model.layers if layer.name in layernames]
    outputmodel = models.Model(inputs=model.inputs, outputs=outputs)
    return outputmodel


def getConvInfo(model):
    convinfo = [{'name': l.name, 'kernel': l.kernel_size if 'conv2d' in l.name else l.pool_size, 'stride': l.strides}
                for l in model.layers if 'conv2d' in l.name or 'max_pooling2d' in l.name]
    return convinfo


def layerIndex(model, layername):
    for li in range((len(model.layers))):
        if model.layers[li].name == layername:
            return li
    return None


def calcReceptiveField(x, y, layers):
    startx = x
    endx = x
    starty = y
    endy = y
    for l in reversed(layers):
        startx = startx * l['stride'][1]
        endx = endx * l['stride'][1] + l['kernel'][1] - 1
        starty = starty * l['stride'][0]
        endy = endy * l['stride'][0] + l['kernel'][0] - 1
    return (startx, endx), (starty, endy)


def heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output[0]]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = (tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)).numpy()
    (w, h) = (img_array.shape[2], img_array.shape[1])
    heatmapimage = cv2.resize(heatmap, (w, h))

    return heatmap, heatmapimage


def overlayHeatmap(image, heat, alpha=0.4):
    (w, h) = (image.shape[2], image.shape[1])
    heatmap = cv2.resize(heat, (w, h))
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + 0
    heatmap = numer / denom
    heatmap = (heatmap * 255).astype("uint8")
    colormap = cv2.COLORMAP_VIRIDIS
    heatmap = cv2.applyColorMap(heatmap, colormap)
    output = cv2.addWeighted(image[0], alpha, heatmap, 1 - alpha, 0)
    return output


def getLayerOutputRange(model, layernames, images):
    layermodel = makeLayerOutputModel(model, layernames)

    class LayerCallback(callbacks.Callback):
        ranges = []

        def __init__(self, model):
            for l in model.output_shape:
                self.ranges.append([(9999.0, -9999.0) for _ in range(l[-1])])
            super(callbacks.Callback, self).__init__()


        def on_predict_batch_end(self, batch, logs=None):
            outs = logs['outputs']
            for i in range(0, len(outs)):
                for fi in range(outs[i].shape[-1]):
                    fmaps = outs[i][:, :, :, fi] if len(outs[i].shape) == 4 else outs[i]
                    minval = tf.reduce_min(fmaps).numpy()
                    maxval = tf.reduce_max(fmaps).numpy()
                    c = self.ranges[i][fi]
                    self.ranges[i][fi] = (min(c[0], minval), max(c[1], maxval))

    layerinfo = LayerCallback(layermodel)
    layermodel.predict(images, callbacks=[layerinfo])
    return layerinfo.ranges


def getConvLayerSubset(model, lastlayer):
    convinfo = getConvInfo(model)
    idx = layerIndex(model, lastlayer)
    convname = model.layers[idx - 1].name
    receptivelayers = []
    for l in convinfo:
        receptivelayers.append(l)
        if l['name'] == convname:
            break
    return receptivelayers


def combineHeatAndFeatures(model, franges, outputlayers, lastconvlayer, image):
    # Get initial outputs
    layermodel = makeLayerOutputModel(model, outputlayers)
    outs = layermodel.predict(np.asarray([image]))
    #outs = layermodel(np.asarray([image]), training=False)
    predicted = np.argmax(outs[-1])
    _, himg = heatmap(np.asarray([image]), model, lastconvlayer, predicted)

    himg[himg < .7] = 0
    himg[himg >= .7] = 1

    heats = []
    for layerindex in range(len(outputlayers) - 1):

        # Get subset of conv layers
        receptivelayerinfo = getConvLayerSubset(model, outputlayers[layerindex])

        # layerindex = 1
        fmaps = outs[layerindex][0]
        r = franges[layerindex]
        currlayerheats = []
        for fi in range(fmaps.shape[2]):
            if r[fi][1] == 0.0:
                currfmap = fmaps[:, : fi]
            else:
                currfmap = (fmaps[:, :, fi] - r[fi][0]) / (r[fi][1] - r[fi][0])
            heatfeatmap = np.zeros(currfmap.shape)
            for x in range(currfmap.shape[1]):
                for y in range(currfmap.shape[0]):
                    xrange, yrange = calcReceptiveField(x, y, receptivelayerinfo)
                    v = currfmap[y, x]

                    # Use all vals in receptive field on heatmap
                    hr = himg[yrange[0]:yrange[1] + 1, xrange[0]:xrange[1] + 1]
                    heatfeatmap[y, x] = v * np.sum(hr)

                    # Only use center val in receptive field on heatmap (assume 3x3 kernel)
                    # heatfeatmap[y, x] = v * himg[yrange[0] + 1, xrange[0] + 1]
            currlayerheats.append(heatfeatmap)
        heats.append(currlayerheats)
    return heats, predicted


def featuresToDataFrame(model, outputlayers, lastlayer, franges, indexes, images, labels):

    # Generate header row
    head = ['index']
    for l in outputlayers[:-1]:
        layer = model.get_layer(l)
        for i in range(layer.output.shape[-1]):
            head.append(l + '-' + str(i))
    head.append('predicted')
    head.append('label')

    # Convert each list of feature map activations to transactions
    transactions = []
    for i in range(len(images)):
        print(str(i) + '/' + str(len(images)))
        index = indexes[i]
        img = images[i]
        label = labels[i]
        heats, predicted = combineHeatAndFeatures(model, franges, outputlayers, lastlayer, img)

        trans = [index]
        for layerindex in range(len(heats)):
            for fi in range(len(heats[layerindex])):
                v = heats[layerindex][fi].max()
                trans.append(v)
        trans.append(predicted)
        trans.append(label)
        transactions.append(trans)

    df = pd.DataFrame(transactions, columns=head)
    df.set_index('index', inplace=True)
    return df
        