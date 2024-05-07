import numpy as np
from tensorflow.keras import models
import tensorflow as tf
import cv2


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


def evalModel(X, y, model):
    ypreds = model.predict(X)
    #correct = list()
    #incorrect = list()
    #for i in range(len(y)):
    #    yp = ypreds[i]
    #    maxclass = np.argmax(yp)
    #    correctclass = y[i][0]
    #    if y[i][0] == maxclass:
    #        correct.append(i)
    #    else:
    #        incorrect.append(i)
    #    #print(1)
    ymaxpreds = [np.argmax(x) for x in ypreds]
    eval = [x[0] == x[1] for x in zip(map(lambda x: x[0], y), ymaxpreds)]
    correct = [i for i, elem in enumerate(eval) if elem]
    incorrect = [i for i, elem in enumerate(eval) if not elem]
    wronganswers = [{'index': i, 'correct': y[i][0], 'predicted': ymaxpreds[i]} for i in incorrect]
    return correct, incorrect, wronganswers


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


def calcReceptiveField2(u, v, layers):
    for l in reversed(layers):
        print(l)
        u = u * l['stride'][1]
        v = v * l['stride'][1] + l['kernel'][1] - 1
    return u, v


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