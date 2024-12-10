import tensorflow as tf
import keras
from keras import models
import cv2
import numpy as np


@tf.keras.utils.register_keras_serializable()
class EvalModel(models.Model):

    def __init__(self, layers, **kwargs):
        super().__init__(**kwargs)
        self.receptive_info = conv_layer_info(self, self.output[-2].name)
        self.output_names = layers
        for i in range(len(layers)):
            self.output[i].name = layers[i]

    def get_config(self):
        config = super().get_config()
        config.update({
            'layers': self.output_names
        })
        return config

    def receptive_subset(self, layer):
        idx = self.layers.index(self.get_layer(layer)) - 1
        name = self.layers[idx].name
        rlayers = []
        for l in self.receptive_info:
            rlayers.append(l)
            if l['name'] == name:
                break
        return rlayers

    def filters_in_layer(self, layer):
        numfilters = self.get_layer(layer).output.shape[-1]
        return [layer + '-' + str(i) for i in range(numfilters)]

    def load_image(self, path):
        imagedata = cv2.imread(path)
        imagedata = cv2.resize(imagedata, (self.input_shape[1], self.input_shape[2]))
        return imagedata


def make_output_model(model, layers=None):
    if layers is None:
        layers = [l.name for l in model.layers if isinstance(l, keras.layers.Activation)]
        layers.append('prediction')
    outputs = [layer.output for layer in model.layers if layer.name in layers]
    outputmodel = EvalModel(layers, inputs=model.inputs, outputs=outputs)
    return outputmodel


def make_output_nodes(model, output_layers=None):
    if output_layers is None:
        output_layers = [l.name for l in model.layers if isinstance(l, keras.layers.Activation)]
        output_layers.append('prediction')
    elif output_layers[-1] != 'prediction':
        output_layers.append('prediction')
    outputs = [layer.output for layer in model.layers if layer.name in output_layers]
    outputmodel = keras.src.Model(inputs=model.inputs, outputs=outputs)
    return outputmodel


def conv_layer_info(model, last=None):
    convinfo = []
    for i in range(len(model.layers)):
        if last is not None and model.layers[i].name == last:
            break
        l = model.layers[i]
        if 'conv2d' in l.name or 'max_pooling2d' in l.name or l.name.startswith('average_pooling2d'):
            convinfo.append({'name': l.name, 'kernel': l.kernel_size if 'conv2d' in l.name else l.pool_size, 'stride': l.strides})

    return convinfo


def layer_subset(layers, last):
    selected = []
    for l in layers:
        selected.append(l)
        if l == last:
            break
    selected.append(layers[-1])
    return selected


def slice_weights(model, layer, filters):
    widx = model.layers.index(model.get_layer(layer)) - 1
    weights = model.layers[widx].weights
    w = weights[0].numpy()[..., filters]
    b = weights[1].numpy()[filters]
    return [w, b]


def parse_feature_ref(name):
    tokens = name.split('-')
    return tokens[0], int(tokens[1])


def receptive_field(x, y, layers):
    startx = x
    endx = x
    starty = y
    endy = y
    for l in reversed(layers):
        startx = startx * l['stride'][1]
        endx = endx * l['stride'][1] + l['kernel'][1] - 1
        starty = starty * l['stride'][0]
        endy = endy * l['stride'][0] + l['kernel'][0] - 1
    return (startx, endx + 1), (starty, endy + 1)    # +1 so it's in [start, end) format


def map_receptive_field(fmap, input_shape, layerinfo):
    mask = np.zeros((input_shape[0], input_shape[1]), np.float32)
    for y in range(fmap.shape[0]):
        for x in range(fmap.shape[1]):
            v = fmap[y, x] #.numpy()
            if v > 0:
                xf, yf = receptive_field(x, y, layerinfo)
                rfield = mask[yf[0]:yf[1], xf[0]:xf[1]]
                vals = np.empty(rfield.shape)
                vals.fill(v)
                np.maximum(rfield, vals, out=rfield)
    return mask


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


def categorical_to_binary(batch, target):
    return tf.map_fn(lambda x: 1 if tf.math.argmax(x) == target else 0, batch)
