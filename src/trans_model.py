from keras import layers, models
import tensorflow as tf
#import tensorflow_probability as tfp
import keras
from keras.src.models import Model, Functional
from keras.src import ops
from keras.src.utils import Progbar
from patterns import feature_activation_max
import mlutil
import pandas as pd
import numpy as np


@tf.keras.utils.register_keras_serializable()
class TransactionLayer(layers.Layer):
    def __init__(self, feature_activation=feature_activation_max, **kwargs):
        super().__init__(**kwargs)
        self.feature_activation = feature_activation
        #self.max_vals = None

    def get_config(self):
        config = super().get_config()
        config.update({
            'feature_activation': self.feature_activation
        })
        return config

    def transform_instance(self, x):
        x = tf.transpose(x, perm=[2, 0, 1])
        x = tf.map_fn(tf.reduce_max, x)
        return x

    def call(self, inputs, training=None):
        outputs = tf.map_fn(self.transform_instance, inputs)
        bmax = ops.max(outputs, axis=0)     # get max from batch
        #if training:
        #    if self.max_vals is None:
        #        self.max_vals = bmax
        #    else:
        #        cmax = tf.stack([self.max_vals, bmax], axis=1)
        #        self.max_vals = tf.map_fn(tf.reduce_max, cmax)
        return outputs

    @staticmethod
    def make_model(base_model, feature_activation=feature_activation_max):
        layermodel = mlutil.make_output_nodes(base_model)

        inputs = keras.Input(base_model.input_shape[1:])
        xb = layermodel(inputs)
        x = [TransactionLayer(feature_activation)(xo) for xo in xb[:-1]]
        x = layers.Concatenate()(x)
        x = TransformLayer(layer_names=layermodel.output_names, transform='binary_median', name='pat_transform')(x)
        xd = layers.Identity()(xb[-1])  # will not deserialize without this
        tmodel = Model(inputs=inputs, outputs=[x, xd])
        return tmodel

    @staticmethod
    def train(model, ds):
        bds = ds.repeat(1)
        numbatches = tf.data.experimental.cardinality(ds).numpy()
        p = Progbar(numbatches)
        for i, batch in enumerate(bds):
            p.update(i + 1)
            model(batch[0], training=True)


@tf.keras.utils.register_keras_serializable()
class TransformLayer(layers.Layer):
    def __init__(self, layer_names=None, transform='scale', threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.layer_names = layer_names
        self.transform = transform
        self.threshold = threshold
        self.max_vals = None
        self.medians = None
        self.medians_calculated = False
        self.median_inputs = []

    def get_config(self):
        config = super().get_config()
        config.update({
            'layer_names': self.layer_names,
            'transform': self.transform,
            'threshold': self.threshold
        })
        return config

    def build(self, input_shape):
        super().build(input_shape)
        self.max_vals = self.add_weight(shape=(input_shape[-1],), initializer='zeros', trainable=True, name='max_vals')
        self.medians = self.add_weight(shape=(input_shape[-1],), initializer='zeros', trainable=True, name='medians')

    def compute_output_shape(self, input_shape):
        return input_shape

    def scale(self, x):
        if x[1] == 0.0:
            return 0.0
        else:
            s = x[0] / x[1]
            return s

    def binarize_threshold(self, x):
        s = self.scale(x)
        return 1.0 if s >= self.threshold else 0.0

    def binarize_median(self, x):
        s = self.scale([x[0], x[2]])
        return 1.0 if s >= x[1] else 0.0

    def scaler(self, x):
        c = tf.stack([x, self.max_vals], axis=1)
        r = tf.map_fn(self.scale, c)
        return r

    def binarizer(self, x):
        if self.transform == 'binary_threshold':
            c = tf.stack([x, self.max_vals], axis=1)
            r = tf.map_fn(self.binarize_threshold, c)
        elif self.transform == 'binary_median':
            c = tf.stack([x, self.medians, self.max_vals], axis=1)
            r = tf.map_fn(self.binarize_median, c)
        return r

    #def transformer(self, x):
    #    c = tf.stack([x, self.max_vals], axis=1)
    #    if self.binary_transform:
    #        r = tf.map_fn(self.binarize, c)
    #    else:
    #        r = tf.map_fn(self.scale, c)
    #    return r

    def calc_median(self, x):
        m = x.get_shape()[0] // 2
        med = tf.reduce_min(tf.nn.top_k(x, m, sorted=False).values)
        return med

    def calculate_medians(self):
        scaled = tf.map_fn(self.scaler, tf.concat(self.median_inputs, axis=0))
        vals = tf.transpose(scaled)
        md = tf.map_fn(self.calc_median, vals)
        self.medians.assign(md)
        self.median_inputs.clear()
        self.medians_calculated = True

    def call(self, inputs, training=None):
        bmax = ops.max(inputs, axis=0)     # get max from batch
        if training:
            if self.trainable:
                cmax = tf.stack([self.max_vals, bmax], axis=1)
                self.max_vals.assign(tf.map_fn(tf.reduce_max, cmax))
                self.median_inputs.append(inputs)
        else:
            if not self.medians_calculated:
                self.calculate_medians()

            if self.transform == 'scale':
                scaled = tf.map_fn(self.scaler, inputs)
            elif self.transform.startswith('binary'):
                scaled = tf.map_fn(self.binarizer, inputs)
            #scaled = tf.map_fn(self.transformer, inputs)
            return scaled
        return inputs


def transactions_to_dataframe(model, trans, ds=None):
    layers = model.get_layer('pat_transform').layer_names
    header = []
    for i in range(len(layers) - 1):
        layername = layers[i]
        nodecount = model.layers[2 + i].input.shape[-1]
        for n in range(nodecount):
            header.append(layername + '-' + str(n))

    df = pd.DataFrame(trans[0], columns=header)
    df.index.name = 'index'

    if ds is not None:
        bds = ds.repeat(1)
        labels = []
        for i, batch in enumerate(bds):
            labels += [np.argmax(x) for x in batch[1]]
        df['predicted'] = [np.argmax(x) for x in trans[1]]
        df['label'] = labels
        df['path'] = ds.file_paths

    return df
