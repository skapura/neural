from keras import layers, models
import tensorflow as tf
import keras
from keras.src.models import Model, Functional
from keras.src import ops
from keras.src.utils import Progbar
from patterns import feature_activation_max
import mlutil
import numpy as np


@tf.keras.utils.register_keras_serializable()
class TransactionLayer(layers.Layer):
    def __init__(self, feature_activation=feature_activation_max, **kwargs):
        super().__init__(**kwargs)
        self.feature_activation = feature_activation
        self.max_vals = None

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
        if training:
            if self.max_vals is None:
                self.max_vals = bmax
            else:
                cmax = tf.stack([self.max_vals, bmax], axis=1)
                self.max_vals = tf.map_fn(tf.reduce_max, cmax)
        return outputs

    @staticmethod
    def make_model(base_model, feature_activation=feature_activation_max):
        layermodel = mlutil.make_output_nodes(base_model)

        inputs = keras.Input(base_model.input_shape[1:])
        x = layermodel(inputs)
        x = [TransactionLayer(feature_activation)(xo) for xo in x[:-1]]
        x = layers.Concatenate()(x)
        x = TransformLayer(binary_transform=True)(x)
        tmodel = Model(inputs=inputs, outputs=x)
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
    def __init__(self, binary_transform=True, threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.binary_transform = binary_transform
        self.threshold = threshold
        self.max_vals = None

    def get_config(self):
        config = super().get_config()
        config.update({
            'binary_transform': self.binary_transform,
            'threshold': self.threshold
        })
        return config

    def build(self, input_shape):
        super().build(input_shape)
        self.max_vals = self.add_weight(shape=(input_shape[-1],), initializer='zeros', trainable=True, name='max_vals')

    def scale(self, x):
        if x[1] == 0.0:
            return 0.0
        else:
            s = x[0] / x[1]
            return s

    def binarize(self, x):
        s = self.scale(x)
        return 1.0 if s >= self.threshold else 0.0

    def transformer(self, x):
        c = tf.stack([x, self.max_vals], axis=1)
        if self.binary_transform:
            r = tf.map_fn(self.binarize, c)
        else:
            r = tf.map_fn(self.scale, c)
        return r

    def call(self, inputs, training=None):
        bmax = ops.max(inputs, axis=0)     # get max from batch
        if training:
            if self.trainable:
                cmax = tf.stack([self.max_vals, bmax], axis=1)
                self.max_vals.assign(tf.map_fn(tf.reduce_max, cmax))
        else:
            scaled = tf.map_fn(self.transformer, inputs)
            return scaled
        return inputs
