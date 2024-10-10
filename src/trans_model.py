from keras import layers
import tensorflow as tf
import keras
from keras.src.models import Model
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
        return outputs


@tf.keras.utils.register_keras_serializable()
class BinarizeLayer(layers.Layer):
    def __init__(self, layer_names=None, **kwargs):
        super().__init__(**kwargs)
        self.layer_names = layer_names
        self.medians = None

    def get_config(self):
        config = super().get_config()
        config.update({
            'layer_names': self.layer_names
        })
        return config

    def build(self, input_shape):
        super().build(input_shape)
        self.medians = self.add_weight(shape=(input_shape[-1],), initializer='zeros', trainable=True, name='medians')

    def compute_output_shape(self, input_shape):
        return input_shape

    def binarize_median(self, x):
        return tf.cond(tf.math.logical_and(tf.math.greater(x[1], 0.0), tf.math.greater_equal(x[0], x[1])),
                lambda: tf.constant(1.0), lambda: tf.constant(0.0))

    def binarizer(self, x):
        c = tf.stack([x, self.medians], axis=1)
        r = tf.map_fn(self.binarize_median, c)
        return r

    def call(self, inputs, training=None):
        binned = tf.map_fn(self.binarizer, inputs)
        return binned

    @staticmethod
    @tf.function
    def median_caller(model, batch):
        outs = model(batch, training=True)
        return outs

    @staticmethod
    @tf.function
    def calc_median(x):
        m = x.get_shape()[0] // 2
        med = tf.reduce_min(tf.nn.top_k(x, m, sorted=False).values)
        return med

    @staticmethod
    def calculate_medians(model, ds):
        bds = ds.repeat(1)
        numbatches = tf.data.experimental.cardinality(ds).numpy()
        p = Progbar(numbatches)
        r = []
        i = 0
        for x, y in bds:
            outs = BinarizeLayer.median_caller(model, x)
            r.append(outs)
            i += 1
            p.update(i)
        c = tf.concat(r, axis=0)

        vals = tf.transpose(c)
        md = tf.map_fn(BinarizeLayer.calc_median, vals)
        return md

    @staticmethod
    def train(base_model, ds):
        base_model.trainable = False
        layermodel = mlutil.make_output_nodes(base_model)

        inputs = keras.Input(base_model.input_shape[1:])
        xb = layermodel(inputs)
        x = [TransactionLayer()(xo) for xo in xb[:-1]]
        x = layers.Concatenate()(x)
        tmodel = Model(inputs=inputs, outputs=x)    # Initial model to get base model layer outputs
        medians = BinarizeLayer.calculate_medians(tmodel, ds)

        x = BinarizeLayer(layer_names=layermodel.output_names, name='pat_transform')(x)
        xd = layers.Identity()(xb[-1])  # will not deserialize without this
        tmodel = Model(inputs=inputs, outputs=[x, xd])  # Final model to binarize base model layer outputs
        tmodel.get_layer('pat_transform').set_weights([medians])

        return tmodel


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
