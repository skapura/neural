import tensorflow as tf
import keras
from keras import layers, models
from keras.src.models import Model
import os
from trans_model import BinarizeLayer, TransactionLayer
import mlutil


@tf.keras.utils.register_keras_serializable()
class PatternMatch(layers.Layer):
    def __init__(self, medians, **kwargs):
        super().__init__(**kwargs)
        self.medians = medians
        self.binarizer = BinarizeLayer()

    def build(self, input_shape):
        super().build(input_shape)
        self.binarizer.build(input_shape)
        if self.medians is not None:
            self.binarizer.set_weights([self.medians])

    def compute_output_shape(self, input_shape):
        return [None, 1]

    def call(self, inputs):
        binaryinputs = self.binarizer(inputs)
        matches = tf.reduce_all(tf.cast(binaryinputs, tf.bool), axis=1)
        return matches


@tf.keras.utils.register_keras_serializable()
class PatternSelect(layers.Layer):
    def __init__(self, pattern_feats, **kwargs):
        super().__init__(**kwargs)
        self.pattern_feats = list(pattern_feats)
        self.pattern_feats.sort()

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], input_shape[2], len(self.pattern_feats)]

    def call(self, inputs):
        outputs = tf.gather(inputs, indices=self.pattern_feats, axis=3)
        return outputs


class BinaryToCategorical(layers.Layer):
    def __init__(self, pattern_class, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.pattern_class = pattern_class
        self.num_classes = num_classes

    def compute_output_shape(self, input_shape):
        return [None, self.num_classes]

    def fan(self, x):
        d = self.num_classes - 1
        y = tf.stack([x if i == self.pattern_class else (1.0 - x) / d for i in range(self.num_classes)], axis=0)
        return y

    def call(self, inputs):
        m = tf.map_fn(lambda x: self.fan(x), inputs)
        m = tf.reshape(m, shape=(inputs.shape[0], self.num_classes))
        return m


@tf.keras.utils.register_keras_serializable()
class PatternLayer(layers.Layer):
    def __init__(self, pattern, pattern_class, medians, base_model=None, **kwargs):
        super().__init__(**kwargs)
        self.pattern = list(pattern)
        self.pattern.sort()
        self.pattern_feats = [int(mlutil.parse_feature_ref(e)[1]) for e in self.pattern]
        self.pattern_layer = mlutil.parse_feature_ref(self.pattern[0])[0]
        self.pattern_class = pattern_class
        self.medians = medians
        self.pattern_medians = tf.gather(self.medians, indices=self.pattern_feats, axis=0)
        self.base_model = base_model
        self.feat_extract = None
        self.base_pred = None
        self.pat_pred = None
        self.pat_branch = None
        self.pat_train = None
        if self.base_model is None:
            self.build_output_shape = [None, 1]
        else:
            self.base_model.trainable = False
            self.build_output_shape = self.base_model.output_shape
            self.build_branches()

    def get_config(self):
        config = super().get_config()
        config.update({
            'pattern': self.pattern,
            'pattern_class': self.pattern_class,
            'medians': self.medians
        })
        return config

    def get_build_config(self):
        build_config = {"output_shape": self.output.shape}
        return build_config

    def build_from_config(self, config):
        self.build_output_shape = config['output_shape']

    def save_assets(self, inner_path):
        self.base_model.trainable = True
        self.base_model.save(os.path.join(inner_path, 'base_model.keras'))
        self.pat_pred.save(os.path.join(inner_path, 'pat_model.keras'))

    def load_assets(self, inner_path):
        self.base_model = models.load_model(os.path.join(inner_path, 'base_model.keras'), compile=True)
        self.base_model.trainable = False
        self.pat_pred = models.load_model(os.path.join(inner_path, 'pat_model.keras'), compile=True)
        self.build_branches()

    def build_branches(self):

        # Feature extraction before pattern
        player = self.base_model.get_layer(self.pattern_layer)
        self.feat_extract = Model(inputs=self.base_model.inputs, outputs=player.output, name='feat_extract')

        # Base model prediction
        pi = self.base_model.layers.index(player) + 1
        self.base_pred = Model(self.base_model.layers[pi].input, self.base_model.output)

        # Pattern model prediction
        p = self.feat_extract.output_shape[1:]
        inputs = keras.Input(shape=(p[0], p[1], len(self.pattern)))
        x = layers.MaxPooling2D((2, 2))(inputs)
        x = layers.Conv2D(64, (3, 3))(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(32, (3, 3))(x)
        x = layers.Activation('relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1, name='prediction', activation='sigmoid')(x)
        x = BinaryToCategorical(pattern_class=self.pattern_class, num_classes=3)(x)
        self.pat_pred = Model(inputs=inputs, outputs=x, name='pat_pred')

        # Pattern branch
        inputs = keras.Input(shape=self.feat_extract.output_shape[1:])
        xt = PatternSelect(self.pattern_feats)(inputs)
        x = TransactionLayer()(xt)
        xm = PatternMatch(medians=self.pattern_medians)(x)
        self.pat_branch = Model(inputs=inputs, outputs=[xt, xm])

        return

        # Pattern batch train
        inputs = keras.Input(shape=self.feat_extract.input_shape[1:])
        x = self.feat_extract(inputs)
        x = self.pat_pred(x)
        self.pat_train = Model(inputs=inputs, outputs=x)
        self.pat_train.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                               metrics=[tf.keras.metrics.BinaryAccuracy()])

    def compute_output_shape(self, input_shape):
        if self.base_model is None:
            return self.build_output_shape
        else:
            return self.base_model.output_shape

    def call(self, inputs):

        # Feature extraction
        feats = self.feat_extract(inputs)
        pouts = self.pat_branch(feats)

        # Separate
        matchindex = tf.where(pouts[1])
        nonmatchindex = tf.where(tf.equal(pouts[1], False))
        matches = tf.reshape(matchindex, shape=-1)
        nonmatches = tf.reshape(nonmatchindex, shape=-1)
        mergedpreds = tf.TensorArray(inputs.dtype, size=inputs.shape[0])

        # Pattern branch
        if matches.shape[0] > 0:
            patbatch = tf.gather(pouts[0], indices=matches, axis=0)
            patpreds = self.pat_pred(patbatch)
            for i in range(matches.shape[0]):
                mergedpreds.write(matches[i], patpreds[i]).mark_used()

        # Base model branch
        if nonmatches.shape[0] > 0:
            basebatch = tf.gather(feats, indices=nonmatches, axis=0)
            basepreds = self.base_pred(basebatch)
            for i in range(nonmatches.shape[0]):
                mergedpreds.write(nonmatches[i], basepreds[i]).mark_used()

        preds = mergedpreds.stack()
        return preds
