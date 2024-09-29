import tensorflow as tf
import keras
from keras import layers, models
from keras.src.models import Model
from keras.src.metrics.accuracy_metrics import accuracy
import keras.src.trainers.data_adapters.data_adapter_utils as data_adapter_utils
import numpy as np
import os
import joblib
import data
import patterns as pats
import mlutil


@tf.keras.utils.register_keras_serializable()
class PatternMatch(layers.Layer):
    def __init__(self, pattern_feats, scaler, **kwargs):
        super().__init__(**kwargs)
        self.pattern_feats = list(pattern_feats)
        self.scaler = scaler

    def compute_output_shape(self, input_shape):
        return [None, 1]

    def call(self, inputs):
        matches = tf.constant(pats.features_match(inputs, self.pattern_feats, self.scaler))
        return matches

@tf.keras.utils.register_keras_serializable()
class PickBestBranch(layers.Layer):
    def __init__(self, base_pred, pat_pred, pattern_class, threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.base_pred = base_pred
        self.pat_pred = pat_pred
        self.pattern_class = pattern_class
        self.threshold = threshold

    def call(self, inputs):
        presults = self.pat_pred(inputs)
        tlist = list()
        for i, r in enumerate(presults):
            v = r.numpy().item()
            if v >= self.threshold:
                pout = np.asarray([(1.0 - v) / 2.0, (1.0 - v) / 2.0, (1.0 - v) / 2.0])
                pout[self.pattern_class] = v
                tlist.append(tf.constant(pout, dtype=tf.float32))
            else:
                bresult = self.base_pred(inputs)
                tlist.append(tf.squeeze(bresult))

        res = tf.stack(tlist)
        return res


@tf.keras.utils.register_keras_serializable()
class PatternSelect(layers.Layer):
    def __init__(self, pattern, **kwargs):
        super().__init__(**kwargs)
        self.pattern = list(pattern)
        self.pattern.sort()

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], input_shape[2], len(self.pattern)]

    def call(self, inputs):
        outputs = tf.gather(inputs, indices=self.pattern, axis=3)
        return outputs


@tf.keras.utils.register_keras_serializable()
class PatternLayer(layers.Layer):

    def __init__(self, pattern, pattern_set, pattern_class, base_model=None, scaler=None, **kwargs):
        super().__init__(**kwargs)
        self.pattern = list(pattern)
        self.pattern.sort()
        self.pattern_set = pattern_set
        self.pattern_feats = [int(mlutil.parse_feature_ref(e)[1]) for e in self.pattern]
        self.pattern_set_feats = [int(mlutil.parse_feature_ref(e)[1]) for e in self.pattern_set]
        self.pattern_layer = mlutil.parse_feature_ref(self.pattern[0])[0]
        self.pattern_class = pattern_class
        self.base_model = base_model
        self.base_pred = None
        self.pat_pred = None
        self.pat_match = None
        self.pat_branch = None
        self.pat_train = None
        self.feat_extract = None
        self.branch_model = None
        self.scaler = scaler
        self.build_output_shape = [None, 1]
        if self.base_model is not None:
            self.build_branches()

    def get_config(self):
        config = super().get_config()
        config.update({
            'pattern': self.pattern,
            'pattern_set': self.pattern_set,
            'pattern_class': self.pattern_class
        })
        return config

    def get_build_config(self):
        build_config = {"output_shape": self.output.shape}
        return build_config

    def build_from_config(self, config):
        self.build_output_shape = config['output_shape']

    def save_assets(self, inner_path):
        self.base_model.trainable = True
        self.pat_pred.trainable = True
        self.base_model.save(os.path.join(inner_path, 'base_model.keras'))
        self.pat_pred.save(os.path.join(inner_path, 'pat_model.keras'))
        joblib.dump(self.scaler, os.path.join(inner_path, 'scaler.save'))

    def load_assets(self, inner_path):
        self.base_model = models.load_model(os.path.join(inner_path, 'base_model.keras'), compile=True)
        self.base_model.trainable = False
        self.base_model.compile()
        self.pat_pred = models.load_model(os.path.join(inner_path, 'pat_model.keras'), compile=True)
        self.pat_pred.trainable = False
        self.pat_pred.compile()
        self.scaler = joblib.load(os.path.join(inner_path, 'scaler.save'))
        self.build_branches()

    def build_branches(self):
        player = self.base_model.get_layer(self.pattern_layer)
        self.feat_extract = Model(inputs=self.base_model.inputs, outputs=player.output, name='feat_extract')
        self.feat_extract.trainable = False

        pi = self.base_model.layers.index(player) + 1
        self.base_pred = Model(self.base_model.layers[pi].input, self.base_model.output)

        if self.pat_pred is None:
            inputs = keras.Input(shape=self.feat_extract.output_shape[1:])
            x = PatternSelect(self.pattern_set_feats)(inputs)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Conv2D(64, (3, 3))(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(32, (3, 3))(x)
            x = layers.Activation('relu')(x)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(1, name='prediction', activation='sigmoid')(x)
            self.pat_pred = Model(inputs=inputs, outputs=x, name='pat_pred')

        if self.pat_match is None:
            self.pat_match = PatternMatch(self.pattern_feats, self.scaler)

        if self.pat_branch is None:
            self.pat_branch = PickBestBranch(self.base_pred, self.pat_pred, self.pattern_class)

        if self.pat_train is None:
            inputs = keras.Input(shape=self.feat_extract.input_shape[1:])
            x = self.feat_extract(inputs)
            x = self.pat_pred(x)
            self.pat_train = Model(inputs=inputs, outputs=x)
            self.pat_train.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                                   metrics=[tf.keras.metrics.BinaryAccuracy()])

    def build(self, input_shape):
        self.built = True

    def compute_output_shape(self, input_shape):
        if self.base_model is None:
            return self.build_output_shape
        else:
            return self.base_model.output_shape

    def fit(self, ds, **kwargs):

        pattern = set(self.pattern)
        classname = ds.class_names[self.pattern_class]

        # Filter training data
        transpath = kwargs['trans_path'] if 'trans_path' in kwargs else None
        bdf, s = pats.preprocess(self.base_model, ds, transpath, self.scaler)
        if self.scaler is None:
            self.scaler = s
        matches, nonmatches = pats.matches(bdf, pattern)
        matchds = data.load_dataset_selection(ds, bdf.loc[matches]['path'].to_list())
        binds = data.split_dataset_paths(matchds, classname, label_mode='binary')

        # Filter validation data
        valbinds = None
        if 'validation_data' in kwargs:
            valds = kwargs['validation_data']
            valpath = kwargs['val_path'] if 'val_path' in kwargs else None
            bdf, _ = pats.preprocess(self.base_model, valds, valpath, self.scaler)
            matches, nonmatches = pats.matches(bdf, pattern)
            matchds = data.load_dataset_selection(valds, bdf.loc[matches]['path'].to_list())
            valbinds = data.split_dataset_paths(matchds, classname, label_mode='binary')

        # Train pattern branch
        epochs = kwargs['epochs'] if 'epochs' in kwargs else 1
        self.pat_train.fit(binds, validation_data=valbinds, epochs=epochs)

    def call(self, inputs):
        x = self.feat_extract(inputs)
        matches = self.pat_match(x)
        tlist = list()
        for i, m in enumerate(matches):
            s = x[i, ...][None, ...]
            if m:
                r = self.pat_branch(s)
                tlist.append(tf.squeeze(r))
            else:
                r = self.base_pred(s)
                tlist.append(tf.squeeze(r))

        res = tf.stack(tlist)
        return res


@tf.keras.utils.register_keras_serializable()
class PatternModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pat_layer = self.layers[-1]

    def build(self, input_shape):
        super().build(input_shape)
        self.built = True

    def fit(self, ds, **kwargs):
        #super().fit(ds)
        self.reset_metrics()
        self.pat_layer.fit(ds, **kwargs)

    def evaluate2(self, ds, **kwargs):
        outputs = None
        for step, (x_batch, y_batch) in enumerate(ds):
            #print(step)
            y_pred = self(x_batch)
            #ll = y_batch.numpy().tolist()
            #ll = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
            #yy = [[0.1, 0.8, 0.1], [0.4, 0.5, 0.1], [0.1, 0.1, 0.8]]
            #b = keras.metrics.categorical_accuracy(ll, yy)
            #abc = tf.math.reduce_sum(b)
            #c = accuracy(ll, yy)
            #a = keras.metrics.categorical_accuracy(y_batch, y_pred)

            #print('pred')
        print(1)

    def test_step(self, data):
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)
        if self._call_has_training_arg:
            y_pred = self(x, training=False)
        else:
            y_pred = self(x)
        loss = self._compute_loss(
            x=x, y=y, y_pred=y_pred, sample_weight=sample_weight, training=False
        )
        #self._loss_tracker.update_state(
        #    loss, sample_weight=tf.shape(tree.flatten(x)[0])[0]
        #)
        return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)