import tensorflow as tf
from keras import layers, models
from keras.src.models import Model
import pandas as pd
import joblib
import os
import data
import patterns as pats
import mlutil


@tf.keras.utils.register_keras_serializable()
class PatternLayer(layers.Layer):

    def __init__(self, pattern, pattern_class, **kwargs):
        super().__init__(**kwargs)
        self.pattern = pattern
        self.pattern_class = pattern_class
        self.layername = mlutil.parse_feature_ref(pattern[0])[0]
        self.build_output_shape = [None, 1]
        self.base_model = None
        self.pat_model = None
        self.pat_feat_extract = None
        self.base_model_prediction = None
        self.pre_pat_model = None
        self.pat_model_prediction = None
        self.scaler = None

    def get_build_config(self):
        build_config = {"output_shape": self.output.shape}
        return build_config

    def build_from_config(self, config):
        self.build_output_shape = config['output_shape']

    def save_assets(self, inner_path):
        joblib.dump(self.scaler, os.path.join(inner_path, 'scaler.save'))
        self.base_model.trainable = True
        self.base_model.save(os.path.join(inner_path, 'base_model.keras'))
        self.pat_model.save(os.path.join(inner_path, 'pat_model.keras'))

    def load_assets(self, inner_path):
        self.scaler = joblib.load(os.path.join(inner_path, 'scaler.save'))
        self.base_model = models.load_model(os.path.join(inner_path, 'base_model.keras'))
        self.pat_model = models.load_model(os.path.join(inner_path, 'pat_model.keras'))
        self.build_model_segments()

    def compute_output_shape(self, input_shape):
        if self.base_model is None:
            return self.build_output_shape
        else:
            return self.base_model.output_shape

    def build_branch(self, base_model):
        self.base_model = base_model
        self.base_model.trainable = False

        # player = self.base_model.get_layer(self.layername)
        pidx = self.base_model.layers.index(self.base_model.get_layer(self.layername))
        beforepat = self.base_model.layers[pidx - 2]

        # Pattern-filter conv2d
        cfg = base_model.layers[pidx - 1].get_config()
        cfg['filters'] = len(self.pattern)
        cfg.pop('name', None)
        patlayer = layers.Conv2D(**cfg)
        x = patlayer(beforepat.output)

        # Pattern-filter activation
        filteridx = [int(e.split('-')[1]) for e in self.pattern]
        w = mlutil.slice_weights(base_model, self.layername, filteridx)
        patlayer.set_weights(w)
        cfg = base_model.layers[pidx].get_config()
        cfg.pop('name', None)
        x = layers.Activation(**cfg)(x)

        # Pattern branch after pattern filter
        x = layers.Conv2D(16, (3, 3))(x)
        x = layers.Activation('relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1, name='prediction', activation='sigmoid')(x)

        # Build pattern model
        self.pat_model = Model(inputs=self.base_model.inputs, outputs=x)
        self.pat_model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                               metrics=[tf.keras.metrics.BinaryAccuracy()])

        self.build_model_segments()

    def build_model_segments(self):

        # Build feature extraction model for pattern detection
        player = self.base_model.get_layer(self.layername)
        self.pat_feat_extract = Model(inputs=self.base_model.inputs, outputs=player.output)

        # Build remainder of base model after pattern detection
        nextlayer = self.base_model.layers.index(player) + 1
        self.base_model_prediction = Model(inputs=self.base_model.layers[nextlayer].input, outputs=self.base_model.output)

        # Pattern branch
        branchidx = self.base_model.layers.index(self.base_model.get_layer(self.layername))
        branchlayer = self.base_model.layers[branchidx]
        patbranchlayer = self.pat_model.get_layer(branchlayer.name)
        self.pat_model_prediction = Model(inputs=patbranchlayer.input, outputs=self.pat_model.output)
        #pidx = self.base_model.layers.index(self.base_model.get_layer('activation'))
        #player = self.base_model.layers[pidx - 2]    # layer just before pattern branch
        #self.pre_pat_model = Model(inputs=self.base_model.inputs, outputs=player.output)

        print(1)

    def fit(self, ds, **kwargs):
        pattern = set(self.pattern)
        classname = ds.class_names[self.pattern_class]

        # Filter training data
        trans = pd.read_csv('session/trans_feat_full.csv', index_col='index')
        if self.scaler is None:
            scaled, self.scaler = data.scale(trans, output_range=(0, 1))
        else:
            scaled, _ = data.scale(trans, output_range=(0, 1), scaler=self.scaler)
        bdf = pats.binarize(scaled, 0.5)
        matches, nonmatches = pats.matches(bdf, pattern)
        matchds = data.load_dataset_selection(ds, trans.loc[matches]['path'].to_list())
        binds = data.split_dataset_paths(matchds, classname, label_mode='binary')

        # Filter validation data
        valbinds = None
        if 'validation_data' in kwargs:
            valds = kwargs['validation_data']
            trans = pd.read_csv('session/vtrans_feat_full.csv', index_col='index')
            scaled, _ = data.scale(trans, output_range=(0, 1), scaler=self.scaler)
            bdf = pats.binarize(scaled, 0.5)
            matches, nonmatches = pats.matches(bdf, pattern)
            matchds = data.load_dataset_selection(valds, trans.loc[matches]['path'].to_list())
            valbinds = data.split_dataset_paths(matchds, classname, label_mode='binary')

        # Train pattern branch
        epochs = kwargs['epochs'] if 'epochs' in kwargs else 1
        self.pat_model.fit(binds, validation_data=valbinds, epochs=epochs)

    def call(self, inputs):
        feats = self.pat_feat_extract(inputs)
        predictions = self.base_model_prediction(feats)
        return predictions
