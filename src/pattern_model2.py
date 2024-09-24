import tensorflow as tf
from keras import layers, models, losses, metrics
from keras.src.models import Model
import pandas as pd
import numpy as np
import joblib
import os
import data
import patterns as pats
import mlutil


@tf.keras.utils.register_keras_serializable()
class PatternSel(layers.Layer):
    def __init__(self, pattern, **kwargs):
        super().__init__(**kwargs)
        self.pattern = pattern

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], input_shape[2], len(self.pattern)]

    def call(self, inputs):
        outputs = tf.gather(inputs, indices=self.pattern, axis=3)
        return outputs


@tf.keras.utils.register_keras_serializable()
class PatternLayer(layers.Layer):

    def __init__(self, pattern, pattern_class, **kwargs):
        super().__init__(**kwargs)
        self.pattern = pattern
        self.pattern_feats = [int(mlutil.parse_feature_ref(e)[1]) for e in self.pattern]
        self.pattern_class = pattern_class
        self.layer_name = mlutil.parse_feature_ref(pattern[0])[0]
        self.build_output_shape = [None, 1]
        self.base_model = None
        self.pat_model = None
        self.pat_feat_extract = None
        self.base_model_prediction = None
        self.pre_pat_model = None
        self.pat_model_prediction = None
        self.scaler = None
        self.eval_mode = False
        self.eval_metrics = []
        self.eval_pat_total = 0
        self.eval_base_total = 0

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
        self.base_model = models.load_model(os.path.join(inner_path, 'base_model.keras'), compile=True)
        self.pat_model = models.load_model(os.path.join(inner_path, 'pat_model.keras'), compile=True)
        self.build_model_segments()

    def compute_output_shape(self, input_shape):
        if self.base_model is None:
            return self.build_output_shape
        else:
            return self.base_model.output_shape

    def build_branch(self, base_model):
        self.base_model = base_model
        self.base_model.trainable = False
        basecopy = models.clone_model(self.base_model)
        #basecopy = self.base_model

        pidx = basecopy.layers.index(basecopy.get_layer(self.layer_name))
        beforepat = basecopy.layers[pidx - 2]

        # Pattern-filter conv2d
        cfg = basecopy.layers[pidx - 1].get_config()
        cfg['filters'] = len(self.pattern)
        patlayer = layers.Conv2D(**cfg)
        x = patlayer(beforepat.output)

        # Pattern-filter activation
        filteridx = [int(e.split('-')[1]) for e in self.pattern]
        w = mlutil.slice_weights(basecopy, self.layer_name, filteridx)
        patlayer.set_weights(w)
        cfg = basecopy.layers[pidx].get_config()
        x = layers.Activation(**cfg)(x)

        # Pattern branch after pattern filter
        x = layers.Conv2D(16, (3, 3), name='pat_hook')(x)
        x = layers.Activation('relu', name='pat_activation')(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1, name='prediction', activation='sigmoid')(x)

        # Build pattern model
        self.pat_model = Model(inputs=basecopy.inputs, outputs=x)
        self.pat_model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                               metrics=[tf.keras.metrics.BinaryAccuracy()])

        self.build_model_segments()

    def build_model_segments(self):

        # Build feature extraction model for pattern detection
        player = self.base_model.get_layer(self.layer_name)
        self.pat_feat_extract = Model(inputs=self.base_model.inputs, outputs=player.output)

        # Build remainder of base model after pattern detection
        nextlayer = self.base_model.layers.index(player) + 1
        self.base_model_prediction = Model(inputs=self.base_model.layers[nextlayer].input, outputs=self.base_model.output)

        # Pattern branch
        hooklayer = self.pat_model.get_layer('pat_hook')
        self.pat_model_prediction = Model(inputs=hooklayer.input, outputs=self.pat_model.output)

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
        self.pat_model.fit(binds, validation_data=valbinds, epochs=epochs)

    def eval_prediction(self, features, m):
        info = {'is_match': m}
        p = 0.0
        if m:
            selfeats = features.numpy()[..., self.pattern_feats]
            p = self.pat_model_prediction(np.asarray([selfeats])).numpy()[0][0]
            if p >= 0.5:
                info['pat_used'] = True
                info['pat_score'] = p
                pout = [(1.0 - p) / 2.0, (1.0 - p) / 2.0, (1.0 - p) / 2.0]
                pout[self.pattern_class] = p
                info['output'] = tf.constant([pout])
        if not m or p < 0.5:
            info['output'] = self.base_model_prediction(np.asarray([features]))
            info['pat_used'] = False
            info['base_score'] = info['output'].numpy()[0]

        return info

    def call(self, inputs):
        feats = self.pat_feat_extract(inputs)
        matches = pats.features_match(feats, self.pattern_feats, self.scaler)
        plist = list()
        for i, m in enumerate(matches):

            info = self.eval_prediction(feats[i, ...], m)
            plist.append(info['output'])
            self.eval_metrics.append(info)

            # pv = 0.0
            # pmetric = {}
            # pmetric['ismatch'] = False
            # pmetric['patscore'] = 0.0
            # if m:
            #     selfeats = feats[i, ...].numpy()[..., self.pattern_feats]
            #     p = self.pat_model_prediction(np.asarray([selfeats]))
            #     pv = p.numpy()[0][0]
            #     if self.eval_mode:
            #         self.eval_pat_total += 1
            #         pmetric['ismatch'] = True
            #         pmetric['patscore'] = pv
            #         pmetric['basescore'] = self.base_model_prediction(np.asarray([feats[i, ...]])).numpy()[0]
            #     if pv >= 0.5:
            #         pl = [(1.0 - pv) / 2.0, (1.0 - pv) / 2.0, (1.0 - pv) / 2.0]
            #         pl[self.pattern_class] = pv
            #         plist.append(tf.constant([pl]))
            # else:
            #     self.eval_base_total += 1
            # if not m or pv < 0.5:
            #     p = self.base_model_prediction(np.asarray([feats[i, ...]]))
            #     plist.append(p)
            # if self.eval_mode:
            #     self.eval_metrics.append(pmetric)
        predictions = tf.concat(plist, 0)
        return predictions


@tf.keras.utils.register_keras_serializable()
class PatternModel(Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pat_layer = self.layers[-1]

    @staticmethod
    def make(base_model, pattern, pattern_class, scaler=None):
        patlist = list(pattern)
        patlist.sort()
        player = PatternLayer(patlist, pattern_class)
        player.scaler = scaler
        player.build_branch(base_model)
        x = player(base_model.input)
        pmodel = PatternModel(inputs=base_model.input, outputs=x)
        return pmodel

    def fit(self, ds, **kwargs):
        self.pat_layer.fit(ds, **kwargs)

    #def call(self, inputs, **kwargs):
    #    print(1)

    def evaluate(self, ds, **kwargs):
        self.pat_layer.eval_metrics.clear()
        self.pat_layer.eval_pat_total = 0
        self.pat_layer.eval_base_total = 0
        self.pat_layer.eval_mode = True
        kwargs['return_dict'] = True
        r = super().evaluate(ds, **kwargs)
        self.pat_layer.eval_mode = False

        pc = self.pat_layer.pattern_class
        y_true = [1.0 if np.argmax(y) == pc else 0.0 for _, y in ds]

        ytmatch = list()
        ypmatch = list()
        ybmatch = list()
        for (yt, m) in zip(y_true, self.pat_layer.eval_metrics):
            if m['pat_used']:
                ytmatch.append(yt)
                ypmatch.append(m['pat_score'])
                #ybmatch.append(m['basescore'])

        bce = losses.BinaryCrossentropy(from_logits=False)
        r['pat_accuracy'] = metrics.binary_accuracy(np.array(ytmatch), np.array(ypmatch)).numpy()
        r['pat_loss'] = bce(np.array(ytmatch), np.array(ypmatch)).numpy()

        return r

    def evaluate2(self, ds, trans_path=None):
        bdf, _ = pats.preprocess(self.pat_layer.base_model, ds, trans_path, self.pat_layer.scaler)
        patds, baseds = pats.match_dataset(ds, bdf, self.pat_layer.pattern, self.pat_layer.pattern_class)
        p = self.pat_layer.pat_model.evaluate(patds, return_dict=True)
        b = self.pat_layer.base_model.evaluate(baseds, return_dict=True)
        return {'base': b, 'pattern': p}

