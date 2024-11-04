import tensorflow as tf
import keras
from keras import layers, models
from keras.src.models import Functional
import pandas as pd
import pickle
from azure import AzureSession
import data
import patterns as pats
from trans_model import build_transaction_model, transactions_to_dataframe, build_feature_extraction
from pattern_model import build_pattern_model, pat_evaluate, PatternTrainer, build_pattern_trainer


def build_base_model():
    inputs = keras.Input(shape=(256, 256, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(16, (3, 3))(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(3, name='prediction', activation='softmax')(x)
    model = Functional(inputs, x)
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    return model


def train_base_model_local(trainds, model_path, epochs):
    base_model = build_base_model()
    base_model.fit(trainds, epochs=epochs)
    base_model.save(model_path)
    return base_model


def train_base_model_remote(session, model_path, epochs):
    base_model = build_base_model()
    base_model = session.train(base_model, epochs=epochs)
    base_model.save(model_path)
    return base_model

def remote_pat():
    trainds = data.load_from_directory('datasets/images_large/train', shuffle=True)
    v = data.load('session/temp_vars.pickle')
    subds = data.image_subset(v['images'], v['labels'], trainds.class_names, binary_target=0)

    featextract = models.load_model('session/temp_feat_extract.keras')
    patpred = models.load_model('session/temp_pat_pred.keras')
    patsetindex = data.load('session/temp_pat_index.pickle')

    ptrain = build_pattern_trainer(patsetindex, featextract, patpred)
    ptrain.fit(subds, epochs=2,
               callbacks=keras.callbacks.EarlyStopping(monitor='accuracy', mode='max', patience=3))
    ptrain.layers[-1].pat_pred.save_weights('session/ptrain.weights.h5')

def train_pat_model_remote(session, ptrain, images, labels):

    # Copy support files to VM
    ptrain.layers[-1].feat_extract.save('session/temp_feat_extract.keras')
    ptrain.layers[-1].pat_pred.save('session/temp_pat_pred.keras')
    data.save(list(ptrain.layers[-1].pat_index), 'session/temp_pat_index.pickle')
    data.save({'images': images, 'labels': labels}, 'session/temp_vars.pickle')
    session.put('session/bdf.csv', 'neural/session')
    session.put('session/temp_feat_extract.keras', 'neural/session')
    session.put('session/temp_pat_pred.keras', 'neural/session')
    session.put('session/temp_pat_index.pickle', 'neural/session')
    session.put('session/temp_vars.pickle', 'neural/session')
    session.upload_function(remote_pat)
    session.execute('python src/remote_spec.py')

    print(1)


def run():
    #tf.config.run_functions_eagerly(True)
    session = AzureSession()
    session.open(working_dir='neural')
    trainds = data.load_from_directory('datasets/images_large/train', shuffle=True)
    valds = data.load_from_directory('datasets/images_large/val', shuffle=True)

    # Base Model
    print('base model')
    base_path = 'session/base.keras'
    #base_model = train_base_model_local(trainds, base_path, epochs=10)
    #base_model = train_base_model_remote(session, base_path, epochs=3)
    base_model = models.load_model(base_path)

    # Transaction model
    print('transaction model')
    pat_layers = ['activation']
    trans_path = 'session/tmodel.keras'
    #featextract, _ = build_feature_extraction(base_model, pat_layers)
    #tmodel = build_transaction_model(base_model, trainds, pat_layers)
    #tmodel.save(trans_path)
    tmodel = models.load_model(trans_path)

    # Transaction dataset
    print('generate transaction dataset')
    #trans = featextract.predict(trainds)
    #df = pd.DataFrame(trans, columns=tmodel.get_layer('pat_binarize').feature_names)
    #df.index.name = 'index'
    #btrans = tmodel.predict(trainds)
    #bdf = transactions_to_dataframe(tmodel, btrans, trainds)
    #bdf.to_csv('session/bdf.csv')
    bdf = pd.read_csv('session/bdf.csv', index_col='index')

    # Mine patterns
    print('mine patterns')
    minsup = 0.5
    minsupratio = 1.1
    sel, notsel = data.filter_transactions(bdf, 'activation-', 0)
    cpats = pats.mine_patterns(sel, notsel, minsup, minsupratio)
    elems, targetmatches, othermatches = pats.unique_elements(cpats)
    patternset = list(elems.keys())
    pattern = cpats[1]

    print('train pattern model')
    pat_path = 'session/pmodel.keras'
    #matches = pattern['targetmatches'] + pattern['othermatches']
    matches = list(targetmatches) + list(othermatches)
    images = [trainds.file_paths[p] for p in matches]
    labels = [trainds.labels[p] for p in matches]
    subds = data.image_subset(images, labels, trainds.class_names, binary_target=0)
    pmodel, ptrain = build_pattern_model(pattern['pattern'], patternset, base_model, tmodel)
    train_pat_model_remote(session, ptrain, images, labels)
    #ptrain.fit(subds, epochs=2,
    #           callbacks=keras.callbacks.EarlyStopping(monitor='accuracy', mode='max', patience=3))
    #ptrain.layers[-1].pat_pred.save_weights('session/ptrain.weights.h5')
    #ptrain.layers[-1].pat_pred.load_weights('session/ptrain.weights.h5')
    ptrain.save('session/ptest.keras')
    ptrain = models.load_model('session/ptest.keras')

    print('train dataset evaluation')
    eval_pat_trn = ptrain.evaluate(subds, return_dict=True)
    eval_pat = pat_evaluate(pmodel, trainds)
    eval_base = base_model.evaluate(trainds, return_dict=True)

    print('eval dataset evaluation')
    #v_eval_pat_trn = ptrain.evaluate(subds, return_dict=True)
    v_eval_pat = pat_evaluate(pmodel, valds)
    v_eval_base = base_model.evaluate(valds, return_dict=True)

    print(eval_pat_trn)
    print(eval_pat)
    print(eval_base)

    print('---')

    print(v_eval_pat)
    print(v_eval_base)

    print(1)
