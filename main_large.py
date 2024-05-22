from keras.utils import image_dataset_from_directory
from keras.src.models import Functional
from keras import layers, models
import tensorflow as tf
import numpy as np
from tensorflow import data as tf_data
import keras
import pickle
import pandas as pd
import cv2
import os
import shutil
import mlutil
import plot
import patterns as pats


def buildModel():
    inputs = keras.Input(shape=(256, 256, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, (3, 3))(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(3, name='prediction', activation='softmax')(x)
    model = Functional(inputs, x)
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    return model


def findContrast():
    trans = pd.read_csv('session/trans.csv', index_col='index')

    # Drop columns that are all close to zero
    droplist = []
    for c in trans.columns[:-2]:
        mx = trans[c].max()
        if mx <= 0.05:
            droplist.append(c)
    trans.drop(droplist, axis=1, inplace=True)
    istarget = ((trans['label'] == 0) & (trans['predicted'] != trans['label'])).to_numpy()
    trans.drop(['predicted', 'label', 'imagepath'], axis=1, inplace=True)

    # Discretize dataset
    cutpoints = []
    infolist = []
    for c in trans.columns:
        print(c)
        vals = trans[c].to_numpy()
        cuts, info = pats.cutPoints(vals, istarget)
        cutpoints.append(cuts)
        infolist.append(info)
    bds = pats.discretize(trans, cutpoints)
    bdf = pats.binarize(bds, cutpoints)

    bdf['istarget'] = istarget
    target = bdf[bdf['istarget']]
    other = bdf[~bdf['istarget']]
    target = target[target.columns[:-1]]
    other = other[other.columns[:-1]]
    cpats = pats.mineContrastPatterns(target, other, 0.8, 1.3)
    with open('session/patterns.pkl', 'wb') as f:
        pickle.dump(cpats, f)
    print(1)


def generateTrans(model, images, outputlayers):

    # Get min/max ranges for all feature maps over dataset
    franges = None
    for imagebatch, labelbatch in images:
        franges = mlutil.getLayerOutputRange(model, outputlayers, imagebatch, franges)
    with open('session/franges.pkl', 'wb') as f:
        pickle.dump(franges, f)

    # Get model outputs as transaction dataset
    #trans = mlutil.featuresToDataFrame(model, outputlayers, images)
    #trans.to_csv('session/trans.csv')
    trans = pd.read_csv('session/trans.csv', index_col='index')

    # Drop columns that are all close to zero
    droplist = []
    for c in trans.columns[:-2]:
        mx = trans[c].max()
        if mx <= 0.05:
            droplist.append(c)
    trans.drop(droplist, axis=1, inplace=True)

    iscorrect = (trans['predicted'] == trans['label']).to_numpy()
    trans.drop(['predicted', 'label', 'imagepath'], axis=1, inplace=True)

    # Discretize dataset
    cutpoints = []
    infolist = []
    for c in trans.columns:
        print(c)
        vals = trans[c].to_numpy()
        cuts, info = pats.cutPoints(vals, iscorrect)
        cutpoints.append(cuts)
        infolist.append(info)
    bds = pats.discretize(trans, cutpoints)
    bdf = pats.binarize(bds, cutpoints)

    bdf['iscorrect'] = iscorrect
    correct = bdf[bdf['iscorrect']]
    incorrect = bdf[~bdf['iscorrect']]
    correct = correct[correct.columns[:-1]]
    incorrect = incorrect[incorrect.columns[:-1]]
    cpats = pats.mineContrastPatterns(incorrect, correct, 0.8, 1.3)
    with open('session/patterns.pkl', 'wb') as f:
        pickle.dump(cpats, f)


def evalMatches(model, trans):
    with open('session/patterns.pkl', 'rb') as f:
        cpats = pickle.load(f)
    matches = set()
    cmatches = set()
    for p in cpats[:10]:
        matches.update(p['incorrectmatches'])
        cmatches.update(p['correctmatches'])
        #print(p['supportdiff'])

    incorrect = trans[trans['predicted'] != trans['label']]
    with open('session/franges.pkl', 'rb') as f:
        franges = pickle.load(f)

    plot.renderFeatureMaps(incorrect.loc[[1, 3, 4], 'imagepath'], model, ['activation', 'activation_1', 'activation_2', 'activation_3'], franges)
    plot.renderHeatmaps(incorrect.loc[[1, 3, 4], ['imagepath', 'predicted']], model, 'activation_3', False)

    print(1)


def renderFeatureActivations(index, image, model, fmapname, franges=None):

    # Get receptive layer info
    tokens = fmapname.split('-')
    layername = tokens[0]
    fmapindex = int(tokens[1])
    layerindex = mlutil.layerIndex(model, layername)
    convlayername = model.layers[layerindex - 1].name
    layerinfo = mlutil.getConvInfo(model)
    receptivelayers = []
    for l in layerinfo:
        receptivelayers.append(l)
        if l['name'] == convlayername:
            break

    layermodel = mlutil.makeLayerOutputModel(model, [layername])
    outs = layermodel.predict(np.asarray([image]))
    fmap = outs[0, :, :, fmapindex]

    # Calculate min/max for scaling
    if franges is None:
        f_min, f_max = fmap.min(), fmap.max()
    else:
        f_min, f_max = franges['name'][layername][fmapindex][0], franges['name'][layername][fmapindex][1]

    # Scale image to 0-255
    if f_max == 0.0:
        scaled = fmap
    else:
        scaled = np.interp(fmap, [f_min, f_max], [0, 255]).astype(np.uint8)

    # Create receptive field mask
    mask = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    for y in range(scaled.shape[0]):
        for x in range(scaled.shape[1]):
            v = scaled[y, x]
            xfield, yfield = mlutil.calcReceptiveField(x, y, receptivelayers)
            for xf in range(xfield[0], xfield[1] + 1):
                for yf in range(yfield[0], yfield[1] + 1):
                    if v > 0:
                        mask[yf, xf] = (0, 0, v)

    combined = cv2.addWeighted(img, 1.0, mask, 1.0, 0)

    cv2.imwrite('session/test.png', combined)
    cv2.imwrite('session/test2.png', mask)
    cv2.imwrite('session/test3.png', img)
    print(1)


trainds = mlutil.load_from_directory('images_large/train', labels='inferred', label_mode='categorical', image_size=(256, 256), shuffle=True)
valds = mlutil.load_from_directory('images_large/val', labels='inferred', label_mode='categorical', image_size=(256, 256), shuffle=True)
model = models.load_model('largeimage.keras', compile=True)
#model = buildModel()
#model.fit(trainds, epochs=10, validation_data=valds)
#model.save('largeimage.keras')

outputlayers = ['activation', 'activation_1', 'activation_2', 'activation_3', 'prediction']
lastlayer = ['activation_3']
#findContrast()
trans = pd.read_csv('session/trans.csv', index_col='index')
#generateTrans(model, valds, ['activation', 'activation_1', 'activation_2'])
#evalMatches(model, trans)

with open('session/franges.pkl', 'rb') as f:
    franges = pickle.load(f)

with open('session/patterns.pkl', 'rb') as f:
    cpats = pickle.load(f)


path = trans.loc[1, 'imagepath']
#sel = trans.loc[:3]['imagepath']
img = cv2.imread(path)
img = cv2.resize(img, (256, 256))
renderFeatureActivations(1, img, model, 'activation_1-54', franges)
print(1)
#newpats = []
#for c in cpats:
#    p = {'pattern': c['pattern'], 'targetsupport': c['incorrectsupport'], 'othersupport': c['correctsupport'],
#         'targetmatches': c['incorrectmatches'], 'othermatches': c['correctmatches'], 'supportdiff': c['supportdiff'], 'supportratio': c['supportratio']}
#    newpats.append(p)
#with open('session/patterns.pkl', 'wb') as f:
#    pickle.dump(newpats, f)
print(1)
