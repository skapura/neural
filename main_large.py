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


def buildModel2():
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
    x = layers.Conv2D(64, (7, 7))(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(3, name='prediction', activation='softmax')(x)
    model = Functional(inputs, x)
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    return model


def save(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def pipeline():
    trainds = mlutil.load_from_directory('images_large/train', labels='inferred', label_mode='categorical',
                                         image_size=(256, 256), shuffle=True)
    valds = mlutil.load_from_directory('images_large/val', labels='inferred', label_mode='categorical',
                                       image_size=(256, 256), shuffle=True)
    #model = buildModel()
    #tensorboard_callback = keras.callbacks.TensorBoard(log_dir='tlogs')
    #model.fit(trainds, epochs=2, validation_data=valds, callbacks=[tensorboard_callback])


    #model.save('largeimage_grass.keras')
    model = models.load_model('largeimage2.keras', compile=True)
    #w = model.layers[13].get_weights()
    #w[0][13][:] = np.zeros(w[0][13].shape)
    #w[0][22][:] = np.zeros(w[0][22].shape)
    #w[0][27][:] = np.zeros(w[0][27].shape)
    #w[0][29][:] = np.zeros(w[0][29].shape)
    #w[0][42][:] = np.zeros(w[0][42].shape)
    #w[0][49][:] = np.zeros(w[0][49].shape)
    #for i in range(63):
    #    w[0][i][:] = np.zeros(w[0][i].shape)
    #model.layers[13].set_weights(w)

    model.fit(trainds, epochs=1, validation_data=valds)

    #model.layers[13].trainable_weights
    #outputlayers = ['activation', 'activation_1', 'activation_2', 'activation_3', 'prediction']
    outputlayers = ['activation_3', 'prediction']
    lastlayer = ['activation_3']

    test_loss, test_acc = model.evaluate(valds)


    #franges = None
    #for imagebatch, labelbatch in valds:
    #    franges = mlutil.getLayerOutputRange(model, outputlayers, imagebatch, franges)
    #save('session/franges_grass.pkl', franges)
    franges = load('session/franges2.pkl')

    #trans = mlutil.featuresToDataFrame(model, outputlayers, valds)
    #trans.to_csv('session/trans_grass.csv')
    trans = pd.read_csv('session/trans2.csv', index_col='index')

    #bdf, _ = pats.makeBinaryDataset(trans)
    #bdf.to_csv('session/bin_grass.csv')
    bdf = pd.read_csv('session/bin2.csv', index_col='index')

    #correct, incorrect = pats.divideByPrediction(bdf, label=2, predicted=[1], activatedonly=True)
    #cpats = pats.mineContrastPatterns(incorrect, correct, 0.7, 1.0)
    #save('session/pats_grass.pkl', cpats)
    cpats = load('session/pats2.pkl')
    #pats.pats2csv(cpats, 'session/patterns_grass.csv')

    f = frozenset(['activation_3-' + str(idx) + '_1' for idx in range(64)])

    #info = trans.loc[indexlist, ['predicted', 'label', 'imagepath']]
    #plot.renderPattern(0, cpats[0], model, imgpath)
    one = trans[trans['label'] == 0][:10].index.values
    two = trans[trans['label'] == 1][:10].index.values
    three = trans[trans['label'] == 2][:10].index.values
    b = np.concatenate([one, two, three])
    plot.renderPattern2(0, f, model, trans, b)
    #plot.renderPattern2(0, cpats[0]['pattern'], model, trans, b) #cpats[0]['targetmatches'][:20])

    images = trans.loc[trans['label'] == 1, 'imagepath'][:20]
    plot.renderFeatureMaps(images, model, ['activation_3', 'prediction'])

    print(1)


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
    #franges = None
    #for imagebatch, labelbatch in images:
    #    franges = mlutil.getLayerOutputRange(model, outputlayers, imagebatch, franges)
    #with open('session/frangesheat.pkl', 'wb') as f:
    #    pickle.dump(franges, f)
    #with open('session/franges.pkl', 'rb') as f:
    #    franges = pickle.load(f)

    # Get model outputs as transaction dataset
    #trans = mlutil.featuresToDataFrame(model, outputlayers, images)
    #trans.to_csv('session/transheat.csv')

    trans = pd.read_csv('session/transheat.csv', index_col='index')

    # Drop columns that are all close to zero
    droplist = []
    for c in trans.columns[:-2]:
        mx = trans[c].max()
        if mx <= 0.05:
            droplist.append(c)
    trans.drop(droplist, axis=1, inplace=True)
    #selcols = [c for c in trans.columns if c.startswith('activation_3')] + ['predicted', 'label', 'imagepath']
    #trans = trans[selcols]

    #trans = trans.loc[~(trans['label'] == trans['predicted'])]
    #trans = trans.loc[(trans['label'] == 2) & (trans['predicted'].isin([1]))]

    labels = trans[['predicted', 'label']]
    iscorrect = (trans['predicted'] == trans['label']).to_numpy()

    trans.drop(['predicted', 'label', 'imagepath'], axis=1, inplace=True)

    # Discretize dataset
    #cutpoints = []
    #infolist = []
    #for c in trans.columns:
    #    print(c)
    #    vals = trans[c].to_numpy()
    #    cuts, info = pats.cutPoints(vals, iscorrect)
    #    cutpoints.append(cuts)
    #    infolist.append(info)
    #bds = pats.discretize(trans, cutpoints)
    #bdf = pats.binarize(bds, cutpoints)
    #bdf.to_csv('session/binheat.csv')
    bdf = pd.read_csv('session/binheat.csv', index_col='index')
    bdf = bdf[[c for c in bdf.columns if c.endswith('_1')]]

    bdf['predicted'] = labels['predicted']
    bdf['label'] = labels['label']
    #bdf['iscorrect'] = iscorrect
    correct = bdf.loc[bdf['predicted'] == bdf['label']]
    incorrect = bdf.loc[~(bdf['predicted'] == bdf['label'])]
    #correct = correct.loc[correct['label'].isin([1, 2])]
    #incorrect = incorrect.loc[(incorrect['label'].isin([1, 2])) & (incorrect['predicted'].isin([1, 2]))]
    correct.drop(['predicted', 'label'], axis=1, inplace=True)
    incorrect.drop(['predicted', 'label'], axis=1, inplace=True)

    #correct = bdf[bdf['iscorrect']]
    #incorrect = bdf[~bdf['iscorrect']]
    #correct = correct[correct.columns[:-1]]
    #incorrect = incorrect[incorrect.columns[:-1]]
    cpats = pats.mineContrastPatterns(incorrect, correct, 0.7, 1.0)
    with open('session/patternsheat.pkl', 'wb') as f:
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

def featureMapEval(trans):

    # Drop columns that are all close to zero
    #droplist = []
    #for c in trans.columns[:-2]:
    #    mx = trans[c].max()
    #    if mx <= 0.05:
    #        droplist.append(c)
    #trans.drop(droplist, axis=1, inplace=True)

    iscorrect = (trans['predicted'] == trans['label']).to_numpy()
    #trans.drop(['predicted', 'label', 'imagepath'], axis=1, inplace=True)

    # Discretize dataset
    #cutpoints = []
    #infolist = []
    #for c in trans.columns:
    #    print(c)
    #    vals = trans[c].to_numpy()
    #    cuts, info = pats.cutPoints(vals, iscorrect)
    #    cutpoints.append(cuts)
    #    infolist.append(info)
    #bds = pats.discretize(trans, cutpoints)
    #bdf = pats.binarize(bds, cutpoints)
    #bdf.to_csv('session/temp.csv')
    bdf = pd.read_csv('session/temp.csv', index_col='index')
    bdf['predicted'] = trans['predicted']
    bdf['label'] = trans['label']
    bdf['iscorrect'] = iscorrect

    # Get activation counts by class
    predcounts = list()
    labelcounts = list()
    correctcounts = list()
    for c in bdf.columns[:-3]:
        sel = bdf[bdf[c]]

        # predicted counts
        cnt = sel['predicted'].value_counts() / len(bdf)
        cnt.name = c
        predcounts.append(cnt)

        # label counts
        cnt = sel['label'].value_counts() / len(bdf)
        cnt.name = c
        labelcounts.append(cnt)

        # correct counts
        cnt = sel['iscorrect'].value_counts() / len(bdf)
        cnt.name = c
        correctcounts.append(cnt)

    preddf = pd.concat(predcounts, axis=1, sort=False)
    preddf.sort_index(inplace=True)
    preddf.index = ['pred_' + str(i) for i in preddf.index.tolist()]

    labeldf = pd.concat(labelcounts, axis=1, sort=False)
    labeldf.sort_index(inplace=True)
    labeldf.index = ['label_' + str(i) for i in labeldf.index.tolist()]

    correctdf = pd.concat(correctcounts, axis=1, sort=False)
    correctdf.sort_index(inplace=True)

    counts = pd.concat([preddf, labeldf, correctdf])

    counts.to_csv('session/counts.csv')
    print(counts)
    print(1)


pd.options.mode.chained_assignment = None
pipeline()
#trainds = mlutil.load_from_directory('images_large/train', labels='inferred', label_mode='categorical', image_size=(256, 256), shuffle=True)
#valds = mlutil.load_from_directory('images_large/val', labels='inferred', label_mode='categorical', image_size=(256, 256), shuffle=True)
#model = models.load_model('largeimage.keras', compile=True)
#model = buildModel()
#model.fit(trainds, epochs=10, validation_data=valds)
#model.save('largeimage.keras')

outputlayers = ['activation', 'activation_1', 'activation_2', 'activation_3', 'prediction']
lastlayer = ['activation_3']
#findContrast()
#trans = pd.read_csv('session/trans.csv', index_col='index')
#featureMapEval(trans)
#generateTrans(model, valds, ['activation', 'activation_1', 'activation_2', 'activation_3', 'prediction'])
#evalMatches(model, trans)

with open('session/frangesheat.pkl', 'rb') as f:
    franges = pickle.load(f)

#with open('session/patternsheat.pkl', 'rb') as f:
#    cpats = pickle.load(f)

#conf = mlutil.confusionMatrix(model, trainds)

trans = pd.read_csv('session/transheat.csv', index_col='index')
#plot.renderPattern(31, cpats[31], model, trans, cpats[31]['targetmatches'][:20])
#bdf, _ = pats.makeBinaryDataset(trans)

#bdf = bdf.loc[(bdf['label'] == 2) & (bdf['predicted'].isin([1, 2]))]
#correct = bdf.loc[bdf['predicted'] == bdf['label']]
#incorrect = bdf.loc[~(bdf['predicted'] == bdf['label'])]
#correct = correct[[c for c in correct.columns if c.endswith('_1')]]
#incorrect = incorrect[[c for c in incorrect.columns if c.endswith('_1')]]
#cpats = pats.mineContrastPatterns(incorrect, correct, 0.7, 1.0)
#with open('session/patternsgolf.pkl', 'wb') as f:
#    pickle.dump(cpats, f)
#bdf.to_csv('session/bintemp.csv')
bdf = pd.read_csv('session/bintemp.csv', index_col='index')
#correct, incorrect = pats.divideByPrediction(bdf, 2, [1], True)
#cpats = pats.mineContrastPatterns(incorrect, correct, 0.7, 1.0)
#with open('session/patterns1.pkl', 'wb') as f:
#    pickle.dump(cpats, f)
cpats = load('session/patterns1.pkl')
plot.renderPattern(0, cpats[0], model, trans, cpats[0]['targetmatches'][:20])

classes = bdf['label'].unique()
classes.sort()
csums = []
for cls in classes:
    ctrans = bdf.loc[bdf['label'] == cls]
    sums = ['L' + str(cls)]
    for c in ctrans.columns[:-3]:
        count = ctrans[c].values.sum()
        sums.append(count)
    csums.append(sums)
for lc in classes:
    for pc in classes:
        ctrans = bdf.loc[(bdf['label'] == lc) & (bdf['predicted'] == pc)]
        sums = ['L' + str(lc) + 'P' + str(pc)]
        for c in ctrans.columns[:-3]:
            count = ctrans[c].values.sum()
            sums.append(count)
        csums.append(sums)

ctrans = bdf.loc[bdf['label'] == bdf['predicted']]
sums = ['correct']
for c in ctrans.columns[:-3]:
    count = ctrans[c].values.sum()
    sums.append(count)
csums.append(sums)
ctrans = bdf.loc[~(bdf['label'] == bdf['predicted'])]
sums = ['incorrect']
for c in ctrans.columns[:-3]:
    count = ctrans[c].values.sum()
    sums.append(count)
csums.append(sums)

def old():
    df = pd.DataFrame(csums, columns=['class'] + list(bdf.columns[:-3]))
    df.to_csv('session/sums.csv')


    with open('session/patternsgolf.pkl', 'rb') as f:
        cpats = pickle.load(f)

    images = trans.loc[trans['label'] == 1, 'imagepath'][:20]
    plot.renderFeatureMaps(images, model, ['activation', 'prediction'])

    patidx = 0
    plot.renderPattern(patidx, cpats[patidx], model, trans, cpats[patidx]['othermatches'][:20], franges)
    plot.renderHeatmaps(trans.loc[cpats[patidx]['targetmatches'][:20], ['imagepath', 'predicted']], model, 'activation_3', 'session/heatmaps')

    trans = trans.loc[(trans['label'] == 2) & (trans['predicted'].isin([1, 2]))]
    correct = trans.loc[trans['predicted'] == trans['label'], ['imagepath', 'predicted']]
    incorrect = trans.loc[~(trans['predicted'] == trans['label']), ['imagepath', 'predicted']]
    plot.renderHeatmaps(correct, model, 'activation_3', 'session/correct')
    plot.renderHeatmaps(incorrect, model, 'activation_3', 'session/incorrect')

    plot.renderHeatmaps(trans.loc[cpats[31]['othermatches'][:20], ['imagepath', 'predicted']], model, 'activation_3')

    images = set([246, 254])
    matches = []
    for p in cpats:
        if images.issubset(p['targetmatches']):
            matches.append(p)

    pats.pats2csv(cpats, 'session/patterns.csv')

    index = 1
    feat = 'activation_1-54'
    path = trans.loc[index, 'imagepath']
    t = feat.split('-')
    layername = t[0]
    fmapindex = int(t[1])
    img = cv2.imread(path)
    img = cv2.resize(img, (256, 256))
    plot.renderPattern(index, cpats[index], model, trans, cpats[index]['targetmatches'][:4], franges)
    #plot.renderHeatmaps(trans.loc[[2, 7, 17, 19], ['imagepath', 'predicted']], model, 'activation_3', False)


    combined, mask, _ = renderFeatureActivation(img, model, feat, franges)
    basepath = 'session/' + str(index).zfill(5) + '/features/feature-' + str(index).zfill(5) + '-' + layername + '-' + str(fmapindex).zfill(4) + '-'
    cv2.imwrite(basepath + 'activation.png', combined)
    cv2.imwrite(basepath + 'mask.png', mask)
    print(1)
    #newpats = []
    #for c in cpats:
    #    p = {'pattern': c['pattern'], 'targetsupport': c['incorrectsupport'], 'othersupport': c['correctsupport'],
    #         'targetmatches': c['incorrectmatches'], 'othermatches': c['correctmatches'], 'supportdiff': c['supportdiff'], 'supportratio': c['supportratio']}
    #    newpats.append(p)
    #with open('session/patterns.pkl', 'wb') as f:
    #    pickle.dump(newpats, f)
    print(1)
