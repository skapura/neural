from tensorflow.keras import datasets, layers, models, callbacks
from keras.src.models import Functional
import tensorflow as tf
import numpy as np
import cv2
import ssl
import random
import math
import warnings
import spmf
import pandas as pd
from itertools import compress
import csv
from mlutil import makeDebugModel, makeLayerOutputModel, heatmap, overlayHeatmap, calcReceptiveField, layerIndex
import mlutil
from cifar10vgg import build_vgg16


def buildModel(train_images, train_labels, test_images, test_labels):
    img_input = layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, (3, 3))(img_input)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(10, name='prediction', activation='softmax')(x)
    model = Functional(img_input, x)
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
    model.save('testmodel_gap_func.keras')

    #model = models.Sequential()
    #model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    #model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    #model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    #model.add(layers.GlobalAveragePooling2D())
    ## model.add(layers.Flatten())
    ## model.add(layers.Dense(64, activation='relu'))
    #model.add(layers.Dense(10, activation='softmax'))
    #model.summary()
    #model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    #history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
    #model.save('testmodel_gap.keras')



def evalModel(model, images, labels):
    labels = np.squeeze(labels)
    outs = model.predict(images)
    o = outs[-1] if type(outs) is list else outs
    ymaxpreds = [np.argmax(x) for x in o]
    iscorrect = [ymaxpreds[i] == labels[i] for i in range(len(labels))]
    results = pd.DataFrame({'predicted': ymaxpreds, 'label': labels, 'iscorrect': iscorrect})
    results.index.name = 'index'
    return results


def filterLabels(labels, classnames, selected):
    selected = [classnames.index(s) for s in selected]
    selectedlabels = [i for i, l in enumerate(labels) if l in selected]
    return selectedlabels


def generateTrans(test_images, test_labels, class_names):
    model = models.load_model('testmodel_gap_func.keras', compile=True)
    test_labels = test_labels.squeeze()
    selectedindexes = filterLabels(test_labels, class_names, ['cat', 'dog'])
    selectedimages = [test_images[i] for i in selectedindexes]
    selectedlabels = [test_labels[i] for i in selectedindexes]

    outputlayers = ['activation', 'activation_1', 'prediction']
    lastlayer = 'activation_2'

    franges = mlutil.getLayerOutputRange(model, outputlayers, test_images)
    trans = mlutil.featuresToDataFrame(model, outputlayers, lastlayer, franges, selectedindexes, selectedimages,
                                       selectedlabels)
    t = trans[trans['predicted'].isin([di, ci])]
    t.to_csv('trans.csv')


def entropy(vals):
    if len(vals) == 0:
        return 0.0
    correctpct = len([v for v in vals if v[1] == 1]) / len(vals)
    incorrectpct = len([v for v in vals if v[1] == 0]) / len(vals)
    if correctpct == 0.0 or incorrectpct == 0.0:
        ent = 0.0
    else:
        ent = -1 * (correctpct * math.log2(correctpct) + incorrectpct * math.log2(incorrectpct))
    return ent


def cutPoints(vals, iscorrect):
    cutpoints = []
    tvals = np.unique(vals)
    testpoints = random.sample(range(len(tvals)), min(100, len(tvals)))
    mincut = -1
    mininfo = 99999
    for tp in testpoints:
        cutpoint = tvals[tp]
        d1 = np.asarray([(vals[i], iscorrect[i]) for i in range(len(iscorrect)) if vals[i] <= cutpoint])
        d2 = np.asarray([(vals[i], iscorrect[i]) for i in range(len(iscorrect)) if vals[i] > cutpoint])
        e1 = entropy(d1)
        e2 = entropy(d2)
        info = (len(d1) / len(vals)) * e1 + (len(d2) / len(vals)) * e2
        if info <= mininfo:
            mincut = cutpoint
            mininfo = info
    cutpoints.append(mincut)
    return cutpoints


def discretize(ds, cutpoints):
    bvals = []
    for index, row in ds.iterrows():
        binned = [index]
        vals = row.to_numpy()
        for i in range(len(vals[:-2])):
            cuts = cutpoints[i]
            found = False
            for ci in range(len(cuts)):
                if vals[i] <= cuts[ci]:
                    binned.append(ci)
                    found = True
                    break
            if not found:
                binned.append(len(cuts))
        binned.append(vals[-2])
        binned.append(vals[-1])
        bvals.append(binned)
    cols = ['index'] + ds.columns.values.tolist()
    bds = pd.DataFrame(bvals, columns=cols)
    bds.set_index('index', inplace=True)
    return bds


def binarize(df, cutpoints):

    # Get binary column names
    bincols = ['index']
    for ci in range(len(df.columns) - 2):
        for i in range(len(cutpoints[ci]) + 1):
            bincols.append(df.columns[ci] + '_' + str(i))
    bincols.append('iscorrect')

    # Binarize data
    brows = []
    for index, row in df.iterrows():
        rowbuf = [index]
        for ci in range(len(df.columns) - 2):
            v = row.iloc[ci]
            for i in range(len(cutpoints[ci]) + 1):
                rowbuf.append(v == i)
        rowbuf.append(row['predicted'] == row['label'])
        brows.append(rowbuf)

    bdf = pd.DataFrame(brows, columns=bincols)
    bdf.set_index('index', inplace=True)
    return bdf


def mineContrastPatterns(correct, incorrect):
    lookup = {}
    for i in range(len(incorrect.columns)):
        lookup[incorrect.columns[i]] = i
    with open('.trans.csv', 'w') as outfile:
        writer = csv.writer(outfile, delimiter=' ')
        for index, row in incorrect.iterrows():
            #itemset = [lookup[item] for item in list(compress(incorrect.columns, row)) if item.endswith('_1')]
            itemset = [lookup[item] for item in list(compress(incorrect.columns, row))]
            writer.writerow(itemset)

    fpclose = spmf.Spmf('FPClose', input_filename='.trans.csv', output_filename='.patterns.csv', arguments=[0.6, 5])
    fpclose.run()
    fpclose.parse_output()
    patlist = []
    for p in fpclose.patterns_:
        items = p[0].split()
        pat = frozenset([incorrect.columns[int(itm)] for itm in items[:-2]])
        sup = int(items[-1]) / float(len(incorrect))
        patlist.append({'pattern': pat, 'incorrectsupport': sup, 'correctmatches': [], 'incorrectmatches': []})

    correctsets = []
    for index, row in correct.iterrows():
        itemset = frozenset(compress(correct.columns, row))
        correctsets.append({'index': index, 'itemset': itemset})

    incorrectsets = []
    for index, row in incorrect.iterrows():
        itemset = frozenset(compress(incorrect.columns, row))
        incorrectsets.append({'index': index, 'itemset': itemset})

    i = 0
    for p in patlist:
        print(str(i) + '/' + str(len(patlist)))
        i += 1
        count = 0
        for c in correctsets:
            if p['pattern'].issubset(c['itemset']):
                p['correctmatches'].append(c['index'])
                count += 1
        for c in incorrectsets:
            if p['pattern'].issubset(c['itemset']):
                p['incorrectmatches'].append(c['index'])

        p['correctsupport'] = count / float(len(correctsets))
        p['supportdiff'] = p['incorrectsupport'] - p['correctsupport']
        p['supportratio'] = p['incorrectsupport'] / p['correctsupport']

    patlist.sort(key=lambda x: x['incorrectsupport'] - x['correctsupport'])
    for p in patlist:
        print(str(p['incorrectsupport'] - p['correctsupport']) + ', ' + str(p['correctsupport']) + ', ' + str(p['incorrectsupport']) + ', ' + str(p['pattern']))
    print('mine')


ssl._create_default_https_context = ssl._create_unverified_context
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)

(orig_train_images, train_labels), (orig_test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = orig_train_images / 255.0, orig_test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

#buildModel(train_images, train_labels, test_images, test_labels)
model = models.load_model('testmodel_gap_func.keras', compile=True)
debugmodel = makeDebugModel(model)
#model = build_vgg16()

#results = evalModel(model, test_images, test_labels)
di = class_names.index('dog')
ci = class_names.index('cat')
#results = results[results['label'].isin([di, ci])]
#results = results[results['predicted'].isin([di, ci])]
#correct = results[results['iscorrect']]
#incorrect = results[~results['iscorrect']]

#index = incorrect.index.values[0]
#index = results.index.values[0]

#generateTrans(test_images, test_labels, class_names)


trans = pd.read_csv('trans.csv', index_col='index')
droplist = []
for c in trans.columns[:-2]:
    mx = trans[c].max()
    if mx == 0.0:
        droplist.append(c)
trans.drop(droplist, axis=1, inplace=True)


iscorrect = (trans['predicted'] == trans['label']).to_numpy()
cutpoints = []
for c in trans.columns[:-2]:
    vals = trans[c].to_numpy()
    cuts = cutPoints(vals, iscorrect)
    cutpoints.append(cuts)
bds = discretize(trans, cutpoints)
bdf = binarize(bds, cutpoints)
correct = bdf[bdf['iscorrect']]
incorrect = bdf[~bdf['iscorrect']]
correct.drop('iscorrect', axis=1, inplace=True)
incorrect.drop('iscorrect', axis=1, inplace=True)
mineContrastPatterns(correct, incorrect)
print(1)


correct = trans[trans['predicted'] == trans['label']]
incorrect = trans[trans['predicted'] != trans['label']]


print(1)
convinfo = [{'name': l.name, 'kernel': l.kernel_size if 'conv2d' in l.name else l.pool_size, 'stride': l.strides} for l in debugmodel.layers if 'conv2d' in l.name or 'max_pooling2d' in l.name]
start, end = calcReceptiveField(0, 0, [convinfo[0]])
#outs = debugmodel.predict(np.asarray([test_images[index]]))

for index, row in correct.iterrows():
    heatmap, himg = heatmap(np.asarray([test_images[index]]), model, 'activation_2', row['predicted'])
    heatimage = overlayHeatmap(np.asarray([orig_test_images[index]]), heatmap)
    hu = np.unique(heatmap)
    hi = np.unique(heatimage)
    print(1)
    #cv2.imwrite('heat/correct/' + str(index).zfill(5) + '.png', heatimage)
for index, row in incorrect.iterrows():
    heatmap = heatmap(np.asarray([test_images[index]]), model, 'activation_2', row['predicted'])
    heatimage = overlayHeatmap(np.asarray([orig_test_images[index]]), heatmap)
    cv2.imwrite('heat/incorrect/' + str(index).zfill(5) + '.png', heatimage)
print(1)
