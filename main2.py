import math

from tensorflow.keras import datasets, layers, models, backend
import tensorflow as tf
import ssl
import numpy as np
import pandas as pd
import random
from itertools import compress
import warnings
import spmf
import csv
from mlxtend.frequent_patterns import fpgrowth, fpmax
from mlutil import makeDebugModel


def featureMapActivation(fmap):
    return np.max(fmap)


def featureActivations(outs, layernames):

    # Group fmap activations for all inputs
    colnames = []
    activations = []
    for oi in range(len(outs)):
        layer = outs[oi]
        numimages = outs[oi].shape[0]
        numfmaps = outs[oi].shape[-1]
        for fi in range(numfmaps):
            colnames.append(layernames[oi] + '_' + str(fi))
            fmaps = [featureMapActivation(layer[i, :, :, fi]) for i in range(numimages)]
            activations.append(fmaps)

    return colnames, activations


def layerNames(model):
    layernames = [l.name for l in model.layers if '2d' in l.name]
    return layernames


def entropy(ds):
    if len(ds) == 0:
        return 0.0
    correctpct = len([v for v in ds if v[1]]) / len(ds)
    incorrectpct = len([v for v in ds if not v[1]]) / len(ds)
    if correctpct == 0.0 or incorrectpct == 0.0:
        ent = 0.0
    else:
        ent = -1 * (correctpct * math.log2(correctpct) + incorrectpct * math.log2(incorrectpct))
    return ent


def cutPoints2(fmaps, iscorrect):
    cutpoints = []
    vals = list(zip(fmaps, iscorrect))
    tvals = np.unique(fmaps)
    testpoints = random.sample(range(len(tvals)), min(100, len(tvals)))
    mincut = -1
    mininfo = 99999
    d1size = 0
    d2size = 0
    for tp in testpoints:
        cutpoint = tvals[tp]
        d1 = [v for v in vals if v[0] <= cutpoint]
        d2 = [v for v in vals if v[0] > cutpoint]
        e1 = entropy(d1)
        e2 = entropy(d2)
        info = (len(d1) / len(vals)) * e1 + (len(d2) / len(vals)) * e2
        if info <= mininfo:
            mincut = cutpoint
            mininfo = info
            d1size = len(d1)
            d2size = len(d2)
    cutpoints.append(mincut)
    print('best info:' + str(mininfo) + ', cutpoint:' + str(mincut) + ', d1:' + str(d1size) + ', d2:' + str(d2size))
    return cutpoints


def cutPoints(fmaps, iscorrect):
    cutpoints = []
    vals = list(zip(fmaps, iscorrect))
    tvals = np.unique(fmaps)
    testpoints = random.sample(range(len(tvals)), min(100, len(tvals)))
    samplevals = [vals[i] for i in random.sample(range(len(vals)), min(1000, len(vals)))]
    mincut = -1
    mininfo = 99999
    d1size = 0
    d2size = 0
    for tp in testpoints:
        cutpoint = tvals[tp]
        d1 = [v for v in samplevals if v[0] <= cutpoint]
        d2 = [v for v in samplevals if v[0] > cutpoint]
        e1 = entropy(d1)
        e2 = entropy(d2)
        info = (len(d1) / len(samplevals)) * e1 + (len(d2) / len(samplevals)) * e2
        if info <= mininfo:
            mincut = cutpoint
            mininfo = info
            d1size = len(d1)
            d2size = len(d2)
    cutpoints.append(mincut)
    print('best info:' + str(mininfo) + ', cutpoint:' + str(mincut) + ', d1:' + str(d1size) + ', d2:' + str(d2size))
    return cutpoints


def discretize(fmaps, cutpoints):
    fbinned = []
    for f in fmaps:
        found = False
        for ci in range(len(cutpoints)):
            if f <= cutpoints[ci]:
                fbinned.append(ci)
                found = True
                break
        if not found:
            fbinned.append(len(cutpoints))
    return fbinned


def binarize(df):
    cuts = [[1] for i in range(len(df.columns) - 1)]

    # Get binary column names
    bincols = []
    for ci in range(len(df.columns) - 1):
        for i in range(len(cuts[ci]) + 1):
            bincols.append(df.columns[ci] + '_' + str(i))
    bincols.append('iscorrect')

    # Binarize data
    brows = []
    for index, row in df.iterrows():
        rowbuf = []
        for ci in range(len(df.columns) - 1):
            v = row.iloc[ci]
            for i in range(len(cuts[ci]) + 1):
                rowbuf.append(v == i)
        rowbuf.append(row['iscorrect'] == 1)
        brows.append(rowbuf)

    bdf = pd.DataFrame(brows, columns=bincols)
    bdf.index.name = 'index'
    return bdf


def evalModel(debugmodel):
    layernames = layerNames(debugmodel)
    outs = debugmodel.predict(test_images)
    ymaxpreds = [np.argmax(x) for x in outs[-1]]
    labels = [l[0] for l in test_labels]
    iscorrect = [ymaxpreds[i] == labels[i] for i in range(len(labels))]
    labelinfo = list(zip(ymaxpreds, labels, iscorrect))
    convouts = [o for o in outs if len(o.shape) == 4]  # get only CNN layers
    colnames, activations = featureActivations(convouts, layernames)

    cuts = [cutPoints(a, iscorrect) for a in activations]
    binned = [discretize(activations[i], cuts[i]) for i in range(len(activations))]
    d = {}
    for i in range(len(colnames)):
        d[colnames[i]] = binned[i]
    d['label'] = labels
    d['predicted'] = ymaxpreds
    d['iscorrect'] = [1 if c else 0 for c in iscorrect]
    df = pd.DataFrame(d)
    df.index.name = 'index'
    df.to_csv('test.csv')


def mineContrastPatterns(correct, incorrect):
    print('mine')

    correctsets = []
    for index, row in correct.iterrows():
        itemset = frozenset(compress(correct.columns, row))
        correctsets.append(itemset)

    dups = []
    for c in incorrect.columns:
        vals = incorrect[c].to_numpy()
        if (vals[0] == vals).all():
            dups.append(c)
    incorrect.drop(dups, axis=1, inplace=True)

    lookup = {}
    for i in range(len(incorrect.columns)):
        lookup[incorrect.columns[i]] = i
    with open('.trans.csv', 'w') as outfile:
        writer = csv.writer(outfile, delimiter=' ')
        for index, row in incorrect.iterrows():
            itemset = [lookup[item] for item in list(compress(incorrect.columns, row))]
            writer.writerow(itemset)

    fpclose = spmf.Spmf('FPClose', input_filename='.trans.csv', output_filename='.patterns.csv', arguments=[0.8, 5])
    fpclose.run()
    fpclose.parse_output()
    patlist = []
    for p in fpclose.patterns_:
        items = p[0].split()
        pat = frozenset([incorrect.columns[int(itm)] for itm in items[:-2]])
        sup = int(items[-1]) / float(len(incorrect))
        patlist.append({'pattern': pat, 'incorrectsupport': sup})
    print(len(patlist))

    i = 0
    for p in patlist:
        print(i)
        i += 1
        count = 0
        for c in correctsets:
            if p['pattern'].issubset(c):
                count += 1
        p['correctsupport'] = count / float(len(correctsets))
        p['supportdiff'] = p['incorrectsupport'] - p['correctsupport']
        p['supportratio'] = p['incorrectsupport'] / p['correctsupport']

    patlist.sort(key=lambda x: x['incorrectsupport'] - x['correctsupport'])
    for p in patlist:
        #if p['supportdiff'] > 0.05:
        print(p)
    print(1)


ssl._create_default_https_context = ssl._create_unverified_context
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)

(orig_train_images, train_labels), (orig_test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = orig_train_images / 255.0, orig_test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

model = models.load_model('testmodel_softmax.keras')
print(model.summary())

debugmodel = makeDebugModel(model)
#evalModel(debugmodel)
df = pd.read_csv('test.csv', index_col=0)
di = class_names.index('dog')
ci = class_names.index('cat')
sel = df[df['label'].isin([di, ci])]
sel = df[df['predicted'].isin([di, ci])]
sel.drop(['label', 'predicted'], axis=1, inplace=True)
#cols = sel.columns[32:-1]   #32-63
cols = [sel.columns[i] for i in range(len(sel.columns)) if 'conv2d_1_' in sel.columns[i] or 'conv2d_2_' in sel.columns[i]]
#cols = sel.columns[:32]
#cols = cols[:30]
cols = [c for c in sel.columns if c not in cols and c != 'iscorrect']
sel.drop(cols, axis=1, inplace=True)
bds = binarize(sel)
correct = bds[bds['iscorrect']]
incorrect = bds[~bds['iscorrect']]
correct.drop('iscorrect', axis=1, inplace=True)
incorrect.drop('iscorrect', axis=1, inplace=True)
mineContrastPatterns(correct, incorrect)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

model.save('testmodel_softmax.keras')
