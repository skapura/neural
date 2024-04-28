import math

from tensorflow.keras import datasets, layers, models, backend
import tensorflow as tf
import ssl
import numpy as np
import random
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


def cutPoints(fmaps, iscorrect):
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


ssl._create_default_https_context = ssl._create_unverified_context

(orig_train_images, train_labels), (orig_test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = orig_train_images / 255.0, orig_test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

model = models.load_model('testmodel_softmax.keras')
print(model.summary())

debugmodel = makeDebugModel(model)
layernames = layerNames(debugmodel)
outs = debugmodel.predict(test_images)
ymaxpreds = [np.argmax(x) for x in outs[-1]]
labels = [l[0] for l in test_labels]
iscorrect = [ymaxpreds[i] == labels[i] for i in range(len(labels))]
labelinfo = list(zip(ymaxpreds, labels, iscorrect))
convouts = [o for o in outs if len(o.shape) == 4]   # get only CNN layers
colnames, activations = featureActivations(convouts, layernames)

cuts = [cutPoints(a, iscorrect) for a in activations]
discretize(activations[0], cuts[0])

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
