from tensorflow.keras import datasets, layers, models, backend
import tensorflow as tf
import ssl
import numpy as np
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
convouts = [o for o in outs if len(o.shape) == 4]   # get only CNN layers
colnames, activations = featureActivations(convouts, layernames)

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
