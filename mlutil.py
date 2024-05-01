import numpy as np
from tensorflow.keras import models


def makeDebugModel(model, onlyconv=False):
    if onlyconv:
        outputs = [layer.output for layer in model.layers if type(layer) is keras.layers.Conv2D]
    else:
        outputs = [layer.output for layer in model.layers]
    debugmodel = models.Model(inputs=model.inputs, outputs=outputs)
    return debugmodel


def evalModel(X, y, model):
    ypreds = model.predict(X)
    #correct = list()
    #incorrect = list()
    #for i in range(len(y)):
    #    yp = ypreds[i]
    #    maxclass = np.argmax(yp)
    #    correctclass = y[i][0]
    #    if y[i][0] == maxclass:
    #        correct.append(i)
    #    else:
    #        incorrect.append(i)
    #    #print(1)
    ymaxpreds = [np.argmax(x) for x in ypreds]
    eval = [x[0] == x[1] for x in zip(map(lambda x: x[0], y), ymaxpreds)]
    correct = [i for i, elem in enumerate(eval) if elem]
    incorrect = [i for i, elem in enumerate(eval) if not elem]
    wronganswers = [{'index': i, 'correct': y[i][0], 'predicted': ymaxpreds[i]} for i in incorrect]
    return correct, incorrect, wronganswers


def calcReceptiveField(x, y, layers):
    startx = x
    endx = x
    starty = y
    endy = y
    for l in reversed(layers):
        startx = startx * l['stride'][1]
        endx = endx * l['stride'][1] + l['kernel'][1] - 1
        starty = starty * l['stride'][0]
        endy = endy * l['stride'][0] + l['kernel'][0] - 1
    return (startx, endx), (starty, endy)

def calcReceptiveField2(u, v, layers):
    for l in reversed(layers):
        print(l)
        u = u * l['stride'][1]
        v = v * l['stride'][1] + l['kernel'][1] - 1
    return u, v
