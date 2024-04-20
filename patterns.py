from skmine.itemsets import SLIM, LCM
from skmine.itemsets.lcm import LCMMax
from skmine.emerging import MBDLLBorder
import numpy as np
import pandas as pd



def isActivated(fmap, threshold):
    return len(fmap[fmap >= threshold]) > 0


def generateDataset(model, images, labels):
    ypreds = model.predict(images)
    ymaxpreds = [np.argmax(x) for x in ypreds[-1]]
    eval = [x[0] == x[1] for x in zip(map(lambda x: x[0], labels), ymaxpreds)]
    layers = [model.layers[i + 1].name for i in range(len(ypreds)) if '2d' in model.layers[i + 1].name]
    thresholds = [np.max(ypreds[i]) * 0.25 for i in range(len(layers))]

    cols = []
    for i in range(len(layers)):
        cols += [layers[i] + '_' + str(j) for j in range(ypreds[i].shape[3])]
        break

    trans = pd.DataFrame(columns=cols + ['predicted', 'label', 'incorrect'])
    for i in range(len(images)):
        layeractivations = []
        for l in range(len(layers)):
            outs = ypreds[l][i]
            activations = []
            for f in range(outs.shape[2]):
                fmap = outs[:, :, f]
                activations.append(int(isActivated(fmap, thresholds[l])))
            layeractivations += activations
            break
        trans.loc[len(trans.index)] = layeractivations + [ymaxpreds[i], labels[i][0], int(eval[i])]

    #trans.to_csv('test.csv')
    return trans






#D = [['bananas', 'milk'], ['milk', 'bananas', 'cookies'], ['cookies', 'butter', 'tea']]
D = [[(0, 1), (2, 2)], [(2, 2), (0, 1), (3, 4)], [(3, 4), (0, 0), (0, 2)]]
#results = SLIM().fit(D).transform(D, singletons=True, return_tids=True)
#print(results)

D = [['bananas', 'milk'], ['milk', 'bananas', 'cookies'], ['cookies', 'butter', 'tea', 'milk']]
#D = [[(0, 1), (2, 2)], [(2, 2), (0, 1), (3, 4)], [(3, 4), (0, 0), (0, 2)]]
#results = SLIM().fit(D).transform(D, singletons=True, return_tids=True)
#print(results)

#results = LCMMax(min_supp=2).fit(D).transform(D, return_tids=True)
#print(results)

#D = pd.Series(
#    [
#        ["banana", "chocolate"],
#        ["sirup", "tea"],
#        ["chocolate", "banana"],
#        ["chocolate", "milk", "banana"],
#    ]
#)

#y = pd.Series(
#    [
#        "food",
#        "drink",
#        "food",
#        "drink",
#    ],
#    dtype="category",
#)

D = pd.Series(
    [
        [2, 3, 1],
        [2, 3, 1],
        [2, 3, 1],
        [5, 6, 4],
        [2, 3, 1],
        [6, 5, 8],
    ]
)

y = pd.Series(
    [
        "p",
        "p",
        "p",
        "n",
        "n",
        "n"
    ],
    dtype="category",
)



ep = MBDLLBorder(min_supp=.6, min_growth_rate=1.6)
patterns = ep.fit_discover(D, y, min_size=1)
print(patterns)

