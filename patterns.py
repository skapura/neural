from skmine.itemsets import SLIM, LCM
from skmine.itemsets.lcm import LCMMax
from skmine.emerging import MBDLLBorder
import numpy as np
import pandas as pd
import math


def findMatches(df, pattern):
    matches = []
    for index, row in df.iterrows():
        if pattern.issubset(set(row.values)):
            matches.append(index)
    return matches


def pat2columns(pattern, columns):
    tpat = [columns[math.floor(item / 100) - 1] for item in pattern]
    return tpat


def isActivated(fmap, threshold):
    return len(fmap[fmap >= threshold]) > 0


def generateDataset(model, images, labels):
    ypreds = model.predict(images)
    ymaxpreds = [np.argmax(x) for x in ypreds[-1]]
    eval = [x[0] == x[1] for x in zip(map(lambda x: x[0], labels), ymaxpreds)]
    layers = [model.layers[i + 1].name for i in range(len(ypreds)) if '2d' in model.layers[i + 1].name]
    thresholds = [np.max(ypreds[i]) * 0.25 for i in range(len(layers))]

    cols = ['index']
    for i in range(len(layers)):
        cols += [layers[i] + '_' + str(j) for j in range(ypreds[i].shape[3])]
        break

    buffer = []
    for i in range(len(images)):
        layeractivations = [i]
        for l in range(len(layers)):
            outs = ypreds[l][i]
            activations = []
            for f in range(outs.shape[2]):
                fmap = outs[:, :, f]
                activations.append(int(isActivated(fmap, thresholds[l])))
            layeractivations += activations
            break
        buffer.append(layeractivations + [ymaxpreds[i], labels[i][0], int(eval[i])])

    trans = pd.DataFrame(buffer, columns=cols + ['predicted', 'label', 'iscorrect'])
    trans.set_index('index', inplace=True)

    #trans.to_csv('test.csv')
    return trans


def transformDataset(trans):
    buffer = []
    for index, row in trans.iterrows():
        newrow = [index]
        newrow += [(c + 1) * 100 + row.iloc[c] for c in range(0, len(trans.columns) - 1)]
        newrow.append(row['iscorrect'])
        buffer.append(newrow)
    cols = [trans.index.name] + trans.columns.values.tolist()
    df = pd.DataFrame(buffer, columns=cols)
    df.set_index(trans.index.name, inplace=True)
    return df


def mineContrastPats(correct, incorrect):
    miner = LCM()
    if 'iscorrect' in correct.columns:
        correct = correct.drop('iscorrect', axis=1)
    correct = correct.drop(correct.columns[[i for i in range(23)]], axis=1)
    buffer = correct.values.tolist()
    correctpats = miner.fit_transform(buffer, return_tids=False)
    #print(correctpats)

    if 'iscorrect' in incorrect.columns:
        incorrect = incorrect.drop('iscorrect', axis=1)
    incorrect = incorrect.drop(incorrect.columns[[i for i in range(23)]], axis=1)
    buffer = incorrect.values.tolist()
    incorrectpats = miner.fit_transform(buffer, return_tids=False)

    contrastpats = []
    for index, row in incorrectpats.iterrows():
        pat = row['itemset']
        ismatch = False
        for ci, cr in correctpats.iterrows():
            if cr['itemset'] == pat:
                ismatch = True
                break
        if not ismatch:
            contrastpats.append({'pattern': set(row['itemset']), 'support': row['support']})

    return contrastpats




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
#print(patterns)

