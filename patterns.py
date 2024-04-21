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

    #TODO: create as list first, then copy into dataframe
    trans = pd.DataFrame(columns=cols + ['predicted', 'label', 'iscorrect'])
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


def transformDataset(trans):
    rows = []
    for index, row in trans.iterrows():
        newrow = [(c + 1) * 100 + row.iloc[c] for c in range(0, len(trans.columns) - 1)]
        newrow.append(row['iscorrect'])
        rows.append(newrow)
    df = pd.DataFrame(rows, columns=trans.columns)
    return df


def mineContrastPats(correct, incorrect):
    miner = LCM()
    if 'iscorrect' in correct.columns:
        correct = correct.drop('iscorrect', axis=1)
    correct = correct.drop(correct.columns[[i for i in range(25)]], axis=1)
    buffer = correct.values.tolist()
    correctpats = miner.fit_transform(buffer, return_tids=False)
    print(correctpats)

    if 'iscorrect' in incorrect.columns:
        incorrect = incorrect.drop('iscorrect', axis=1)
    incorrect = incorrect.drop(incorrect.columns[[i for i in range(25)]], axis=1)
    buffer = incorrect.values.tolist()
    incorrectpats = miner.fit_transform(buffer, return_tids=False)
    print(incorrectpats)

    print(1)




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

