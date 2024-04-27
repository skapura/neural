from skmine.itemsets import SLIM, LCM
from skmine.itemsets.lcm import LCMMax
from skmine.emerging import MBDLLBorder
import numpy as np
import pandas as pd
import math
from mlxtend.frequent_patterns import fpgrowth, fpmax
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def findMatches2(df, pattern):
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


def generateDataset2(model, images, labels):
    ypreds = model.predict(images)
    ymaxpreds = [np.argmax(x) for x in ypreds[-1]]
    eval = [x[0] == x[1] for x in zip(map(lambda x: x[0], labels), ymaxpreds)]
    layers = [model.layers[i + 1].name for i in range(len(ypreds)) if '2d' in model.layers[i + 1].name]
    tmin = [np.min(ypreds[i]) for i in range(len(layers))]
    tmean = [np.mean(ypreds[i]) for i in range(len(layers))]
    tmed = [np.median(ypreds[i].flatten()) for i in range(len(layers))]
    tmax = [np.max(ypreds[i]) for i in range(len(layers))]
    thresholds = tmean

    cols = ['index']
    for i in range(len(layers)):
        for j in range(ypreds[i].shape[3]):
            cols.append(layers[i] + '_' + str(j) + '_0')
            cols.append(layers[i] + '_' + str(j) + '_1')
        break

    buffer = []
    for i in range(len(images)):
        layeractivations = [i]
        for l in range(len(layers)):
            outs = ypreds[l][i]
            activations = []
            for f in range(outs.shape[2]):
                fmap = outs[:, :, f]
                r = isActivated(fmap, thresholds[l])
                activations.append(r == 0)
                activations.append(r == 1)
            layeractivations += activations
            break
        buffer.append(layeractivations + [ymaxpreds[i], labels[i][0], int(eval[i])])

    trans = pd.DataFrame(buffer, columns=cols + ['predicted', 'label', 'iscorrect'])
    trans.set_index('index', inplace=True)
    return trans


def generateDataset(model, images, labels):
    ypreds = model.predict(images)
    #a = model.predict_proba(images)
    b = model.predict_classes(images)
    ymaxpreds = [np.argmax(x) for x in ypreds[-1]]
    eval = [x[0] == x[1] for x in zip(map(lambda x: x[0], labels), ymaxpreds)]
    layers = [model.layers[i + 1].name for i in range(len(ypreds)) if '2d' in model.layers[i + 1].name]
    #thresholds = [np.max(ypreds[i]) * 0.75 for i in range(len(layers))]
    thresholds = [np.mean(ypreds[i]) for i in range(len(layers))]

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


def row2itemset(row):
    items = set()
    for index, val in row.items():
        if val == 1:
            items.add(index)
    return items


def findMatches(pattern, trans):
    matches = []
    for index, row in trans.iterrows():
        itemset = row2itemset(row)
        if pattern.issubset(itemset):
            matches.append(index)
    return matches


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


def minePats(ds, minsup):
    pats = fpgrowth(ds, min_support=minsup, use_colnames=True)

    # Select closed patterns
    su = pats.support.unique()  # all unique support count
    # Dictionay storing itemset with same support count key
    fredic = {}
    for i in range(len(su)):
        inset = list(pats.loc[pats.support == su[i]]['itemsets'])
        fredic[su[i]] = inset
    patlist = []
    for index, row in pats.iterrows():
        isclose = True
        cli = row['itemsets']
        cls = row['support']
        patlist.append({'itemset': row['itemsets'], 'support': cls})
        #checkset = fredic[cls]
        #for i in checkset:
        #    if cli != i:
        #        if frozenset.issubset(cli, i):
        #            isclose = False
        #            break
        #if (isclose):
        #    patlist.append({'itemset': row['itemsets'], 'support': cls})
    return patlist


def mineContrastPats(correct, incorrect):
    if 'iscorrect' in incorrect.columns:
        incorrect = incorrect.drop('iscorrect', axis=1)
    incorrectpats = minePats(incorrect, 0.5)

    correctset = []
    for index, row in correct.iterrows():
        itemset = row2itemset(row)
        correctset.append(itemset)

    contrastpats = []
    for row in incorrectpats:
        pat = row['itemset']
        correctcount = 0
        for c in correctset:
            if pat.issubset(c):
                correctcount += 1
        correctsup = correctcount / float(len(correctset))
        if row['support'] > correctsup:
            contrastpats.append({'correctsupport': correctsup, 'incorrectsupport': row['support'], 'growth': 0.0,
                                'pattern': row['itemset']})


    contrastpats.sort(key=lambda x: x['incorrectsupport'] - x['correctsupport'])

    for c in contrastpats:
        print(str(c['incorrectsupport']) + ', ' + str(c['correctsupport']) + ', ' + str(c['pattern']))
    print(1)


def mineContrastPats2(correct, incorrect):

    if 'iscorrect' in correct.columns:
        correct = correct.drop('iscorrect', axis=1)
    correct = correct.drop(correct.columns[[i for i in range(23)]], axis=1)
    correctpats = minePats(correct, 0.1)



    if 'iscorrect' in incorrect.columns:
        incorrect = incorrect.drop('iscorrect', axis=1)
    #incorrect = incorrect.drop(incorrect.columns[[i for i in range(23)]], axis=1)
    incorrectpats = minePats(incorrect, 0.5)


    contrastpats = []
    #for index, row in incorrectpats.iterrows():
    for row in incorrectpats:
        pat = row['itemset']
        print('checking:' + str(pat))
        iscontrast = False
        ismatch = False
        correctsup = 0.0
        growth = 0.0
        #for ci, cr in correctpats.iterrows():
        for cr in correctpats:
            if cr['itemset'] == pat:
                ismatch = True
                correctsup = cr['support']
                growth = row['support'] / cr['support']
                iscontrast = row['support'] > cr['support']
                print('correct:' + str(cr['support']) + ', incorrect:' + str(row['support']))
                break
        #if not ismatch:
        if iscontrast or not ismatch:
            contrastpats.append({'correctsupport': correctsup, 'incorrectsupport': row['support'], 'growth': growth, 'pattern': set(row['itemset'])})

    contrastpats.sort(key=lambda x: x['incorrectsupport'])

    for c in contrastpats:
        print(str(c['incorrectsupport']) + ', ' + str(c['correctsupport']) + ', ' + str(c['pattern']))

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

