import spmf
import numpy as np
import random
import math
import pandas as pd
from itertools import compress
import csv



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
    return cutpoints, mininfo


def discretize(ds, cutpoints):
    bvals = []
    for index, row in ds.iterrows():
        binned = [index]
        vals = row.to_numpy()
        for i in range(len(vals)):
            cuts = cutpoints[i]
            found = False
            for ci in range(len(cuts)):
                if vals[i] <= cuts[ci]:
                    binned.append(ci)
                    found = True
                    break
            if not found:
                binned.append(len(cuts))
        #binned.append(vals[-2])
        #binned.append(vals[-1])
        bvals.append(binned)
    cols = ['index'] + ds.columns.values.tolist()
    bds = pd.DataFrame(bvals, columns=cols)
    bds.set_index('index', inplace=True)
    return bds


def binarize(df, cutpoints):

    # Get binary column names
    bincols = ['index']
    for ci in range(len(df.columns)):
        for i in range(len(cutpoints[ci]) + 1):
            bincols.append(df.columns[ci] + '_' + str(i))
    #bincols.append('iscorrect')

    # Binarize data
    brows = []
    for index, row in df.iterrows():
        rowbuf = [index]
        for ci in range(len(df.columns)):
            v = row.iloc[ci]
            for i in range(len(cutpoints[ci]) + 1):
                rowbuf.append(v == i)
        #rowbuf.append(row['predicted'] == row['label'])
        brows.append(rowbuf)

    bdf = pd.DataFrame(brows, columns=bincols)
    bdf.set_index('index', inplace=True)
    return bdf


def mineContrastPatterns(target, other, minsup, supratio):
    print('start mining')

    # Write transactions to temp file
    lookup = {}
    for i in range(len(target.columns)):
        lookup[target.columns[i]] = i
    with open('.trans.csv', 'w') as outfile:
        writer = csv.writer(outfile, delimiter=' ')
        for index, row in target.iterrows():
            itemset = [lookup[item] for item in list(compress(target.columns, row))]
            writer.writerow(itemset)

    # Perform pattern mining on 'target' dataset
    fpclose = spmf.Spmf('FPClose', input_filename='.trans.csv', output_filename='.patterns.csv', arguments=[minsup, 5])
    fpclose.run()
    fpclose.parse_output()
    patlist = []
    for p in fpclose.patterns_:
        items = p[0].split()
        pat = frozenset([target.columns[int(itm)] for itm in items[:-2]])
        sup = int(items[-1]) / float(len(target))
        patlist.append({'pattern': pat, 'targetsupport': sup, 'othermatches': [], 'targetmatches': []})

    # Convert transactions to sets for pattern matching
    othersets = []
    for index, row in other.iterrows():
        itemset = frozenset(compress(other.columns, row))
        othersets.append({'index': index, 'itemset': itemset})
    targetsets = []
    for index, row in target.iterrows():
        itemset = frozenset(compress(target.columns, row))
        targetsets.append({'index': index, 'itemset': itemset})

    # Find pattern support in both datasets
    i = 0
    for p in patlist:
        print(str(i) + '/' + str(len(patlist)))
        i += 1
        count = 0
        for c in othersets:
            if p['pattern'].issubset(c['itemset']):
                p['othermatches'].append(c['index'])
                count += 1
        for c in targetsets:
            if p['pattern'].issubset(c['itemset']):
                p['targetmatches'].append(c['index'])
        p['othersupport'] = count / float(len(othersets))
        p['supportdiff'] = p['targetsupport'] - p['othersupport']
        p['supportratio'] = p['targetsupport'] / p['othersupport'] if p['othersupport'] > 0.0 else -1.0

    # Prune low support/contrast patterns
    #patlist.sort(key=lambda x: x['targetsupport'] - x['othersupport'], reverse=True)
    patlist.sort(key=lambda x: x['supportratio'], reverse=False)
    selectedpats = []
    for p in patlist:
        if p['supportratio'] >= supratio:
            print(str(p['targetsupport'] - p['othersupport']) + ', ' + str(p['othersupport']) + ', ' + str(p['targetsupport']) + ', ' + str(p['supportratio']) + ', ' + str(p['pattern']))
            selectedpats.append(p)
    return selectedpats


def mineContrastPatterns2(correct, incorrect, minsup, supratio):
    print('start mining')

    # Write transactions to temp file
    lookup = {}
    for i in range(len(incorrect.columns)):
        lookup[incorrect.columns[i]] = i
    with open('.trans.csv', 'w') as outfile:
        writer = csv.writer(outfile, delimiter=' ')
        for index, row in incorrect.iterrows():
            #itemset = [lookup[item] for item in list(compress(incorrect.columns, row)) if item.endswith('_1')]
            itemset = [lookup[item] for item in list(compress(incorrect.columns, row))]
            writer.writerow(itemset)

    # Perform pattern mining on 'incorrect' dataset
    fpclose = spmf.Spmf('FPClose', input_filename='.trans.csv', output_filename='.patterns.csv', arguments=[minsup, 5])
    fpclose.run()
    fpclose.parse_output()
    patlist = []
    for p in fpclose.patterns_:
        items = p[0].split()
        pat = frozenset([incorrect.columns[int(itm)] for itm in items[:-2]])
        sup = int(items[-1]) / float(len(incorrect))
        patlist.append({'pattern': pat, 'incorrectsupport': sup, 'correctmatches': [], 'incorrectmatches': []})

    # Convert transactions to sets for pattern matching
    correctsets = []
    for index, row in correct.iterrows():
        itemset = frozenset(compress(correct.columns, row))
        correctsets.append({'index': index, 'itemset': itemset})
    incorrectsets = []
    for index, row in incorrect.iterrows():
        itemset = frozenset(compress(incorrect.columns, row))
        incorrectsets.append({'index': index, 'itemset': itemset})

    # Find pattern support in both datasets
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
        p['supportratio'] = p['incorrectsupport'] / p['correctsupport'] if p['correctsupport'] > 0.0 else -1.0

    # Prune low support/contrast patterns
    patlist.sort(key=lambda x: x['incorrectsupport'] - x['correctsupport'], reverse=True)
    selectedpats = []
    for p in patlist:
        if p['supportratio'] >= supratio:
            print(str(p['incorrectsupport'] - p['correctsupport']) + ', ' + str(p['correctsupport']) + ', ' + str(p['incorrectsupport']) + ', ' + str(p['supportratio']) + ', ' + str(p['pattern']))
            selectedpats.append(p)
    return selectedpats
