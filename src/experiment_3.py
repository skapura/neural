from keras import models
import pandas as pd
from data import load_dataset, scale
import patterns as pats
import const



def run():
    #model = azure.train(model_path='src/models/test.keras')
    #model.save('src/models/test.keras')

    model = models.load_model('largeimage16.keras')
    trainds, valds = load_dataset('images_large')

    # Load transactions
    trans = pd.read_csv('session/trans_feat16.csv', index_col='index')
    #franges = get_ranges(trans, zeromin=True)
    scaled = scale(trans, output_range=(0, 1))
    bdf = pats.binarize(scaled, 0.5)
    #col = bdf.columns.difference(const.META, sort=False)
    col = [c for c in bdf.columns if 'activation_3-' in c]

    # church patterns
    sel = bdf.loc[bdf['label'] == 2.0].drop(const.META, axis=1)[col]
    notsel = bdf.loc[bdf['label'] != 2.0].drop(const.META, axis=1)[col]
    minsup = 0.1
    minsupratio = 1.1
    cpats = pats.mine_patterns(sel, notsel, minsup, minsupratio)

    # dog patterns
    sel = bdf.loc[bdf['label'] == 0.0].drop(const.META, axis=1)[col]
    notsel = bdf.loc[bdf['label'] != 0.0].drop(const.META, axis=1)[col]
    minsup = 0.7
    minsupratio = 1.1
    cpats = pats.mine_patterns(sel, notsel, minsup, minsupratio)
    print(1)