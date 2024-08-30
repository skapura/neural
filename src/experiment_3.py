from keras import models
import pandas as pd
from data import load_dataset, load_dataset_selection, filter_dataset_paths, scale
import patterns as pats
import mlutil
import azure as az
import const



def run():
    model = az.train('src/models/golf.py')
    model.save('src/models/golf.keras')

    outputlayers = ['activation', 'activation_1', 'activation_2', 'activation_3', 'prediction']
    model = models.load_model('largeimage16.keras')
    pmodel = mlutil.make_output_model(model, outputlayers)
    trainds, valds = load_dataset('images_large')


    golfimages = filter_dataset_paths(trainds, 'golf_ball')
    notgolfimages = filter_dataset_paths(trainds, 'golf_ball', in_flag=False)
    golfds = load_dataset_selection('images_large', selection=(golfimages, notgolfimages), label_mode='binary')

    churches = [i for i, r in enumerate(trainds.file_paths) if 'church' in r]
    rfield0 = mlutil.receptive_field(0, 0, pmodel.receptive_subset('activation'))
    rfield1 = mlutil.receptive_field(0, 0, pmodel.receptive_subset('activation_1'))
    rfield2 = mlutil.receptive_field(0, 0, pmodel.receptive_subset('activation_2'))
    rfield3 = mlutil.receptive_field(0, 0, pmodel.receptive_subset('activation_3'))

    model = az.train('models/golf.py')
    model.save('models/golf.keras')

    # Generate transactions
    franges = get_ranges(trans, zeromin=True)
    pats.model_to_transactions(model, outputlayers, golfds)

    # Load transactions
    trans = pd.read_csv('session/trans_feat16.csv', index_col='index')
    #franges = get_ranges(trans, zeromin=True)
    scaled = scale(trans, output_range=(0, 1))
    bdf = pats.binarize(scaled, 0.5)

    # Restrict patterns to layer
    #col = bdf.columns.difference(const.META, sort=False)
    col = [c for c in bdf.columns if 'activation_3-' in c]

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