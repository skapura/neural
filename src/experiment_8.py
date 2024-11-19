import pandas as pd
import data
import patterns as pats

def remote():
    bdf = pd.read_csv('session/bdf.csv', index_col='index')

    # Mine patterns
    print('mine patterns')
    minsup = 0.5
    minsupratio = 1.1
    sel, notsel = data.filter_transactions(bdf, 'activation-', 0)
    cpats = pats.mine_patterns(sel, notsel, minsup, minsupratio)
    elems, targetmatches, othermatches = pats.unique_elements(cpats)
    patternset = list(elems.keys())
    pattern = cpats[0]

def run():
    bdf = pd.read_csv('session/bdf.csv', index_col='index')


    print(1)
