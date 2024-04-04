from skmine.itemsets import SLIM, LCM
from skmine.itemsets.lcm import LCMMax
from skmine.emerging import MBDLLBorder
import pandas as pd

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

