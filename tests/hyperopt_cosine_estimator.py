import re, pdb
from box import Box
from ml_tools import CosineEstimator, Similars
from ml_tools.fixtures import articles
import numpy as np
import pandas as pd
from hyperopt import hp
from hyperopt.pyll import scope
from hyperopt import fmin, tpe, space_eval

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--winner', action='store_true', help='Just try winning hypers (1 run)')
args_p = parser.parse_args()

lhs = articles()
lhs = Similars(lhs).embed().cluster(algo='agglomorative').value()

rhs = np.load('/storage/libgen/testing.npy') #, mmap_mode='r')
books = pd.read_feather('/storage/libgen/testing.df')

dnn = CosineEstimator(lhs, rhs)

adjusts = Box(
    entries=r"(cbt|virtual|cognitive)",
    mine_up= r"(python|javascript|tensorflow)",
    mine_down=r"(america|politics|united states)",
)
adjusts.update(
    other_up=r"(president|trump|elections|fitness|gym|exercise)",
    other_down=r"(python|javascript|astrology|moon)"
)
vote_ct = 100
def adjust(k, std):
    ct = 0
    def adjust_(text):
        nonlocal ct
        if ct > vote_ct: return 0.
        res = std if re.search(adjusts[f"{k}_up"], text, re.IGNORECASE)\
            else -std if re.search(adjusts[f"{k}_down"], text, re.IGNORECASE)\
            else 0.
        if res: ct += 1
        return res
    return adjust_
all_txt = (books.title + books.text)


def ct_match(regex):
    def ct_match_(txt):
        return 1 if re.search(regex, txt, re.IGNORECASE) else 0
    return ct_match_

from sklearn.utils import class_weight
will_adjust = np.zeros(rhs.shape[0])
will_adjust[:vote_ct*2] = np.ones(vote_ct*2)
cw = class_weight.compute_sample_weight('balanced', (will_adjust != 0))
max_sample_weight = cw.max() / cw.min()
print('max_sample_weight', max_sample_weight)

table, max_evals = [], 1000
def objective(args):
    # first override the unecessary nesting, I don't like that
    for i in [0, 1, 2]:
        k = f"l{i}"
        if type(args[k]) == dict:
            args[k] = args[k][f"{k}_n"]
    print(args)


    std_mine, std_other = args['std_mine'], args['std_mine'] * args['std_other']
    adjust_mine = all_txt.apply(adjust('mine', std_mine))
    # start from other end so there's no overlap
    adjust_other = all_txt[::-1].apply(adjust('other', std_other))[::-1]
    adjust_ = (adjust_mine + adjust_other).values

    dnn.hypers = Box(args)
    dnn.adjustments = adjust_
    dnn.init_model(load=False)
    dnn.fit()
    mse = np.clip(dnn.loss, 0., 1.)

    # see how many subjectively-good books it recommended
    books['dist'] = dnn.predict()
    df = books.sort_values('dist').iloc[:300]
    texts = df.title + df.text
    print(df.title.iloc[:50])

    args['mse'] = mse
    args['n_orig'] = texts.apply(ct_match(adjusts.entries)).sum()
    for k in ['mine_up', 'mine_down', 'other_up', 'other_down']:
        args[f"n_{k}"] = texts.apply(ct_match(adjusts[k])).sum()
    score = args['n_orig'] + args['n_mine_up'] - args['n_mine_down']\
        + args['n_other_up']/10 - args['n_other_down']/10
    score = score - 10*np.log10(mse)
    args['score'] = score
    table.append(args)

    df = pd.DataFrame(table).sort_values('score', ascending=False)
    print(f"Top 5 ({df.shape[0]}/{max_evals})")
    print(df.iloc[:5])
    print("All")
    print(df)
    if not args_p.winner:
        df.to_csv('./hypers.csv')

    return -score

# define a search space
space = {
    # actually comes through as {"l0": val}, see above
    'l0': {'l0_n': hp.uniform('l0_n', 0.1, 1.)},
    'l1': hp.choice('l1', [
        {'l1_n': False},
        {'l1_n': hp.uniform('l1_n', 0.1, 1.)}
    ]),
    'l2': {'l2_n': False},
    # hp.choice('l2', [
    #     {'l2_n': False},
    #     {'l2_n': hp.uniform('l2_n', 0.1, 1.)}
    # ]),
    'act': hp.choice('act', ['relu', 'elu', 'tanh']),
    # no relu, since even though we constrain cosine positive, the adjustments may become negative
    'final': 'linear',  # hp.choice('final', ['sigmoid', 'linear']),
    'loss': hp.choice('loss', ['mse', 'mae']),
    'batch': scope.int(hp.quniform('batch', 32, 512, 32)),
    'bn': hp.choice('bn', [True, False]),
    'opt': hp.choice('opt', ['adam', 'nadam']),
    'lr': hp.uniform('lr', .0001, .001),
    'sample_weight': hp.uniform('sample_weight', 1., max_sample_weight),
    'std_mine': hp.uniform('std_mine', .01, 1.),
    'std_other': hp.uniform('std_other', .0, 1.)  # is multiplied by std_min
}

if args_p.winner:
    objective(dnn.hypers)
else:
    # minimize the objective over the space
    best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, show_progressbar=False)
    print(best)
    # -> {'a': 1, 'c2': 0.01420615366247227}
    print(space_eval(space, best))
    # -> ('case 2', 0.01420615366247227}
