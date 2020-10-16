import re, pdb, os
from box import Box
from ml_tools import CosineEstimator, Similars
from ml_tools.fixtures import articles
import numpy as np
import pandas as pd
import optuna

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--winner', action='store_true', help='Just try winning hypers (1 run)')
args_p = parser.parse_args()

lhs = articles()
lhs = Similars(lhs).embed().cluster(algo='agglomorative').value()

rhs = np.load('/storage/libgen/testing.npy') #, mmap_mode='r')
books = pd.read_feather('/storage/libgen/testing.df')

dnn = CosineEstimator(lhs, rhs)

votes = Box(
    mine_up= r"tensorflow|keras",
    other_up=r"(cookbook|recipes)"
)
votes.update(
    mine_down=votes.other_up,
    other_down=votes.mine_up
)
searches = Box(
    entries=r"(cbt|virtual|cognitive)",
    mine_up= r"(python|tensorflow|machine learning|keras|pytorch|scikit|pandas)",
    other_up=r"(cook|recipes|comfort food|meal)",
)
searches.update(
    mine_down=searches.other_up,
    other_down=searches.mine_up
)

vote_ct = 100
def adjust(k, std):
    ct = 0
    def adjust_(text):
        nonlocal ct
        if ct > vote_ct: return 0.
        v = std if re.search(votes[f"{k}_up"], text, re.IGNORECASE)\
            else -std if re.search(votes[f"{k}_down"], text, re.IGNORECASE)\
            else 0.
        if v: ct += 1
        return v
    return adjust_
all_txt = (books.title + books.text)


def ct_match(k):
    def ct_match_(txt):
        return 1 if re.search(searches[k], txt, re.IGNORECASE) else 0
    return ct_match_

from sklearn.utils import class_weight
will_adjust = np.zeros(rhs.shape[0])
will_adjust[:vote_ct*2] = np.ones(vote_ct*2)
cw = class_weight.compute_sample_weight('balanced', (will_adjust != 0))
max_sample_weight = cw.max() / cw.min()
print('max_sample_weight', max_sample_weight)


max_evals = 1000
def objective(trial):
    h = Box({})
    h['layers'] = trial.suggest_int('layers', 1, 2)
    for i in range(h.layers):
        h[f"l{i}"] = trial.suggest_uniform(f"l{i}", .1, 1.)
    h['act'] = 'elu'  # hp.choice('act', ['relu', 'elu', 'tanh'])
    # no relu, since even though we constrain cosine positive, the adjustments may become negative
    h['final'] = trial.suggest_categorical('final', ['sigmoid', 'linear'])
    h['loss'] = 'mae'  # hp.choice('loss', ['mse', 'mae'])
    h['batch'] = int(trial.suggest_uniform('batch', 32, 512))
    h['bn'] = False  # hp.choice('bn', [True, False])
    h['opt'] = trial.suggest_categorical('opt', ['sgd', 'nadam'])
    h['lr'] = trial.suggest_uniform('lr', .0001, .001)
    h['sample_weight'] = trial.suggest_uniform('sample_weight', 1., max_sample_weight)
    h['std_mine'] = trial.suggest_uniform('std_mine', .1, 1.)
    h['std_other'] = trial.suggest_uniform('std_other', .1, .6)  # is multiplied by std_min

    std_other = h.std_mine * h.std_other
    adjust_mine = all_txt.apply(adjust('mine', h.std_mine))
    # start from other end so there's no overlap
    adjust_other = all_txt[::-1].apply(adjust('other', std_other))[::-1]
    adjust_ = (adjust_mine + adjust_other).values

    dnn.hypers = Box(h)
    dnn.adjustments = adjust_
    dnn.init_model(load=False)
    dnn.fit()
    mse = np.clip(dnn.loss, 0., 1.)

    # see how many subjectively-good books it recommended
    books['dist'] = dnn.predict()
    df = books.sort_values('dist').iloc[:300]
    texts = df.title + df.text
    print(df.title.iloc[:50])

    cts = Box({})
    cts['orig'] = texts.apply(ct_match('entries')).sum()
    for k in ['mine_up', 'mine_down', 'other_up', 'other_down']:
        cts[k] = texts.apply(ct_match(k)).sum()
    score = cts.orig + cts.mine_up*.75 - cts.mine_down\
        + cts.other_up*.1 - cts.other_down*.1
    trial.set_user_attr('mse', float(mse))
    for k, v in cts.items():
        trial.set_user_attr(k, float(v))
    #score = score - 10*np.log10(mse)

    return -score

if args_p.winner:
    raise Exception("Won't work after Optuna conversion, fix this.")
    objective(dnn.hypers)
else:
    study = optuna.create_study(study_name='study2', storage=os.getenv("DB_URL", None), load_if_exists=True)
    study.optimize(objective, n_trials=max_evals)
