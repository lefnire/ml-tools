import re, pdb, os
from box import Box
from ml_tools import CosineEstimator, Similars
from ml_tools.fixtures import articles
import numpy as np
import pandas as pd
import optuna

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--jobs', help='Number of threads', default=1)
parser.add_argument('--init', action='store_true', help='initialize starter trials')
args_p = parser.parse_args()

lhs = articles()
lhs = Similars(lhs).embed().cluster(algo='agglomorative').value()

rhs = np.load('/storage/libgen/testing.npy') #, mmap_mode='r')
books = pd.read_feather('/storage/libgen/testing.df')

food_re = "cook|recipe|food|meal"
# these should be really specific (think about edge-cases)
votes = Box(
    mine_up=r"(tensorflow|keras)",
    other_up=rf"({food_re}|econom|republican)",
    mine_down=rf"({food_re})",
    other_down=r"(artificial|\bai\b|python|java|cbt|cognitive.?behav)"
)

# these should be pretty general, catch-all
searches = Box(
    entries=r"(virtual|\bvr\b|oculus|cognitive|therap|cbt|dbt|dialectical|depression|anxiety|mindful)",
    mine_up= r"(python|tensorflow|machine.?learn|keras|pytorch|scikit|pandas|numpy|artificial|\bai\b|data|deep.?learn)",
    other_up=rf"({food_re}|trump|president|politic|financ|econom|republican|democrat)",
    mine_down=votes.mine_down,
    other_down=votes.other_down
)

vote_ct = 200
def adjust(k):
    ct = 0
    def adjust_(text):
        nonlocal ct
        if ct > vote_ct: return 0.
        v = 1. if re.search(votes[f"{k}_up"], text, re.IGNORECASE)\
            else -1. if re.search(votes[f"{k}_down"], text, re.IGNORECASE)\
            else 0.
        if v: ct += 1
        return v
    return adjust_

def ct_match(k):
    def ct_match_(txt):
        return 1 if re.search(searches[k], txt, re.IGNORECASE) else 0
    return ct_match_

# from sklearn.utils import class_weight
# will_adjust = np.zeros(rhs.shape[0])
# will_adjust[:vote_ct*2] = np.ones(vote_ct*2)
# cw = class_weight.compute_sample_weight('balanced', (will_adjust != 0))
# max_sample_weight = cw.max() / cw.min()
# print('max_sample_weight', max_sample_weight)
max_sample_weight = 100.


max_evals = 400
def objective(trial):
    h = Box({})
    h['layers'] = 1 # trial.suggest_int('layers', 1, 2)
    for i in range(h.layers):
        h[f"l{i}"] = trial.suggest_uniform(f"l{i}", .2, .6)
    h['act'] = 'relu' # trial.suggest_categorical('act', ['relu', 'elu', 'tanh'])
    h['loss'] = 'mae' # trial.suggest_categorical('loss', ['mse', 'mae'])
    h['batch'] = int(trial.suggest_uniform('batch', 32, 325))
    h['bn'] = True # trial.suggest_categorical('bn', [True, False])
    h['normalize'] = trial.suggest_categorical('normalize', [True, False])
    h['opt'] = trial.suggest_categorical('opt', ['amsgrad', 'nadam'])
    h['lr'] = .0004 # trial.suggest_uniform('lr', .0001, .001)
    h['sw_mine'] = trial.suggest_uniform('sw_mine', 2., max_sample_weight)
    sw_other = trial.suggest_uniform('sw_other', .01, 1.)
    h['sw_other'] = max(1.5, sw_other * h.sw_mine)
    h['std_mine'] = .3 # trial.suggest_uniform('std_mine', .1, .5)
    std_other = .15 # trial.suggest_uniform('std_other', .1, 1.)
    h['std_other'] = max(.01, std_other * h.std_mine)

    df = pd.DataFrame({'title': books.title})
    adjusts = []
    for k in ['mine', 'other']:
        shuff = df.sample(frac=1).index
        df.loc[shuff, k] = df.loc[shuff, 'title'].apply(adjust(k))
        adjusts.append(dict(values=df[k].values, amount=h[f'std_{k}'], weight=h[f'sw_{k}']))

    dnn = CosineEstimator(lhs, rhs, adjustments=adjusts, hypers=h)
    dnn.fit()
    mse = np.clip(dnn.loss, 0., 1.)

    # see how many subjectively-good books it recommended
    books['dist'] = dnn.predict()
    df = books.sort_values('dist').iloc[:300]
    titles = df.title
    print(titles.iloc[:50])

    cts = Box({})
    cts['orig'] = titles.apply(ct_match('entries')).sum()
    for k in ['mine_up', 'mine_down', 'other_up', 'other_down']:
        cts[k] = titles.apply(ct_match(k)).sum()
    score = cts.orig + cts.mine_up * 1.5 - cts.mine_down\
        + cts.other_up*.1 - cts.other_down*.1
    trial.set_user_attr('mse', float(mse))
    for k, v in cts.items():
        trial.set_user_attr(k, float(v))
    score = score - 10*np.log10(mse)

    return -score


STUDY = "study3"
DB = os.getenv("DB_URL", None)
from sqlalchemy import create_engine
engine = create_engine(DB)

def save_results(study, frozen_trial):
    try:
        # fails if only 1 trial
        imp = optuna.importance.get_param_importances(study)
        print(imp)
    except: pass
    df = study.trials_dataframe(attrs=('value', 'params', 'user_attrs')).sort_values("value")
    for c in df.columns:
        df.rename(columns={c: re.sub(r"(user_attrs_|params_)", "", c)}, inplace=True)
    try:
        # fails if race-condition on drop-table
        df.to_sql(f"results_{STUDY}", engine, if_exists="replace")
    except: pass


study = optuna.create_study(study_name=STUDY, storage=DB, load_if_exists=True)
if args_p.init:
    study.enqueue_trial(dict(l0=.5, batch=250, opt='nadam', normalize=True, sw_mine=50., sw_other=.04))
    study.enqueue_trial(dict(l0=.25, batch=250, opt='nadam', normalize=True, sw_mine=50., sw_other=.04))
    study.enqueue_trial(dict(l0=.5, batch=250, opt='nadam', normalize=False, sw_mine=50., sw_other=.04))
    study.enqueue_trial(dict(l0=.5, batch=250, opt='amsgrad', normalize=True, sw_mine=50., sw_other=.04))
study.optimize(objective, n_trials=max_evals, n_jobs=int(args_p.jobs), callbacks=[save_results])
