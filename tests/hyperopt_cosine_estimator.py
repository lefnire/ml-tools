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
parser.add_argument('--dump', action='store_true', help='Dump results to DB')
args_p = parser.parse_args()

lhs = articles()
lhs = Similars(lhs).embed().cluster(algo='agglomorative').value()

rhs = np.load('/storage/libgen/testing.npy') #, mmap_mode='r')
books = pd.read_feather('/storage/libgen/testing.df')
rhs_norm = Similars(rhs).normalize().value()

dnn = CosineEstimator(lhs, rhs)

votes = Box(
    mine_up= r"(tensorflow|keras)",
    other_up=r"(cookbook|recipe)",
    other_down=r"(artificial|\bai\b|python|java|cbt|cognitive.?behav)"
)
votes['mine_down'] = votes.other_up
searches = Box(
    entries=r"(virtual.?reality|\bvr\b|oculus|cognitive.?behav|therap|cbt|dbt|dialectical|depression|anxiety)",
    mine_up= r"(python|tensorflow|machine.?learn|keras|pytorch|scikit|pandas|artificial|\bai\b|data.?science|deep.?learn)",
    other_up=r"(cook|recipe|comfort.?food|meal)",
)
searches.update(
    mine_down=searches.other_up,
    other_down=votes.other_down
)

vote_ct = 200
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

# from sklearn.utils import class_weight
# will_adjust = np.zeros(rhs.shape[0])
# will_adjust[:vote_ct*2] = np.ones(vote_ct*2)
# cw = class_weight.compute_sample_weight('balanced', (will_adjust != 0))
# max_sample_weight = cw.max() / cw.min()
# print('max_sample_weight', max_sample_weight)
max_sample_weight = 500.


max_evals = 1000
def objective(trial):
    h = Box({})
    h['layers'] = trial.suggest_int('layers', 1, 2)
    for i in range(h.layers):
        h[f"l{i}"] = trial.suggest_uniform(f"l{i}", .1, 1.)
    h['act'] = trial.suggest_categorical('act', ['relu', 'elu', 'tanh'])
    # no relu, since even though we constrain cosine positive, the adjustments may become negative
    h['final'] = trial.suggest_categorical('final', ['sigmoid', 'linear'])
    if h.final == 'linear':
        h['loss'] = trial.suggest_categorical('loss', ['mse', 'mae'])
    else:
        h['loss'] = 'binary_crossentropy'
    h['batch'] = int(trial.suggest_uniform('batch', 32, 512))
    h['bn'] = trial.suggest_categorical('bn', [True, False])
    h['normalize'] = trial.suggest_categorical('normalize', [True, False])
    h['opt'] = trial.suggest_categorical('opt', ['sgd', 'nadam'])
    h['lr'] = trial.suggest_uniform('lr', .0001, .001)
    h['sample_weight'] = trial.suggest_uniform('sample_weight', 1., max_sample_weight)
    h['std_mine'] = trial.suggest_uniform('std_mine', .1, 1.)
    h['std_other'] = trial.suggest_uniform('std_other', .1, .6)  # is multiplied by std_min

    print(h)

    std_other = h.std_mine * h.std_other
    adjust_mine = all_txt.apply(adjust('mine', h.std_mine))
    # start from other end so there's no overlap
    adjust_other = all_txt[::-1].apply(adjust('other', std_other))[::-1]
    adjust_ = (adjust_mine + adjust_other).values


    dnn.hypers = Box(h)
    dnn.rhs = rhs_norm if h.normalize else rhs
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
    score = cts.orig + cts.mine_up * 1.15 - cts.mine_down\
        + cts.other_up*.1 - cts.other_down*.1
    trial.set_user_attr('mse', float(mse))
    for k, v in cts.items():
        trial.set_user_attr(k, float(v))
    #score = score - 10*np.log10(mse)

    return -score

if args_p.winner:
    # https://blog.devart.com/pivot-tables-in-postgresql.html
    raise Exception("Won't work after Optuna conversion, fix this.")
    objective(dnn.hypers)
    exit(0)

STUDY = "study4"
DB = os.getenv("DB_URL", None)
study = optuna.create_study(study_name=STUDY, storage=DB, load_if_exists=True)
if not args_p.dump:
    study.optimize(objective, n_trials=max_evals)

from sqlalchemy import create_engine
engine = create_engine(DB)
df = study.trials_dataframe(attrs=('value', 'params', 'user_attrs')).sort_values("value")
for c in df.columns:
    df.rename(columns={c: re.sub(r"(user_attrs_|params_)", "", c)}, inplace=True)
df.to_sql(f"results_{STUDY}", engine, if_exists="replace")
