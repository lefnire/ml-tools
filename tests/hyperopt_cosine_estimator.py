import re
from tensorflow.keras.losses import mean_squared_error
from box import Box
from ml_tools import CosineEstimator, Similars
from ml_tools.fixtures import articles
import numpy as np
import pandas as pd
from hyperopt import hp
from hyperopt.pyll import scope
from hyperopt import fmin, tpe, space_eval

lhs = articles()
lhs = Similars(lhs).embed().cluster(algo='agglomorative').value()

rhs = np.load('/storage/libgen/testing.npy') #, mmap_mode='r')
books = pd.read_feather('/storage/libgen/testing.df')

dnn = CosineEstimator(rhs)
# TODO hyperopt adjustments
adjustments = np.zeros((rhs.shape[0],))
regex = r"(cbt|virtual|cognitive)"


table, max_evals = [], 300
def objective(args):
    # first override the unecessary nesting, I don't like that
    for i in [0, 1, 2]:
        args[f"l{i}"] = args[f"l{i}"][f"l{i}_n"]
    print(args)
    dnn.hypers = Box(args)
    dnn.init_model(load=False)
    dnn.fit_cosine()
    # TODO hyper-opt adjustment phase
    # self.fit_adjustments(lhs, adjustments)

    # mse will be score
    preds = dnn.predict(lhs[0:1])
    y_true = Similars(lhs[0:1], rhs).normalize().cosine(abs=True).value().squeeze()
    mse = mean_squared_error(y_true, preds).numpy()

    # but also want to see how many subjectively-good books it recommended
    df = books.copy()
    df['dist'] = preds
    df = df.sort_values('dist').iloc[:200]
    text = df.title + df.text
    nbooks = sum([
        1 if re.search(regex, x, re.IGNORECASE) else 0
        for x in text
    ])

    args['mse'] = mse
    args['nbooks'] = nbooks
    table.append(args)

    df = pd.DataFrame(table).sort_values('mse')
    print(f"Top 5 ({df.shape[0]}/{max_evals})")
    print(df.iloc[:5])
    print("All")
    print(df)
    df.to_csv('./hypers.csv')

    return mse

# define a search space
space = {
    # actually comes through as {"l0": val}, see above
    'l0': {'l0_n': hp.uniform('l0_n', 0.1, 1.)},
    'l1': hp.choice('l1', [
        {'l1_n': False},
        {'l1_n': hp.uniform('l1_n', 0.1, 1.)}
    ]),
    'l2': hp.choice('l2', [
        {'l2_n': False},
        {'l2_n': hp.uniform('l2_n', 0.1, 1.)}
    ]),
    'act': hp.choice('act', ['elu', 'relu', 'tanh']),
    # no relu, since even though we constrain cosine positive, the adjustments may become negative
    'final': hp.choice('final', ['sigmoid', 'linear']),
    'loss': 'mse',  # hp.choice('loss', ['mse', 'mae']),
    'batch': scope.int(hp.quniform('batch', 32, 512, 32)),
    'bn': hp.choice('bn', [True, False]),
    'opt': hp.choice('opt', ['adam', 'nadam'])
}

# minimize the objective over the space
best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, show_progressbar=False)
print(best)
# -> {'a': 1, 'c2': 0.01420615366247227}
print(space_eval(space, best))
# -> ('case 2', 0.01420615366247227}
