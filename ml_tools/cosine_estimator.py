# https://github.com/tensorflow/tensorflow/issues/2117
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence

import pdb, logging, math, re
from os.path import exists
from box import Box
from ml_tools import Similars
import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)


def permute(arr: np.ndarray):
    return np.random.permutation(arr.shape[0])

class CosineEstimator:
    """
    Neural network that learns the cosine DISTANCE function (between 0-1, 0 being similar, 1 being distant). Also
    allows fine-tuning adjustments of those similarities, eg in the case of user-ratings on documents (embedded).
    """
    def __init__(self, rhs, filename=None):
        """
        :param rhs: right-hand-side, ie the database/corpus/index you'll be comparing things against later. Usually
            this will be the much larger of the two matrices in a comparison.
        """
        self.hypers = Box({
            'l1': 800,
            'l2': {'n': 300},
            'act': 'tanh',
            'final': 'sigmoid',
            'loss': 'mse',  # winner
            'batch': 300,
            'norm': False  # winner
        })

        self.rhs = rhs
        self.filename = filename
        self.split = int(rhs.shape[0] * .7)
        self.model = None
        self.es = EarlyStopping(monitor='val_loss', mode='min', patience=3, min_delta=.0001)
        self.loaded = False
        self.init_model()

    def _dset(self, arr: np.ndarray, validation):
        return arr[self.split:] if validation else arr[:self.split]

    def _nsteps(self, arr: np.ndarray, batch_size, split):
        return math.ceil(arr.shape[0] * split / batch_size)

    def generator_cosine(self, batch_size, validation=False):
        """
        # TODO move to Sequence subclass
        # https://stackoverflow.com/questions/55889923/how-to-handle-the-last-batch-using-keras-fit-generator
        """
        rhs = self._dset(self.rhs, validation)
        while True:
            idx = permute(rhs)[:batch_size]
            a = rhs[idx]
            b = a[permute(a)]
            chain = Similars(a, b).normalize()
            if self.hypers.norm is True:
                a, b = chain.value()

            x = np.hstack([a, b])
            y = chain.cosine(abs=True).value()
            y = y.diagonal()
            yield x, y

    def generator_adjustments(self, lhs, adjustments, batch_size, validation=False):
        rhs = self._dset(self.rhs, validation)
        adjustments = self._dset(adjustments, validation)
        while True:
            idx = permute(rhs)[:batch_size]
            b = rhs[idx]
            adj = adjustments[idx]

            # repeat lhs, cut off excess
            n_lhs, n_rhs = lhs.shape[0], b.shape[0]
            a = np.tile(lhs, (math.ceil(n_rhs/n_lhs), 1))[:n_rhs]

            chain = Similars(a, b).normalize()
            if self.hypers.norm is True:
                a, b = chain.value()
            x = np.hstack([a, b])
            y = chain.cosine(abs=True).value().diagonal()
            # Push highly-rated docs up, low-rated docs down. Using negative-score because cosine DISTANCE
            # (less is better)
            y = y - y.std() * adj
            yield x, y

    def generator_predict(self, batch_size, x):
        rhs = self.rhs
        for i in range(0, rhs.shape[0], batch_size):
            b = rhs[i:i+batch_size]
            a = np.repeat(x, b.shape[0], axis=0)
            if self.hypers.norm is True:
                a, b = Similars(a, b).normalize().value()
            yield np.hstack([a, b])

    def init_model(self, load=True):
        if load and self.filename and exists(self.filename):
            logger.info("DNN: pretrained model")
            self.model = load_model(self.filename)
            self.loaded = True
            return
        self.loaded = False

        h = self.hypers

        input = Input(shape=(self.rhs.shape[1] * 2,))
        m = input
        if h.norm == 'bn':
            m = BatchNormalization()(m)
        m = Dense(int(h.l1), activation=h.act)(m)
        if h.l2.n:
            m = Dense(int(h.l2.n), activation=h.act)(m)
        m = Dense(1, activation=h.final)(m)
        m = Model(input, m)
        # http://zerospectrum.com/2019/06/02/mae-vs-mse-vs-rmse/
        # MAE because we _want_ outliers (user score adjustments)
        loss = 'binary_crossentropy' if h.final == 'sigmoid' else h.loss
        m.compile(
            loss=loss,
            optimizer=Adam(learning_rate=.0001),
        )
        m.summary()
        self.model = m

    def fit_cosine(self):
        if self.loaded:
            logger.info("DNN: using cosine-pretrained")
            return
        else:
            logger.info("DNN: learn cosine function")

        # https://www.machinecurve.com/index.php/2020/04/06/using-simple-generators-to-flow-data-from-file-with-keras/
        # https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        batch_size = int(self.hypers.batch)
        self.model.fit(
            self.generator_cosine(batch_size),
            epochs=50,
            callbacks=[self.es],
            validation_data=self.generator_cosine(batch_size, validation=True),
            steps_per_epoch=self._nsteps(self.rhs, batch_size, .7),
            validation_steps=self._nsteps(self.rhs, batch_size, .3),
            # workers=THREADS,
            # use_multiprocessing=True
        )
        if self.filename:
            self.model.save(self.filename)

    def fit_adjustments(self, lhs, adjustments):
        if not adjustments.any(): return
        logger.info("DNN: learn adjustments function")
        batch_size = 16
        self.model.fit(
            self.generator_adjustments(lhs, adjustments, batch_size),
            epochs=7,  # too many epochs overfits (eg to CBT). Maybe adjust LR *down*, or other?
            # callbacks=[self.es],
            validation_data=self.generator_adjustments(lhs, adjustments, batch_size, validation=True),
            steps_per_epoch=self._nsteps(lhs, batch_size, .7),
            validation_steps=self._nsteps(lhs, batch_size, .3),
            # workers=THREADS,
            # use_multiprocessing=True
        )

    def predict(self, x):
        batch_size = 1000
        best = None
        for x_ in x:
            preds = self.model.predict(
                self.generator_predict(batch_size, [x_]),
                steps=self._nsteps(self.rhs, batch_size, 1),
                verbose=1,
                # workers=THREADS,
                # use_multiprocessing=True
            ).squeeze()
            # preds = Similars([x], books).normalize().cosine(abs=True).value()
            if best is None:
                best = preds
                continue
            best = np.vstack([best, preds]).min(axis=0)
        return best


    def hyperopt(self, lhs, adjustments, dataframe, regex: str):
        table, max_evals = [], 100
        def objective(args):
            print(args)
            self.hypers = Box(args)
            self.init_model(load=False)
            self.fit_cosine()
            self.fit_adjustments(lhs, adjustments)
            preds = self.predict(lhs)
            df = dataframe.copy()
            df['dist'] = preds
            df = df.sort_values('dist').iloc[:200]
            text = df.title + df.text
            score = sum([
                1 if re.search(regex, x, re.IGNORECASE) else 0
                for x in text
            ])
            args['score'] = score
            table.append(args)
            df = pd.DataFrame(table).sort_values('score', ascending=False)
            print(f"Top 5 ({df.shape[0]}/{max_evals})")
            print(df.iloc[:5])
            print("All")
            print(df)
            df.to_csv('./hypers.csv')
            return -score

        # define a search space
        from hyperopt import hp
        space = {
            'l1': hp.quniform('l1', 400, 1400, 100),
            'l2': hp.choice('l2', [
                {'n': None},
                {'n': hp.quniform('n', 10, 600, 20)}
            ]),
            # no relu, since we may want negative values downstream
            'act': hp.choice('act', ['tanh', 'elu']),
            # no relu, since even though we constrain cosine positive, the adjustments may become negative
            'final': hp.choice('final', ['sigmoid', 'linear', 'elu']),
            'loss': 'mse',  # hp.choice('loss', ['mse', 'mae']),
            'batch': 300,  # hp.quniform('batch', 32, 512, 32),
            'norm': False,  # hp.choice('norm', [True, False, 'bn'])
        }

        # minimize the objective over the space
        from hyperopt import fmin, tpe, space_eval
        best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, show_progressbar=False)

        print(best)
        # -> {'a': 1, 'c2': 0.01420615366247227}
        print(space_eval(space, best))
        # -> ('case 2', 0.01420615366247227}
