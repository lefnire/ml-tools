# https://github.com/tensorflow/tensorflow/issues/2117
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, Nadam
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
            'l0': .5,
            'l1': .4,
            'l2': .3,
            'act': 'tanh',
            'final': 'linear',
            'loss': 'mae',
            'batch': 64,
            'bn': False,
            'opt': 'adam',
            'lr': .0002,
            'fine_tune': 1
        })

        self.rhs = rhs
        self.filename = filename
        self.model = None
        self.es = EarlyStopping(monitor='val_loss', mode='min', patience=3, min_delta=.0001)
        self.loaded = False
        self.init_model()

    def _dset(self, arr: np.ndarray, validation):
        split = int(arr.shape[0] * .7)
        return arr[split:] if validation else arr[:split]

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
            x = np.hstack([a, b])
            y = Similars(a, b).normalize().cosine(abs=True).value().diagonal()
            yield x, y

    def generator_adjustments(self, lhs, adjustments, batch_size, validation=False):
        mask = adjustments != 0
        rhs = self._dset(self.rhs[mask], validation)
        adjustments = self._dset(adjustments[mask], validation)
        while True:
            idx = permute(rhs)[:batch_size]
            b = rhs[idx]
            adj = adjustments[idx]

            # repeat lhs, cut off excess
            n_lhs, n_rhs = lhs.shape[0], b.shape[0]
            a = np.tile(lhs, (math.ceil(n_rhs/n_lhs), 1))[:n_rhs]

            x = np.hstack([a, b])
            y = Similars(a, b).normalize().cosine(abs=True).value().diagonal()
            # Push highly-rated docs up, low-rated docs down. Using negative-score because cosine DISTANCE
            # (less is better)
            # y = y - y.std() * adj
            y = y - adj
            yield x, y

    def generator_predict(self, batch_size, x):
        rhs = self.rhs
        for i in range(0, rhs.shape[0], batch_size):
            b = rhs[i:i+batch_size]
            a = np.repeat(x, b.shape[0], axis=0)
            yield np.hstack([a, b])

    def init_model(self, load=True):
        if load and self.filename and exists(self.filename):
            logger.info("DNN: pretrained model")
            self.model = load_model(self.filename)
            self.loaded = True
            return
        self.loaded = False

        h = self.hypers
        dims = self.rhs.shape[1] * 2
        input = Input(shape=(dims,))
        m = input
        last_dim = dims
        for i in [0,1,2]:
            d = h[f"l{i}"]
            if not d: continue
            d = math.ceil(d * last_dim)
            last_dim = d
            kwargs = dict(activation=h.act)
            kwargs['kernel_initializer'] = 'glorot_uniform' \
                if h.act == 'tanh' else 'he_uniform'
            m = Dense(d, **kwargs)(m)
            if h.bn: m = BatchNormalization()(m)
        m = Dense(1, activation=h.final)(m)
        m = Model(input, m)
        # http://zerospectrum.com/2019/06/02/mae-vs-mse-vs-rmse/
        # MAE because we _want_ outliers (user score adjustments)
        loss = 'binary_crossentropy' if h.final == 'sigmoid' else h.loss
        opt = Nadam if h.opt == 'nadam' else Adam
        m.compile(
            loss=loss,
            optimizer=opt(learning_rate=h.lr),
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
            epochs=self.hypers.fine_tune,  # too many epochs overfits (eg to CBT). Maybe adjust LR *down*, or other?
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
