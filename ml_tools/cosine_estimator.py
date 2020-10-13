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
from tqdm import tqdm
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
    def __init__(self, lhs, rhs, filename=None):
        """
        :param rhs: right-hand-side, ie the database/corpus/index you'll be comparing things against later. Usually
            this will be the much larger of the two matrices in a comparison.
        """
        self.hypers = Box({
            'l0': .5,
            'l1': .4,
            'l2': False,
            'act': 'tanh',
            'final': 'linear',
            'loss': 'mae',
            'batch': 128,
            'bn': False,
            'opt': 'adam',
            'lr': .0002,
            'fine_tune': 2
        })

        self.lhs = lhs
        self.rhs = rhs
        self.y = np.hstack([d for d in self.gen_dists()])
        self.filename = filename
        self.model = None
        self.loss = None
        self.es = EarlyStopping(monitor='val_loss', mode='min', patience=3, min_delta=.0001)
        self.loaded = False
        self.init_model()

    def gen_dists(self):
        lhs, rhs = self.lhs, self.rhs
        batch = 100
        for i in range(0, rhs.shape[0], batch):
            yield Similars(lhs, rhs[i:i+batch]).normalize()\
                .cosine(abs=True).value().min(axis=0).squeeze()

    def init_model(self, load=True):
        if load and self.filename and exists(self.filename):
            logger.info("DNN: pretrained model")
            self.model = load_model(self.filename)
            self.loaded = True
            return
        self.loaded = False

        h = self.hypers
        dims = self.rhs.shape[1]
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
        history = self.model.fit(
            self.rhs, self.y,
            epochs=50,
            callbacks=[self.es],
            batch_size=batch_size,
            shuffle=True,
            validation_split=.3
        )
        self.loss = history.history['val_loss'][-1]
        if self.filename:
            self.model.save(self.filename)

    def fit_adjustments(self, adjustments):
        if not adjustments.any(): return
        logger.info("DNN: learn adjustments function")
        y = self.y - adjustments
        mask = adjustments != 0
        rhs, y = self.rhs[mask], y[mask]

        batch_size = 16
        self.model.fit(
            rhs, y,
            batch_size=batch_size,
            epochs=self.hypers.fine_tune,  # too many epochs overfits (eg to CBT). Maybe adjust LR *down*, or other?
            # callbacks=[self.es],
            shuffle=True,
            validation_split=.3
        )

    def predict(self):
        return self.model.predict(self.rhs, batch_size=200).squeeze()
