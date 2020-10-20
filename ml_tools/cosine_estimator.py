# https://github.com/tensorflow/tensorflow/issues/2117
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD, Nadam

import pdb, logging, math, re
from os.path import exists
from box import Box
from ml_tools import Similars
import numpy as np
logger = logging.getLogger(__name__)


def permute(arr: np.ndarray):
    return np.random.permutation(arr.shape[0])

class CosineEstimator:
    """
    Neural network that learns the cosine DISTANCE function (between 0-1, 0 being similar, 1 being distant). Also
    allows fine-tuning adjustments of those similarities, eg in the case of user-ratings on documents (embedded).
    """
    def __init__(self, lhs, rhs, adjustments=None, filename=None):
        """
        :param rhs: right-hand-side, ie the database/corpus/index you'll be comparing things against later. Usually
            this will be the much larger of the two matrices in a comparison.
        """
        self.hypers = Box(
            layers=1,
            l0=.32,
            act='elu',
            final='linear',
            loss='mse',
            batch=224,
            bn=True,
            opt='nadam',
            lr=.0004,
            sample_weight=2.,
            std_mine=.183,
            std_other=.307,
            normalize=True
        )

        if self.hypers.normalize:
            lhs, rhs = Similars(lhs, rhs).normalize().value()

        self.lhs = lhs
        self.rhs = rhs
        self.y = np.hstack([d for d in self.gen_dists()])
        self.adjustments = adjustments
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
            c = Similars(lhs, rhs[i:i+batch])
            if not self.hypers.normalize:
                # don't double-normalize; but do normalize here if not already done
                c = c.normalize()
            yield c.cosine(abs=True).value().min(axis=0).squeeze()

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
        for i in range(h.layers):
            d = math.ceil(h[f"l{i}"] * last_dim)
            last_dim = d
            kwargs = dict(activation=h.act)
            kwargs['kernel_initializer'] = 'glorot_uniform' \
                if h.act == 'tanh' else 'he_uniform'
            m = Dense(d, **kwargs)(m)
            # Don't apply batchnorm to the last hidden layer
            if h.bn and i < h.layers-1:
                m = BatchNormalization()(m)
        m = Dense(1, activation=h.final)(m)
        m = Model(input, m)
        # http://zerospectrum.com/2019/06/02/mae-vs-mse-vs-rmse/
        # MAE because we _want_ outliers (user score adjustments)
        loss = 'binary_crossentropy' if h.final == 'sigmoid' else h.loss
        opt = Nadam(learning_rate=h.lr) if h.opt == 'nadam'\
            else SGD(lr=h.lr, momentum=0.9, decay=0.01, nesterov=True)
        m.compile(
            loss=loss,
            optimizer=opt,
        )
        m.summary()
        self.model = m


    def fit(self):
        if self.loaded:
            logger.info("DNN: using cosine-pretrained")
            return
        else:
            logger.info("DNN: learn cosine function")

        h = self.hypers

        shuff = permute(self.y)
        x, y = self.rhs[shuff], self.y[shuff]

        extra = {}
        sample_weight, adjustments = h.sample_weight, self.adjustments
        if sample_weight and adjustments is not None and adjustments.any():
            logger.info("Using sample weight")
            adjustments = adjustments[shuff]
            y = y - adjustments
            if h.final == 'sigmoid':
                y = np.clip(y, 0., 1.)
            sw = np.ones(y.shape[0])
            sw[adjustments != 0] = sample_weight
            extra['sample_weight'] = sw

        history = self.model.fit(
            x, y,
            **extra,
            epochs=50,
            callbacks=[self.es],
            batch_size=h.batch,
            shuffle=True,
            validation_split=.3
        )
        self.loss = history.history['val_loss'][-1]
        if self.filename:
            self.model.save(self.filename)

    def predict(self):
        return self.model.predict(self.rhs, batch_size=200).squeeze()
