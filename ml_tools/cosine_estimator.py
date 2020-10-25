# https://github.com/tensorflow/tensorflow/issues/2117
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD, Nadam, Adam

import pdb, logging, math, re
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from os.path import exists
from box import Box
from ml_tools import Similars
import numpy as np
from typing import List
logger = logging.getLogger(__name__)


def permute(arr: np.ndarray):
    return np.random.permutation(arr.shape[0])

class CosineEstimator:
    default_hypers = dict(
        layers=1,  # win=1
        l0=.65,  # win=.55-.8
        act='relu',  # win=relu
        loss='mae',  # win=mae
        batch=300,  # win=300
        bn=True,  # inconclusive (since win.layers=1)
        opt='nadam',  # winner=nadam (TODO try AdamW)
        lr=.0004,  # winner=.0004
        normalize=True,  # win=True
        shuffle=True,

        # The above are solid winners. Below are still work in progress / hard to optmize.
        # sw_mine, sw_other winners
        # 95.90945467591888, 0.03522867388231844
        # 94.29366495956107, 0.2479051508079999
        # 33.48236563125175, 0.135911997641221
        # 45.1694413229026, 0.13853704778654738
        sw_mine=75.,
        sw_other=1.15,
        std_mine=.3,
        std_other=.1
    )
    """
    Neural network that learns the cosine DISTANCE function (between 0-1, 0 being similar, 1 being distant). Also
    allows fine-tuning adjustments of those similarities, eg in the case of user-ratings on documents (embedded).
    """
    def __init__(
        self,
        lhs: np.ndarray,
        rhs: np.ndarray,
        adjustments: List = [],
        filename: str = None,
        hypers: dict = {}
    ):
        """

        :param lhs: left-hand-side, a small-ish corpus you want to find most similar documents to
        :param rhs: right-hand-side, a large database/corpus/index you want the most-similar documents from
        :param adjustments: a list of dict(amount,weight,values).
            * amount(float): the amount you want to adjust similarity-score by. Generally range .1-.3 (think in terms of
                standard deviations). .3 is a good number
            * weight(float): how much to weight the network on these adjusted values? Uses tf.sample_weight. Try 50.
            * values(np.ndarray): array of rows which will be multiplied by amount (values*amount). Make everything
                zero except for the rows you want adjusted, which should probably be 1. `arr=np.zeros(); arr[mask] = 1.`
        :param filename: if you want to save/load the trained model, specify a filename
        """
        self.hypers = Box({**CosineEstimator.default_hypers, **hypers})
        print(self.hypers)

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
        m = Dense(1, activation='linear')(m)
        m = Model(input, m)
        # http://zerospectrum.com/2019/06/02/mae-vs-mse-vs-rmse/
        # MAE because we _want_ outliers (user score adjustments)
        opt = Nadam(learning_rate=h.lr) if h.opt == 'nadam'\
            else Adam(learning_rate=h.lr, amsgrad=True) if 'amsgrad'\
            else SGD(lr=h.lr, momentum=0.9, decay=0.01, nesterov=True)
        m.compile(
            loss=h.loss,
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
        x, y = self.rhs, self.y

        sw = np.ones(y.shape[0])
        for adj in self.adjustments:
            vals = adj['values']
            y = y - vals * adj['amount']
            mask = vals != 0
            sw[mask] = np.maximum(sw[mask], adj['weight'])

        # Stratify-split based on adjustments; that is, try to ensure some adjustments present for each
        # training / validation batch.
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html
        labels = (sw != 1).astype(int)
        if labels.any():
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3)
            train_idx, val_idx = next(sss.split(x, labels))
            x_train, y_train, sw_train = x[train_idx], y[train_idx], sw[train_idx]
            x_val, y_val, sw_val = x[val_idx], y[val_idx], sw[val_idx]
            fit_args = dict(
                sample_weight=sw_train,
                validation_data=(x_val, y_val, sw_val),
                shuffle=h.shuffle  # 'batch' totally breaks accuracy!
            )
        else:
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.3, shuffle=True)
            fit_args = dict(
                validation_data=(x_val, y_val),
                shuffle=True
            )

        history = self.model.fit(
            x_train, y_train,
            epochs=30,
            callbacks=[self.es],
            batch_size=h.batch,
            **fit_args
        )
        self.loss = history.history['val_loss'][-1]
        if self.filename:
            self.model.save(self.filename)

    def predict(self):
        return self.model.predict(self.rhs, batch_size=200).squeeze()
