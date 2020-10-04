"""
Copied from very old code, this won't work just yet. Need to fix up.
"""

import math, os, pdb
import numpy as np
from keras import backend as K
from keras.layers import Layer, Input, Dense, Lambda
from keras.layers.merge import concatenate
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn import preprocessing as pp
from sklearn.metrics import pairwise_distances_chunked, pairwise_distances

class AE():
    model_path = 'tmp/ae.tf'

    def __init__(self,
        input_dim=768,
        latent=20,
        preserve_cosine=False  # Reduce embeddings while trying to maintain their cosine similarities
    ):
        K.clear_session()
        self.input_dim = input_dim
        self.latent = latent
        self.preserve_cosine = preserve_cosine

        self.loaded = False
        self.init_model()
        self.load()

    def init_model(self):
        x_input = Input(shape=(self.input_dim,), name='x_input')
        e1 = Dense(500, activation='elu')(x_input)
        e2 = Dense(150, activation='elu')(e1)
        # linear for no bounds on encoder (simplest)
        # tanh to force normalized, in case clusterer wants that. This makes most sense to me
        # elu just performs better, but no intuition
        e3 = Dense(self.latent, activation='tanh')(e2)
        e_last = e3

        if self.preserve_cosine:
            x_other_input = Input(shape=(self.input_dim,), name='x_other_input')
            merged = concatenate([e_last, x_other_input])
        else:
            merged = e_last
        d1 = Dense(150, activation='elu')(merged)
        d2 = Dense(500, activation='elu')(d1)
        d3 = Dense(self.input_dim, activation='linear', name='decoder_out')(d2)
        d_last = d3

        d_in = [x_input]
        d_out, e_out = [d_last], [e_last]
        if self.preserve_cosine:
            dist_out = Dense(1, activation='sigmoid', name='dist_out')(d_last)
            d_in.append(x_other_input)
            d_out.append(dist_out)
        decoder = Model(d_in, d_out)
        encoder = Model(x_input, e_out)

        loss, loss_weights = {'decoder_out': 'mse'}, {'decoder_out': 1.}
        if self.preserve_cosine:
            loss['dist_out'] = 'binary_crossentropy'
            loss_weights['dist_out'] = 1.
        decoder.compile(
            # metrics=['accuracy'],
            loss=loss,
            loss_weights=loss_weights,
            optimizer=Adam(learning_rate=.0005),
        )
        decoder.summary()

        self.decoder, self.encoder = decoder, encoder

    def fit(self, x):
        # x = pp.normalize(x)  # assumed to be done before this via chain.normalize()
        np.random.shuffle(x)  # shuffle all data first, since validation_split happens before shuffle

        shuffle = np.arange(x.shape[0])
        np.random.shuffle(shuffle)

        if self.preserve_cosine:
            print("Calc distances")
            dists = []
            pdc = pairwise_distances_chunked(x, metric='cosine', working_memory=64)
            for i, chunk in enumerate(pdc):
                sz = chunk.shape[0]
                start, stop = i * sz, (i + 1) * sz
                dist = chunk[np.arange(sz), shuffle[start:stop]]
                dists.append(dist)
            # cosine values bw [-1 1], no loss function for that (well, mse..) Scale to [0 1] and use binary_xentropy
            dists = np.concatenate(dists)
            dists = pp.minmax_scale(dists) # (dists + 1) / 2

        # https://wizardforcel.gitbooks.io/deep-learning-keras-tensorflow/content/8.2%20Multi-Modal%20Networks.html
        inputs = {'x_input': x}
        outputs = {'decoder_out': x}
        if self.preserve_cosine:
            inputs['x_other_input'] = x[shuffle]
            outputs['dist_out'] = dists

        es = EarlyStopping(monitor='val_loss', mode='min', patience=3, min_delta=.0001)
        self.decoder.fit(
            inputs,
            outputs,
            epochs=100,
            batch_size=128,
            # journal entries + books, need them mixed up. [update] shuffled up to, since validation_split used
            # shuffle=True,
            callbacks=[es],
            validation_split=.3,
        )
        self.decoder.save_weights(self.model_path)

    def load(self):
        if os.path.exists(self.model_path + '.index'):
            self.decoder.load_weights(self.model_path)
            self.loaded = True
