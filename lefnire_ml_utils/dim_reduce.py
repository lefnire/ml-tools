"""
Copied from very old code, this won't work just yet. Need to fix up.
"""

import math, os, pdb
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Input, Dense, Lambda, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

def autoencode(
    x,
    latent=80,
    save_load_path='/storage/autoencoder.tf',
    preserve=None
):
    """
    Auto-encode X from input_dim to latent.
    :param x: embeddings to encode. Should already be normalized: Similars(x).normalize().value()
    :param latent: latent dims to embed
    :param save_load_path: if provided, will attempt to load this model for use. If not exists, will train the
        model & save here, for use next time
    :param preserve: Interesting param. If provided, AE will try to preserve whatever it is you pass here while
        learning the embeddings. Must be an array of X.shape[0]. Eg, if you wanted to preserve cosine distances
        you'd pre-compute the cosine distances & pass them here. preserve=Similars(x).normalize().cosine().value()
    """

    ###
    # Run the model
    K.clear_session()
    if save_load_path and os.path.exists(save_load_path):
        decoder = load_model(save_load_path)
        return decoder.predict(x)

    ###
    # Compile model
    input_dim = x.shape[1]
    preserve_ = preserve is not None
    x_input = Input(shape=(input_dim,), name='x_input')
    e1 = Dense(500, activation='elu')(x_input)
    e2 = Dense(150, activation='elu')(e1)
    # linear for no bounds on encoder (simplest)
    # tanh to force normalized, in case clusterer wants that. This makes most sense to me
    # elu just performs better, but no intuition
    e3 = Dense(latent, activation='tanh')(e2)
    e_last = e3

    if preserve_:
        x_other_input = Input(shape=(input_dim,), name='x_other_input')
        merged = Concatenate(1)([e_last, x_other_input])
    else:
        merged = e_last
    d1 = Dense(150, activation='elu')(merged)
    d2 = Dense(500, activation='elu')(d1)
    d3 = Dense(input_dim, activation='linear', name='decoder_out')(d2)
    d_last = d3

    d_in = [x_input]
    d_out, e_out = [d_last], [e_last]
    if preserve is not None:
        dist_out = Dense(1, activation='sigmoid', name='dist_out')(d_last)
        d_in.append(x_other_input)
        d_out.append(dist_out)
    decoder = Model(d_in, d_out)
    encoder = Model(x_input, e_out)

    loss, loss_weights = {'decoder_out': 'mse'}, {'decoder_out': 1.}
    if preserve_:
        loss['dist_out'] = 'binary_crossentropy'
        loss_weights['dist_out'] = 1.
    decoder.compile(
        # metrics=['accuracy'],
        loss=loss,
        loss_weights=loss_weights,
        optimizer=Adam(learning_rate=.0005),
    )
    decoder.summary()

    ###
    # Train model
    np.random.shuffle(x)  # shuffle all data first, since validation_split happens before shuffle

    shuffle = np.arange(x.shape[0])
    np.random.shuffle(shuffle)

    # https://wizardforcel.gitbooks.io/deep-learning-keras-tensorflow/content/8.2%20Multi-Modal%20Networks.html
    inputs = {'x_input': x}
    outputs = {'decoder_out': x}
    if preserve_:
        inputs['x_other_input'] = x[shuffle]
        outputs['dist_out'] = preserve[np.arange(preserve.shape[0]), shuffle]

    es = EarlyStopping(monitor='val_loss', mode='min', patience=3, min_delta=.0001)
    decoder.fit(
        inputs,
        outputs,
        epochs=100,
        batch_size=128,
        # journal entries + books, need them mixed up. [update] shuffled up to, since validation_split used
        # shuffle=True,
        callbacks=[es],
        validation_split=.3,
    )
    encoder.save(save_load_path)
    return encoder.predict(x)
