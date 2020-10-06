# https://github.com/tensorflow/tensorflow/issues/2117
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

import math, os, pdb, gc
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Input, Dense, Lambda, Concatenate, BatchNormalization, Reshape
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import logging
logger = logging.getLogger(__name__)


def autoencode_(x, dims, filename, batch_norm):
    if filename and os.path.exists(filename):
        encoder = load_model(filename)
        return encoder.predict(x)

    ###
    # Model

    # activation intuitions:
    # linear + mse: no bounds, let DNN learn what it learns - simplest
    # tanh/sigmoid + (mse?)/binary_cross_entropy: constrain output to [-1 1] / [0 1]. Esp. useful if using output in
    #   downstream tasks like cosine() or DNN which want normalized outputs.
    # elu + mse: seems to perform best, but I don't have intuition
    encode_act = 'tanh'
    dist_act = 'sigmoid'
    act = 'relu'

    input_dim = x.shape[1]

    ## Encoder
    encoder = Sequential()
    if batch_norm: encoder.add(BatchNormalization())
    for d in dims[:-1]:
        encoder.add(Dense(d, activation=act))
    encoder.add(Dense(dims[-1], activation=encode_act))
    # want encodings to be normalized for downstream. I think tanh achieves this goal, so if not present add BN
    if encode_act != 'tanh': encoder.add(BatchNormalization())

    ## Decoder
    decoder = Sequential()
    for d in dims[::-1][1:]:
        decoder.add(Dense(d, activation=act))
    decoder.add(Dense(input_dim, activation='linear', name='decoder'))

    ## Siamese
    # https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d
    # https://medium.com/@prabhnoor0212/siamese-network-keras-31a3a8f37d04
    CosSim = Lambda(lambda t: tf.keras.losses.cosine_similarity(t[0], t[1]))
    # FIXME without Reshape, it's not exporting shape and causes error
    Diff = Lambda(lambda t: Reshape((-1,1))(K.abs(t[0] - t[1])))

    input_a = Input(shape=(input_dim,), name='input_a')
    input_b = tf.random.shuffle(input_a)
    orig_sims = CosSim([input_a, input_b])
    enco_a, enco_b = encoder(input_a), encoder(input_b)
    deco_a, deco_b = decoder(enco_a), decoder(enco_b)

    enco_sims = CosSim([enco_a, enco_b])
    dist = Diff([enco_sims, orig_sims])
    dist = Dense(1, activation=dist_act, name='dist')(dist)

    # the loss is internal (cos(emb) v cos(orig)), ignore y_true
    dist_loss = lambda y_true, y_pred: y_pred

    ## Compile
    encoder = Model(input_a, enco_a)
    full = Model(input_a, [deco_a, deco_b, dist])

    loss = dict(
        sequential_1='mse',
        # sequential_1_1='mse',  # ignore, since we don't have the shuffle index order outside tensorflow
        dist=dist_loss
    )
    # loss_weights = {'decoder': 1., 'dist': 1.}
    # BatchNormalization allows much higher learning rates.
    lr = .001 if batch_norm else .0003
    full.compile(
        metrics=loss,
        loss=loss,
        # loss_weights=loss_weights,
        optimizer=Adam(learning_rate=lr),
    )
    full.summary()

    ###
    # Train
    # https://wizardforcel.gitbooks.io/deep-learning-keras-tensorflow/content/8.2%20Multi-Modal%20Networks.html
    # TODO use generator, x can be very large https://mc.ai/train-keras-model-with-large-dataset-batch-training/
    es = EarlyStopping(monitor='val_loss', mode='min', patience=3, min_delta=.0002)
    full.fit(
        dict(input_a=x),
        dict(
            # my names ('encoder' 'decoder') don't seem picked up, since siamese?
            sequential_1=x,
            # the loss is internal (cos(emb) v cos(orig)), so pass in junk which is ignored (y_true required)
            dist=np.zeros(x.shape[0])
        ),
        epochs=50,
        batch_size=128,
        shuffle=True,
        callbacks=[es],
        validation_split=.3,
    )
    encoder.save(filename)

    ###
    # Predict
    return encoder.predict(x)

def autoencode(
    x,
    dims=[500, 150, 20],
    filename=None,
    batch_norm=True
):
    """
    Auto-encode X from input_dim to latent, while preserving cosine similarity
    :param x: embeddings to encode. Don't need to pre-normalize, see batch_norm
    :param dims: ae architecture, with last value being latent dim
    :param filename: if provided, will attempt to load this model for use. If not exists, will train the
        model & save here, for use next time
    :param batch_norm: Whether to batch-normalize the input. It's a learned layer, so you'd be able to then use this
        trained model later without needing to normalize future inputs. I'm trying with False, hoping the DNN itself
        learns normalization in the process. Note, the embedding layer (the AE output) itself is normalized, since it's
        tanh; so this has less to do with downstream use, and more about training theory.
    """

    # Wrap function call so all Keras models lose context for garbage-collection. It doesn't work, Keras and its
    # memory leaks... but hey, worth the try.
    preds = autoencode_(x, dims, filename, batch_norm)
    gc.collect()
    K.clear_session()
    return preds
