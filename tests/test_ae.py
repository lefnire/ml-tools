from ml_tools import Similars
from ml_tools.fixtures import articles
import numpy as np

corpus = articles()

def test_ae():
    chain = Similars(corpus).embed()
    vecs = chain.value()

    orig_cosines = chain.normalize().cosine().value()
    orig_cosines = np.argsort(orig_cosines, axis=1)

    dims = 20
    reduced = chain.autoencode(dims=[400,20]).value()
    assert vecs.shape[0] == reduced.shape[0]
    assert reduced.shape[1] == dims[-1]

    # TODO do some comparison between original cosines & new cosines
