from lefnire_ml_utils import Similars
from lefnire_ml_utils.fixtures import articles

corpus = articles()

def test_ae():
    chain = Similars(corpus).embed().normalize()
    vecs = chain.value()
    dims = 20
    reduced = chain.autoencode(dims=[400,20], preserve_cosine=True).value()
    assert vecs.shape[0] == reduced.shape[0]
    assert reduced.shape[1] == dims[-1]

    # TODO do some comparison between original cosines & new cosines
