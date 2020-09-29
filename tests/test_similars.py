import pytest
import numpy as np
from lefnire_ml_utils import Similars

corpus = ['A man is eating food.',
          'A man is eating a piece of bread.',
          'A man is eating pasta.',
          'The girl is carrying a baby.',
          'The baby is carried by the woman',
          'A man is riding a horse.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'Someone in a gorilla costume is playing a set of drums.',
          'A cheetah is running behind its prey.',
          'A cheetah chases prey on across a field.'
          ] * 5

split_ = len(corpus)//3
X, Y = corpus[:split_], corpus[split_:]


@pytest.mark.parametrize(
    "chain_step,x,y",
    [
        (1, corpus, None),
        (1, X, Y),
        (2, corpus, None),
        (2, X, Y),
        (3, corpus, None),
        (3, X, Y)
    ])
def test_similars(chain_step, x, y):
    res = Similars(x, y).embed()
    if chain_step > 1: res = res.normalize()
    if chain_step > 2: res = res.cosine()
    res = res.value()
    if chain_step == 3: return  # will test normalize below
    if y is None:
        res = [res]
    else:
        assert len(res) == 2
    for r in res:
        assert type(r) == np.ndarray
        assert r.shape[1] == 768

@pytest.mark.parametrize(
    "abs,x,y",
    [
        (True, corpus, None),
        (True, X, Y),
        (False, corpus, None),
        (False, X, Y),
    ])
def test_normalize(abs, x, y):
    res = Similars(x, y).embed().normalize().cosine(abs=abs).value()
    assert type(res) == np.ndarray
    if abs:
        all_pos = (res > 0-1e-15).all()
        assert all_pos


@pytest.mark.parametrize(
    "algo,cluster_both,x,y",
    [
        ('kmeans', True, corpus, None),
        ('kmeans', True, X, Y),
        ('kmeans', False, corpus, None),
        ('kmeans', False, X, Y),
        ('agglomorative', True, corpus, None),
        ('agglomorative', True, X, Y),
        ('agglomorative', False, corpus, None),
        ('agglomorative', False, X, Y),
    ])
def test_cluster(algo, cluster_both, x, y):
    res = Similars(x, y).embed().normalize().cluster(algo=algo, cluster_both=cluster_both).value()
    print(res)
    if cluster_both and y:
        assert len(res) == 2
        assert len(res[1]) == len(y)
    else:
        assert len(res) == len(x)
