import pytest, pdb
import numpy as np
from lefnire_ml_utils import Similars, cleantext
from lefnire_ml_utils.fixtures import articles

corpus = articles()

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

@pytest.mark.parametrize(
    "methods,x,y",
    [
        ([cleantext.keywords], corpus, None),
        ([cleantext.keywords], X, Y),
        ([cleantext.strip_html, cleantext.keywords], corpus, None),
        ([cleantext.strip_html, cleantext.keywords], X, Y),
    ])
def test_cleantext(methods, x, y):
    res = Similars(x, y).cleantext(methods).value()
    print(res)
    if y is None:
        assert len(res) == len(x)
    else:
        assert len(res) == 2
        assert len(res[1]) == len(y)


@pytest.mark.parametrize(
    "path,k,algo,x,y",
    [
        ('/storage/tmp1.bin',10,None,corpus,None),
        ('/storage/tmp2.bin',10,None,corpus,Y),
        (None,10,None,corpus,None),
        (None,10,None,corpus,Y),
        ('/storage/tmp1.bin',10,'agglomorative',corpus,None),
        ('/storage/tmp2.bin',10,'agglomorative',corpus,Y),
        ('/storage/tmp2.bin',10,'kmeans',corpus,Y),
        ('/storage/tmp1.bin',1,False,corpus,None),
        ('/storage/tmp2.bin',1,False,corpus,Y),
    ])
def test_ann(path, k, algo, x, y):
    def fn():
        chain = Similars(x, y).embed()
        if algo: chain = chain.cluster(algo=algo)
        res = chain.ann(y_from_file=path, top_k=k).value()
        print(res)
        return res
    if (algo and not k) or (y is None):
        with pytest.raises(Exception): fn()
    else:
        fn()


@pytest.mark.parametrize(
    "x,y,algo,k",
    [
        (corpus,None,None,None),
        (X,Y,None,None),
        (corpus,None,'agglomorative',None),
        (X,Y,'agglomorative',None),
        # (corpus,None,'agglomorative',10),
        (X,Y,'agglomorative',10),
        # (corpus,None,'kmeans',10),
        (X,Y,'kmeans',10),
    ])
def test_cosine(x,y,algo,k):
    def fn():
        chain = Similars(x, y).embed()
        if algo: chain = chain.cluster(algo=algo)
        res = chain.cosine(top_k=k).value()
        print(res)
        return res
    if algo and not k:
        with pytest.raises(Exception): fn()
    else:
        fn()


