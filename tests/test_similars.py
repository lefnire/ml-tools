import pytest, pdb
import numpy as np
from ml_tools import Similars, cleantext
from ml_tools.fixtures import articles
from box import Box

corpus = articles()

split_ = len(corpus)//3
x, y = corpus[:split_], corpus[split_:]

chain = Box(
    x=Similars(x),
    xy=Similars(x, y)
)
emb = Box(
    x=chain.x.embed(),
    xy=chain.xy.embed()
)
xy_param = ("k", ("x", "xy"))

@pytest.mark.parametrize(*xy_param)
def test_embed(k):
    res = emb[k].value()
    if k == "x":
        res = [res]
    else:
        assert len(res) == 2
    for r in res:
        assert type(r) == np.ndarray
        assert r.shape[1] == 768

@pytest.mark.parametrize(*xy_param)
@pytest.mark.parametrize("abs", (True, False))
def test_cosine(k, abs):
    res = emb[k].normalize().cosine(abs=abs).value()
    assert type(res) == np.ndarray
    if abs:
        all_pos = (res > 0-1e-15).all()
        assert all_pos

@pytest.mark.parametrize(*xy_param)
@pytest.mark.parametrize("algo", ("agglomorative", "kmeans"))
def test_cluster(k, algo):
    chain = emb[k].normalize().cluster(algo=algo)
    res = chain.value()
    # print(res)
    if k == "xy":
        assert len(res) == 2
        assert len(res[1]) < len(y)
        assert len(chain.data.labels[1]) == len(y)
    else:
        assert len(chain.data.labels) == len(x)
        assert len(res) < len(x)


@pytest.mark.parametrize(*xy_param)
@pytest.mark.parametrize("algo", ("agglomorative", "kmeans"))
@pytest.mark.parametrize("n", (3, 4))
def test_cluster_not_enough(k, algo, n):
    if k == 'x':
        x_ = x[:n]
        c = Similars(x_)
    else:
        x_, y_ = x[:1], y[:n-1]
        c = Similars(x_, y_)
    c = c.embed().normalize().cluster(algo=algo)
    res = c.value()
    if k == "xy":
        assert len(res) == 2
        assert len(res[0]) == len(x_)
        assert len(res[1]) == len(y_)
        assert len(c.data.labels[1]) == len(y_)
        assert (c.data.labels[1] == 1).all()
    else:
        assert len(c.data.labels) == len(x_)
        assert (c.data.labels == 1).all()
        assert len(res) == len(x_)


# @pytest.mark.skip
# @pytest.mark.parametrize(
#     "path,k,algo,x,y",
#     [
#         ('/storage/tmp1.bin',10,None,corpus,None),
#         ('/storage/tmp2.bin',10,None,corpus,Y),
#         (None,10,None,corpus,None),
#         (None,10,None,corpus,Y),
#         ('/storage/tmp1.bin',10,'agglomorative',corpus,None),
#         ('/storage/tmp2.bin',10,'agglomorative',corpus,Y),
#         ('/storage/tmp2.bin',10,'kmeans',corpus,Y),
#         ('/storage/tmp1.bin',1,False,corpus,None),
#         ('/storage/tmp2.bin',1,False,corpus,Y),
#     ])
# def test_ann(path, k, algo, x, y):
#     def fn():
#         chain = Similars(x, y).embed()
#         if algo: chain = chain.cluster(algo=algo)
#         res = chain.ann(y_from_file=path, top_k=k).value()
#         print(res)
#         return res
#     if (algo and not k) or (y is None):
#         with pytest.raises(Exception): fn()
#     else:
#         fn()
#
#
# @pytest.mark.parametrize(
#     "x,y,algo,k",
#     [
#         (corpus,None,None,None),
#         (X,Y,None,None),
#         (corpus,None,'agglomorative',None),
#         (X,Y,'agglomorative',None),
#         # (corpus,None,'agglomorative',10),
#         (X,Y,'agglomorative',10),
#         # (corpus,None,'kmeans',10),
#         (X,Y,'kmeans',10),
#     ])
# def test_cosine(x,y,algo,k):
#     def fn():
#         chain = Similars(x, y).embed()
#         if algo: chain = chain.cluster(algo=algo)
#         res = chain.cosine(top_k=k).value()
#         print(res)
#         return res
#     if algo and not k:
#         with pytest.raises(Exception): fn()
#     else:
#         fn()
