import math, pdb
import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from kneed import KneeLocator
from sklearn.cluster import MiniBatchKMeans as KMeans
from sentence_transformers import SentenceTransformer
from typing import List, Union


# https://stackoverflow.com/a/7590709/362790
def chain(device_in=None, device_out=None):
    def decorator(fn):
        def wrapper(self, *args, **kwargs):
            x, y = self.get_values(device_in)
            self.device_out = device_out
            res = fn(self, x, y, *args, **kwargs)
            if type(res) != list: res = [res, None]
            self.result = res
            return self
        return wrapper
    return decorator


class Similars(object):
    def __init__(self):
        self.result = [None, None]
        self.device_out = None

    def value(self):
        x, y = self.get_values(self.device_out)
        if y is None: return x
        return [x, y]

    def get_values(self, device=None):
        x, y = self.result
        if device == 'gpu':
            if not torch.is_tensor(x):
                x = torch.tensor(x)
            if y is not None and not torch.is_tensor(y):
                y = torch.tensor(y)
            return [x, y]
        if device == 'cpu':
            if torch.is_tensor(x):
                x = x.cpu()
            if y is not None and torch.is_tensor(y):
                y = y.cpu()
            return [x, y]
        return [x, y]

    def start(
        self,
        x: Union[List[str], np.array],
        y: Union[List[str], np.array]=None
    ):
        self.result = [x, y]
        return self

    @chain()
    def embed(self, x: List[str], y: List[str]=None):
        sentences = x if y is None else x + y
        m = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
        enco = m.encode(sentences, batch_size=32, show_progress_bar=True, convert_to_tensor=True)
        if y is None:
            x = enco
        else:
            x, y = enco[:len(x)], enco[len(x):]
        return [x, y]

    @chain()
    def dim_reduce(self, x, y): pass

    @chain()
    def tf_idf(self, x, y): pass

    @chain(device_in='gpu', device_out='cpu')
    def normalize(self, x, y):
        norm = x if y is None else torch.cat((x, y), 0)
        norm = norm / norm.norm(dim=1)[:, None]
        if y is None:
            x = norm
        else:
            x, y = norm[:x.shape[0]], norm[x.shape[0]:]
        return [x, y]

    @chain(device_in='gpu', device_out='cpu')
    def cosine(self, x, y, abs=False):
        if y is None:
            y = x
        sim = torch.mm(x, y.T)

        if abs:
            # Hierarchical clustering wants [0 1], and dists.sort_by(0->1), but cosine is [-1 1]
            # print("sim.min=", sim.min(), "sim.max=", sim.max())
            dist = (sim - 1).abs()
            # See https://stackoverflow.com/a/63532174/362790 for other options
            # dist = sim.acos() / np.pi
            # dist = 1 - (sim + 1) / 2
        else:
            dist = 1. - sim
        return dist

    def _default_n_clusters(self, x):
        return math.floor(1 + 3.5 * math.log10(x.shape[0]))

    @chain(device_in='gpu')
    def agglomorative(self, x, y):
        if y is not None:
            self.result = [torch.cat((x, y), 0), None]
        self.cosine(abs=True)
        all = self.result[0].cpu()
        nc = self._default_n_clusters(all)
        labels = AgglomerativeClustering(
            n_clusters=nc,
            affinity='precomputed',
            linkage='average'
        ).fit_predict(all)
        if y is not None:
            return [labels[:x.shape[0]], labels[x.shape[0]:]]
        return [labels, None]

    @chain(device_in='cpu')
    def kmeans(self, x, y):
        all = x
        if y is not None:
            all = np.vstack([x, y])

        # Code from https://github.com/arvkevi/kneed/blob/master/notebooks/decreasing_function_walkthrough.ipynb
        step = 2  # math.ceil(guess.max / 10)
        K = range(2, 40, step)
        inertias = []
        for k in K:
            km = KMeans(n_clusters=k).fit(all)
            inertias.append(km.inertia_)
        S = 1  # math.floor(math.log(all.shape[0]))  # 1=default; 100entries->S=2, 8k->3
        kn = KneeLocator(list(K), inertias, S=S, curve='convex', direction='decreasing', interp_method='polynomial')
        nc = kn.knee or self._default_n_clusters(all)
        labels = KMeans(n_clusters=nc).fit(all).labels_
        if y is not None:
            return [labels[:x.shape[0]], labels[x.shape[0]:]]
        return [labels, None]
