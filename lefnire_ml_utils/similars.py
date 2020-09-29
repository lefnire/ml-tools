import math, pdb
import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from kneed import KneeLocator
from sklearn.cluster import MiniBatchKMeans as KMeans
from sentence_transformers import SentenceTransformer
from typing import List, Union


# https://stackoverflow.com/a/7590709/362790
def chain(device_in=None):
    """
    Decorator for chaining methods in Similars like Similars(x, y).embed().normalize().cosine()
    When you want the output at any step, call .value(). It will retain its intermediate step
    so you can continue chaining later, and call subsequent .value()
    :param device_in: gpu|cpu|None. What device does this chain-step expect its values from?
    """
    def decorator(fn):
        def wrapper(self, *args, **kwargs):
            # Place x,y on device this chain method expects
            x, y = self.get_values(device_in)
            res = fn(self, x, y, *args, **kwargs)
            # Always maintain [x, y] for consistency
            if type(res) != list: res = [res, None]
            # Save intermediate result, and chained methods can continue
            return Similars(*res)
        return wrapper
    return decorator


class Similars(object):
    """
    Various similarity helper functions.

    * NLP methods: clean_text, tf_idf, embed (via sentence_transformers), etc
    Call like Similars(sentences).embed() or Similars(lhs, rhs).clean_text().tfidf()

    * Similarity methods: normalize, cosine, kmeans, agglomorative, etc
    Call like Similars(x, y).noramlize().cosine().agglomorative()

    Takes x, y. If y is provided, then we're comparing x to y. If y is None, then operations
    are pairwise on x (x compared to x).
    """
    def __init__(
        self,
        x: Union[List[str], np.ndarray],
        y: Union[List[str], np.ndarray] = None
    ):
        self.result = [x, y]

    def value(self):
        x, y = self.get_values('cpu')
        if y is None: return x
        return [x, y]

    def get_values(self, device=None):
        x, y = self.result
        if device == 'gpu':
            if not torch.is_tensor(x):
                x = torch.tensor(x)
            if y is not None and not torch.is_tensor(y):
                y = torch.tensor(y)
        elif device == 'cpu':
            if torch.is_tensor(x):
                x = x.cpu().numpy()
            if y is not None and torch.is_tensor(y):
                y = y.cpu().numpy()
        return [x, y]

    def _join(self, x, y):
        if y is None: return x
        if type(x) == list: return x + y
        if type(x) == np.ndarray: return np.vstack([x, y])
        if torch.is_tensor(x): return torch.cat((x, y), 0)

    def _split(self, joined, x, y):
        if y is None: return [joined, None]
        at = len(x) if type(x) == list else x.shape[0]
        return [joined[:at], joined[at:]]

    @chain()
    def embed(self, x: List[str], y: List[str]=None):
        enco = SentenceTransformer('roberta-base-nli-stsb-mean-tokens').encode(
            self._join(x, y),
            batch_size=32,
            show_progress_bar=True,
            convert_to_tensor=True
        )
        return self._split(enco, x, y)

    @chain()
    def dim_reduce(self, x, y): pass

    @chain()
    def tf_idf(self, x, y): pass

    @chain(device_in='gpu')
    def normalize(self, x, y):
        norm = self._join(x, y)
        norm = norm / norm.norm(dim=1)[:, None]
        return self._split(norm, x, y)

    @chain(device_in='gpu')
    def cosine(self, x, y, abs=False):
        """
        :param abs: Hierarchical clustering wants [0 1], and dists.sort_by(0->1), but cosine is [-1 1]. Set True to
            ensure cosine>0. Only needed currently for agglomorative()
        """
        if y is None:
            y = x
        sim = torch.mm(x, y.T)

        if abs:
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

    # Don't set device_in, in case cluster_both=True and we're getting gpu (if False, it's cpu)
    @chain()
    def agglomorative(self, x, y, cluster_both=False):
        """
        Agglomorative (hierarchical) clustering.
        :param cluster_both: if True, cluster x & y from the same pool & return [x_labels, y_labels]; otherwise just
            cluster x.
        """
        x_orig = x
        if cluster_both and y is not None:
            x = self._join(x, y)
        x = Similars(x).cosine(abs=True).value()
        nc = self._default_n_clusters(x)
        labels = AgglomerativeClustering(
            n_clusters=nc,
            affinity='precomputed',
            linkage='average'
        ).fit_predict(x)
        return self._split(labels, x_orig, y) if cluster_both else labels

    @chain(device_in='cpu')
    def kmeans(self, x, y, cluster_both=False):
        combo = self._join(x, y) if cluster_both else x

        # Code from https://github.com/arvkevi/kneed/blob/master/notebooks/decreasing_function_walkthrough.ipynb
        step = 2  # math.ceil(guess.max / 10)
        K = range(2, 40, step)
        scores = []
        for k in K:
            km = KMeans(n_clusters=k).fit(combo)
            scores.append(km.inertia_)
        S = 1  # math.floor(math.log(all.shape[0]))  # 1=default; 100entries->S=2, 8k->3
        kn = KneeLocator(list(K), scores, S=S, curve='convex', direction='decreasing', interp_method='polynomial')
        nc = kn.knee or self._default_n_clusters(combo)
        labels = KMeans(n_clusters=nc).fit(combo).labels_
        return self._split(labels, x, y) if cluster_both else labels
