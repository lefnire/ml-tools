import math, pdb, os
import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from kneed import KneeLocator
from sklearn.cluster import MiniBatchKMeans as KMeans
from sentence_transformers import SentenceTransformer, util
from typing import List, Union
from sklearn.metrics import pairwise_distances, pairwise_distances_chunked
from scipy.spatial.distance import jensenshannon
from sklearn.feature_extraction.text import TfidfVectorizer
from . import cleantext
from .dim_reduce import autoencode

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
            last_fn = fn.__name__
            if last_fn == 'cluster':
                return Similars(x, y, last_fn=last_fn, labels=res)
            return Similars(*res, last_fn=last_fn)
        return wrapper
    return decorator


class Similars(object):
    """
    Various similarity helper functions.

    * NLP methods: cleantext, tf_idf, embed (via sentence_transformers), etc
    Call like Similars(sentences).embed() or Similars(lhs, rhs).cleantext().tfidf()

    * Similarity methods: normalize, cosine, kmeans, agglomorative, etc
    Clustering: Similars(x, y).normalize().cluster(algo='agglomorative')
    Similarity: Similars(x).normalize.cosine()  (then sort low to high)

    Takes x, y. If y is provided, then we're comparing x to y. If y is None, then operations
    are pairwise on x (x compared to x).
    """
    def __init__(
        self,
        x: Union[List[str], np.ndarray],
        y: Union[List[str], np.ndarray] = None,
        last_fn: str = None,
        labels: List[np.ndarray] = None
    ):
        self.result = [x, y]
        self.last_fn = last_fn
        self.labels = labels

    def value(self):
        x, y = self.labels if self.last_fn == 'cluster'\
            else self.get_values('cpu')
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
            batch_size=16,
            show_progress_bar=True,
            convert_to_tensor=True
        )
        return self._split(enco, x, y)

    @chain()
    def dim_reduce(self, x, y, method='autoencoder'):
        """
        :param method: autoencoder|pca|tsne
        """
        raise Exception("dim_reduce not yet implemented. I'll add basic AutoEncoder soon")

    @chain()
    def cleantext(self, x, y, methods=[cleantext.keywords]):
        combo = self._join(x, y)
        combo = cleantext.multiple(combo, methods)
        return self._split(combo, x, y)

    @chain()
    def tf_idf(self, x, y):
        combo = self._join(x, y)
        combo = TfidfVectorizer().fit_transform(combo)
        return self._split(combo, x, y)

    @chain(device_in='gpu')
    def normalize(self, x, y):
        norm = self._join(x, y)
        norm = norm / norm.norm(dim=1)[:, None]
        return self._split(norm, x, y)

    @staticmethod
    def _unsqueeze(t):
        if len(t.shape) > 1: return t
        return t.unsqueeze(0)

    def _sims_by_clust(self, x, top_k, fn):
        assert torch.is_tensor(x), "_sims_by_clust written assuming GPU in, looks like I was wrong & got a CPU val; fix this"
        assert top_k, "top_k must be specified if using clusters in similarity functions"
        labels = self.labels[0]
        res = []
        for l in range(labels.max()):
            mask = (labels == l)
            if mask.sum() == 0: continue
            k = math.ceil(mask.sum() / top_k)
            x_ = x[mask].mean(0)
            r = fn(x_, k)
            res.append(torch.cat((r.values, r.indices), 1))
        return torch.stack(res)

    def _cosine(self, x, y, abs=False, top_k=None):
        x = self._unsqueeze(x)
        if y is None: y = x
        else: y = self._unsqueeze(y)

        sim = torch.mm(x, y.T)

        if abs:
            # print("sim.min=", sim.min(), "sim.max=", sim.max())
            dist = (sim - 1).abs()
            # See https://stackoverflow.com/a/63532174/362790 for other options
            # dist = sim.acos() / np.pi
            # dist = 1 - (sim + 1) / 2
        else:
            dist = 1. - sim
        if top_k is None: return dist
        return torch.topk(dist, min(top_k, dist.shape[1] - 1), dim=1, largest=False, sorted=False)

    @chain(device_in='gpu')
    def cosine(self, x, y, abs=False, top_k=None):
        """
        :param abs: Hierarchical clustering wants [0 1], and dists.sort_by(0->1), but cosine is [-1 1]. Set True to
            ensure cosine>0. Only needed currently for agglomorative()
        :param top_k: only return top-k smallest distances. If you cluster() before this cosine(), top_k is required.
            It will return (n_docs_in_cluster/top_k) per cluster.
        """
        if self.labels is None:
            res = self._cosine(x, y, abs=abs)
            if not top_k: return res

        def fn(x_, k):
            return self._cosine(x_, y, abs=abs, top_k=k)
        return self._sims_by_clust(x, top_k, fn)

    @chain(device_in='cpu')
    def ann(self, x, y, y_from_file=None, top_k=None):
        """
        Adapted from https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/semantic_search_quora_hnswlib.py
        Finds top-k similar y similar embeddings to x.
        :param y_from_file: if provided, will attempt to load from this path. If fails, will train index & save to
            this path, to be loaded next time
        :param top_k: how many results per x-row to return? If 1, just find closest match per row.
            cluster-mean
        """
        import hnswlib
        if y is None: raise Exception("y required; it's the index you query")
        if y_from_file and os.path.exists(y_from_file):
            index = hnswlib.Index(space='cosine', dim=x.shape[1])
            index.load_index(y_from_file)
        else:
            # Defining our hnswlib index
            # We use Inner Product (dot-product) as Index. We will normalize our vectors to unit length, then is Inner Product equal to cosine similarity
            index = hnswlib.Index(space='cosine', dim=y.shape[1])

            ### Create the HNSWLIB index
            print("Start creating HNSWLIB index")
            # UKPLab tutorial used M=16, but https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md suggests 64
            # for word embeddings (though these are sentence-embeddings)
            index.init_index(max_elements=y.shape[0], ef_construction=200, M=64)

            # Then we train the index to find a suitable clustering
            index.add_items(y, np.arange(y.shape[0]))
            index.save_index(y_from_file)

        # Controlling the recall by setting ef:
        # ef = 50
        ef = max(top_k + 1, min(1000, index.get_max_elements()))
        index.set_ef(ef)  # ef should always be > top_k_hits

        def fn(x_, k):
            # We use hnswlib knn_query method to find the top_k_hits
            return index.knn_query(x_, k)

        if self.labels is None:
            return fn(x, top_k)
        return self._sims_by_clust(x, top_k, fn)

    def jensenshannon(self, x, y):
        # TODO ignores y for now, x expected to be square tf-idf matrix. Probably doesn't work currently
        return pairwise_distances(x, metric=jensenshannon)

    def _default_n_clusters(self, x):
        return math.floor(1 + 3.5 * math.log10(x.shape[0]))

    # Don't set device_in, in case algo=agg & cluster_both=True
    @chain()
    def cluster(self, x, y, algo='agglomorative', cluster_both=False):
        """
        :param cluster_both: if True, cluster x & y from the same pool & return [x_labels, y_labels];
            otherwise just cluster x.
        """
        combo = x
        if cluster_both and y is not None:
            combo = self._join(x, y)
        combo = Similars(combo)
        if algo == 'agglomorative':
            combo = combo.cosine(abs=True).value()
            nc = self._default_n_clusters(combo)
            labels = AgglomerativeClustering(
                n_clusters=nc,
                affinity='precomputed',
                linkage='average'
            ).fit_predict(combo)
        elif algo == 'kmeans':
            combo = combo.value()
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
        else:
            raise Exception("Other clusterers not yet supported (use kmeans|agglomorative)")
        return self._split(labels, x, y) if cluster_both else labels

    @chain(device_in='cpu')
    def autoencode(self, x, y, latent=80, save_load_path='/storage/autoencoder.tf', preserve=None):
        assert y is None, "Don't pass y into autoencode (FIXME)"
        return autoencode(x, latent, save_load_path, preserve)
