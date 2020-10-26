import math, pdb, os
import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from kneed import KneeLocator
from sklearn.cluster import MiniBatchKMeans as KMeans
from sentence_transformers import SentenceTransformer, util
from typing import List, Union
from sklearn.metrics import pairwise_distances, silhouette_score
from scipy.spatial.distance import jensenshannon
from sklearn.feature_extraction.text import TfidfVectorizer
from .autoencoder import autoencode
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from box import Box

# https://stackoverflow.com/a/7590709/362790
def chain(device_in=None, keep=None, together=False):
    """
    Decorator for chaining methods in Similars like Similars(x, y).embed().normalize().cosine()
    When you want the output at any step, call .value(). It will retain its intermediate step
    so you can continue chaining later, and call subsequent .value()
    :param device_in: gpu|cpu|None. What device does this chain-step expect its values from?
    :param keep: keep data across chains by this key. chain.data[key]
    :param together: whether to process x & y as a whole, then split back apart (eg, tf-idf)
    """
    def decorator(fn):
        def wrapper(self, *args, **kwargs):
            # Place x,y on device this chain method expects
            x, y = self.get_values(device_in)
            x_y = (self._join(x, y),) if together else (x, y)
            res = fn(self, *x_y, *args, **kwargs)

            data = self.data
            if keep:
                data[keep] = res[-1]
                res = res[0]

            if together:
                res = self._split(res, x, y)

            # Always maintain [x, y] for consistency
            if type(res) != list: res = [res, None]
            # Save intermediate result, and chained methods can continue

            return self.__class__(*res, last_fn=fn.__name__, data=data)
        return wrapper
    return decorator


class Similars:
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
        last_fn = None,
        data=Box(default_box=True, default_box_attr=None)
    ):
        self.result = [x, y]
        self.last_fn = last_fn
        self.data = data

    def value(self):
        x, y = self.get_values('cpu', unsqueeze=False)
        if y is None: return x
        return [x, y]

    def get_values(self, device=None, unsqueeze=True):
        x, y = self.result
        if device is None:
            return x, y
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
        else:
            raise Exception("Device must be (gpu|cpu)")
        if unsqueeze:
            x, y = self._unsqueeze(x), self._unsqueeze(y)
        return x, y

    @staticmethod
    def _join(x, y):
        if y is None: return x
        if type(x) == list: return x + y
        if type(x) == np.ndarray: return np.vstack([x, y])
        if torch.is_tensor(x): return torch.cat((x, y), 0)

    @staticmethod
    def _split(joined, x, y):
        if y is None: return [joined, None]
        at = len(x) if type(x) == list else x.shape[0]
        return [joined[:at], joined[at:]]

    @staticmethod
    def _unsqueeze(t):
        if t is None: return t
        if len(t.shape) > 1: return t
        return t.unsqueeze(0)

    @chain(together=True)
    def embed(self, both: List[str], batch_size=32):
        return SentenceTransformer('roberta-base-nli-stsb-mean-tokens').encode(
            both,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True
        )

    @chain(together=True)
    def pca(self, both, **kwargs):
        return PCA(**kwargs).fit_transform(both)

    @chain(together=True)
    def tf_idf(self, both):
        return TfidfVectorizer().fit_transform(both)

    @chain(device_in='gpu', together=True)
    def normalize(self, both):
        return both / both.norm(dim=1)[:, None]

    def _sims_by_clust(self, x, top_k, fn):
        assert torch.is_tensor(x), "_sims_by_clust written assuming GPU in, looks like I was wrong & got a CPU val; fix this"
        assert top_k, "top_k must be specified if using clusters in similarity functions"
        labels = self.data.labels[0]
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
        if y is None: y = x
        sim = torch.mm(x, y.T)

        if abs:
            # See https://stackoverflow.com/a/63532174/362790 for the various options
            # print("sim.min=", sim.min(), "sim.max=", sim.max())
            eps = np.finfo(float).eps
            dist = sim.clamp(-1+eps, 1-eps).acos() / np.pi
            # dist = (sim - 1).abs()  # <-- used this before, ends up in 0-2 range
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
        if self.last_fn != 'cluster':
            res = self._cosine(x, y, abs=abs)
            if not top_k: return res

        def fn(x_, k):
            return self._cosine(x_, y, abs=abs, top_k=k)
        return self._sims_by_clust(x, top_k, fn)

    @chain(device_in='cpu')
    def cdist(self, x, y, **kwargs):
        return cdist(x, y, **kwargs)

    @chain(device_in='cpu')
    def ann(self, x, y, y_from_file=None, top_k=None):
        """
        Adapted from https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/semantic_search_quora_hnswlib.py
        Finds top-k similar y similar embeddings to x.
        Make sure you call .normalize() before this step!
        :param y_from_file: if provided, will attempt to load from this path. If fails, will train index & save to
            this path, to be loaded next time
        :param top_k: how many results per x-row to return? If 1, just find closest match per row.
            cluster-mean
        """
        try:
            import hnswlib
        except:
            raise Exception("hnswlib not installed; install it manually yourself via `pip install hnswlib`")
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

        if self.data.labels is None:
            return fn(x, top_k)
        return self._sims_by_clust(x, top_k, fn)

    @chain(device_in='cpu')
    def jensenshannon(self, x, y):
        if y is None:
            return pairwise_distances(x, metric=jensenshannon)
        else:
            return cdist(x, y, metric='jensenshannon')

    # Don't set device_in, in case algo=agg & cluster_both=True
    @chain(keep='labels', device_in='cpu')
    def cluster(self, x, y, algo='agglomorative'):
        """
        Clusters x, returning the cluster centroids and saving .data.labels away for later use.
        If y is provided, x & y get clustered together from the same cluster-pool, then split. Make sure
        this is what you want, otherwise cluster x separately on a different chain
        """
        both = self._join(x, y)
        n = both.shape[0]

        # Number entries < 4 not enough to cluster from, return as-is
        if n < 5:
            if y is None:
                return x, np.ones(n).astype(int)
            else:
                x, y = self._split(both, x, y)
                return (x, y), (np.ones(x.shape[0]).astype(int), np.ones(y.shape[0]).astype(int))

        if algo == 'agglomorative':
            model = AgglomerativeClustering
            model_args = dict(affinity='precomputed', linkage='average')
            sil_args = dict(metric='precomputed', linkage='average')
            c = Similars(both)
            if self.last_fn != 'normalize': c = c.normalize()
            both = c.cosine(abs=True).value()
            np.fill_diagonal(both, 0)  # silhouette_score precomputed requires this
        else:
            model, model_args, sil_args = KMeans, {}, {}

        # Find optimal number of clusters (62006ffb for kmeans.intertia_ knee approach)
        guess = Box(
            guess=math.floor(1 + 3.5 * math.log10(n)),
            max=min(math.ceil(n / 2), 50),  # math.floor(1 + 5 * math.log10(n))
        )
        guess['step'] = math.ceil(guess.max / 10)
        K = range(2, guess.max, guess.step)
        best = Box(model=None, nc=None, score=None)
        for nc in K:
            m = model(n_clusters=nc, **model_args).fit(both)
            score = silhouette_score(both, m.labels_, **sil_args)
            if best.model is None or score > best.score:
                best.score, best.model, best.nc = score, m, nc
        print(f"{algo}(n={n}) best.nc={best.nc},score={round(best.score, 2)}. guess.guess={guess.guess},max={guess.max}")

        # Label & return clustered centroids + labels
        labels = best.model.labels_
        if y is None:
            return self.centroids(x, labels), labels
        l_x, l_y = self._split(labels, x, y)
        return (self.centroids(x, l_x), self.centroids(y, l_y)), (l_x, l_y)

    @staticmethod
    def centroids(x, labels):
        return np.array([
            x[labels == l].mean(0).squeeze()
            for l in range(labels.max())
        ])

    @chain(device_in='cpu')
    def autoencode(self, x, y, dims=[500,150,20], filename=None, batch_norm=True):
        assert y is None, "Don't pass y into autoencode (FIXME)"
        return autoencode(x, dims, filename, batch_norm)
