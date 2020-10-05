Various ML utilities I use in most of my projects. Includes

* Dockerfiles ([Dockerhub](https://hub.docker.com/repository/docker/lefnire/ml-tools))
  1. `cuda101-py38.dockerfile`: CUDA 10.1, CuDNN 7, Python 3.8 (Miniconda). I've had trouble getting Python>3.6 with Tensorflow, so I made this to address that issue.
  1. `transformers-pt.dockerfile`: (1), Transformers, Sentence Transformers, Pytorch
  1. `transformers-pt-tf.dockerfile`: (1), (2), Tensorflow
  1. `Dockerfile`: (1-3), packages needed to run samples in this repo. So start with that Dockerfile to play with this project. 
* NLP
  * Text cleanup & similarity methods using Spacy + lemmatization, TF-IDF or BERT embeddings, cosine / Jensen-Shannon 
  * AutoEncoder
  * BERT utils, like batch-chunking multiple docs to GPU only what fits (coming soon)
* Other
  * XGBoost hyper-optimization & feature_importances_ extraction (coming soon)
  
No PyPI yet, do `pip install git+git://github.com/lefnire/ml_tools.git`. And no modules installed via that command, see `./Dockerfile` to get started.


## Similars
Handy NLP utility for performing various text-cleanup, vectorization, & similarity methods. Meant for passing in lists of strings, but can be used just for the vector utilities (similarity methods, etc). 

`Similars()` is a chainer pattern, so each step you call returns another instance of itself so you can grab intermediate values along the way and then keep going. Call `.value()` when you want intermediate values. See `similars.py` for the details, but this is what it looks like:

```python
from ml_tools import Similars, cleantext

chain = Similars(sentences)

### BERT
# Gets pairwise similarities of sentences
dists = chain.embed().normalize().cosine().value()
# Gets x compared to y
dists = Similars(sentences_a, sentences_b).embed().normalize().cosine().value()

### TF-IDF
clean = chain.cleantext()
# actually, I want to save away the clean-text for later
for_later = clean.value()
dists = clean.tfidf().jensenshannon().value()
# hmm... those clean texts weren't very clean, let's clean it up some more
clean = chain.cleantext(methods=[
    cleantext.strip_hml, 
    cleantext.fix_punct, 
    cleantext.only_ascii,
    cleantext.keywords
])

### Just vectors
# Similars(x: Union[List[str], np.ndarray], y: <same> = None) works like this. If y passed in, you operate on x vs y (eg, x cosine-sim to y); if not passed in, operate pairwise on x. x,y can be lists of texts, or vectors. So you can start the process at any point
dists = Similars(vecs_a, vecs_b).cosine().value()

### Clustering
clusters = Similars(sentences).embed().normalize().cluster(method='kmeans').value()
# If y is provided, x is the only thing clustered unless you specify cluster_both
chain = Similars(sentences_a, sentences_b).normalize()
clusters = chain.cluster(method='agglomorative').value()
assert len(clusters) == len(sentences_a)
# cluster both
clusters = chain.cluster(cluster_both=True).value()
assert len(clusters[0]) == len(sentences_a)
assert len(clusters[1]) == len(sentences_b)
# cluster vectors, I'm not using NLP at all
clusters = Similars(vecs).normalize().cluster().value()
```

The Keras Autoencoder is very valuable for dimensionality reduction, which makes downstream clustering, cosine similarity, etc; simpler, faster, and more accurate.

```python
# see autoencode() signature for options.
chain = Similars(sentences).embed().autoencode().value()

# My recommendation is to grab a huge corpus List[str], train the autoencoder once, then use that in teh future to 
# to dim-reduce all embeddings 
fname = '/storage/ae.tf'
corpus = Similars(sentences_a).embed().autoencode(filename=fname)
# Now you have a trained model at that path
x = Similars(sentences_b).embed().autoencode(filename=fname).value()
# Kmeans/euclidean works better than agglom/cosine after dim-reduction
dists = Similars(x, corpus).cdist().value()
clusters = Similars(x).cluster(algo='kmeans').value()

# Note, you don't need to normalize() before using autoencode. See the function's signature (eg batch_norm). The 
# embeddings will be normalized, so it both learns to reduce, and normalize (helping downstream tasks) 
```
