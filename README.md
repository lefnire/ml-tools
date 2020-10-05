Various ML utilities I use in most of my projects. Includes

* Dockerfiles
* Text cleanup & similarity methods using Spacy + lemmatization, TF-IDF or BERT embeddings, cosine / Jensen-Shannon 
* Basic AutoEncoder
* XGBoost hyper-optimization & feature_importances_ extraction (coming soon)
* BERT utils, like batch-chunking multiple docs to GPU only what fits (coming soon)

No modules are installed for you, see https://github.com/lefnire/blob/master/gnothi/gpu.dockerfile for modules I'm using. I'll fix this.

No PyPI yet, do `pip install git+git://github.com/lefnire/lefnire_ml_utils.git`

## Similars
Handy NLP utility for performing various text-cleanup, vectorization, & similarity methods. Meant for passing in lists of strings, but can be used just for the vector utilities (similarity methods, etc). 

`Similars()` is a chainer pattern, so each step you call returns another instance of itself so you can grab intermediate values along the way and then keep going. Call `.value()` when you want intermediate values. See `similars.py` for the details, but this is what it looks like:

```python
from lefnire_ml_utils import Similars, cleantext

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
save_load_path = '/storage/ae.tf'
corpus = Similars(sentences_a).embed().autoencode(save_load_path=save_load_path)
# Now you have a trained model at that path
x = Similars(sentences_b).embed().autoencode(save_load_path=save_load_path).value()
dists = Similars(x, corpus).cosine().value()
# Kmeans works much better with dim-reduced data
clusters = Similars(x).cluster(algo='kmeans').value()

# Note, you don't need to normalize() before using autoencode. See the function's signature (eg batch_norm). The 
# embeddings will be normalized, so it both learns to reduce, and normalize (helping downstream tasks) 
```
