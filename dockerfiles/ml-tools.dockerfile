FROM lefnire/ml-tools:transformers-pt-tf-0.0.5

RUN \
  pip install --no-cache-dir \
  python-box \
  pandas \
  xgboost \
  sklearn \
  gensim \
  beautifulsoup4 \
  markdown \
  scipy \
  html5lib \
  hyperopt \
  kneed \
  tqdm \
  textacy \
  pytest \
  spacy-stanza
