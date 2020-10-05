FROM lefnire/ml-tools:transformers-pt-tf-0.0.3

RUN \
  pip install --no-cache-dir spacy && python -m spacy download en_core_web_sm && \
  pip install --no-cache-dir \
  python-box \
  pandas \
  xgboost \
  sklearn \
  gensim \
  beautifulsoup4 \
  markdown \
  lemminflect \
  scipy \
  html5lib \
  hyperopt \
  kneed \
  tqdm \
  textacy \
  pytest \
  hnswlib
