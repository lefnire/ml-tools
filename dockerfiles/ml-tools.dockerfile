FROM lefnire/ml-tools:transformers-pt-tf-0.0.5

RUN \
  pip install --no-cache-dir \
  # misc
  python-box \
  tqdm \
  pytest \
  # ML
  pandas \
  xgboost \
  sklearn \
  scipy \
  hyperopt \
  kneed \
  # CleanText
  beautifulsoup4 \
  markdown \
  markdownify \
  html5lib \
  textacy \
  # NLP
  spacy \
  spacy-stanza \
  lemminflect \
  gensim

# TODO move this to /storage setup
RUN python -m spacy download en_core_web_sm
