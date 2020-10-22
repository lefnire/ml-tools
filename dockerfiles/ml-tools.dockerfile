FROM lefnire/ml-tools:transformers-0.0.18

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
  optuna \
  kneed \
  # CleanText
  beautifulsoup4 \
  markdown2 \
  markdownify \
  html5lib \
  textacy \
  # NLP
  spacy \
  spacy-stanza \
  lemminflect \
  gensim

# TODO move this to /storage setup
RUN python -m spacy download en
