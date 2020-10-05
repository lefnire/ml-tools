FROM lefnire/ml-tools:transformers-pt-tf-0.0.2

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
  hnswlib \
  git+git://github.com/UKPLab/sentence-transformers.git@dc84bb7644946d8217fa2ea5211a75c53be89101

COPY . /app
WORKDIR /app
RUN pip install -e .

ENV TORCH_HOME=/storage
ENV PYTHONPATH=.

ENTRYPOINT /bin/bash
