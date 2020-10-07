FROM lefnire/ml-tools:cuda101-py38-0.0.5

RUN pip install --no-cache-dir \
  mkl \
  torch \
  transformers==3.3.1 \
  git+git://github.com/UKPLab/sentence-transformers.git@dc84bb7644946d8217fa2ea5211a75c53be89101
