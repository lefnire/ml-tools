FROM lefnire/ml-tools:cuda101-py38-0.0.2

RUN \
  pip install --no-cache-dir torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html &&\
  pip install --no-cache-dir transformers==3.3.1 sentence-transformers==0.3.7
