FROM lefnire/ml-tools:cuda101-py38-0.0.18

RUN pip install --no-cache-dir \
  torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html \
  transformers \
  sentence-transformers \
  https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-2.3.0-cp38-cp38-manylinux2010_x86_64.whl
