FROM lefnire/ml-tools:transformers-pt-0.0.2

RUN pip install --no-cache-dir https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-2.3.0-cp38-cp38-manylinux2010_x86_64.whl
