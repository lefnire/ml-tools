FROM lefnire/ml-tools:ml-tools-0.0.3

COPY . /app
WORKDIR /app
RUN pip install -e .

ENV TORCH_HOME=/storage
ENV PYTHONPATH=.

ENTRYPOINT /bin/bash
