FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    curl -sSf https://sh.rustup.rs | sh -s -- -y && \
    rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.cargo/bin:${PATH}"
