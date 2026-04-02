# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG TRITON_VERSION=26.02
ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3
FROM ${BASE_IMAGE}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        autoconf \
        autogen \
        libb64-dev \
        libz-dev \
        rapidjson-dev && \
    DEBIAN_FRONTEND=noninteractive apt-get upgrade -y

RUN pip3 install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 torch==2.6.0+cu118 && \
    pip3 install --no-cache-dir --extra-index-url https://pypi.ngc.nvidia.com regex fire tritonclient[all] && \
    pip3 install --no-cache-dir accelerate transformers huggingface_hub tokenizers SentencePiece sacrebleu datasets tqdm omegaconf rouge_score && \
    pip3 install --no-cache-dir cmake==3.31.10 && \
    pip3 install --no-cache-dir langid==1.1.6 && \
    pip3 install --no-cache-dir lingua-language-detector==2.0.2

# Fetch FasterTransformer source (minimal: single commit, no history, no tags)
ARG FT_REPO=https://github.com/neevaco/FasterTransformer.git
ARG FT_COMMIT=b6b21406449ab19f00d1d5f97338065037b5f8e3
RUN mkdir -p /workspace/FasterTransformer && cd /workspace/FasterTransformer && \
    git init -q && \
    git fetch --depth 1 --no-tags ${FT_REPO} ${FT_COMMIT} && \
    git checkout -q FETCH_HEAD && \
    git log --oneline -1 | tee /workspace/FT_version.txt

# Patch FasterTransformer sources for CUDA 13 / C++17 / GCC 13 compatibility
COPY docker/patch_cuda_type_utils.py /tmp/patch_cuda_type_utils.py
RUN python3 /tmp/patch_cuda_type_utils.py /workspace/FasterTransformer

# backend build
ADD . /workspace/build/fastertransformer_backend
RUN mkdir -p /workspace/build/fastertransformer_backend/build

WORKDIR /workspace/build/fastertransformer_backend/build
ARG FORCE_BACKEND_REBUILD=0
# Configure and build
RUN CUDAFLAGS="-include stdio.h" cmake \
      -D CMAKE_EXPORT_COMPILE_COMMANDS=1 \
      -D CMAKE_BUILD_TYPE=Release \
      -D ENABLE_FP8=OFF \
      -D CMAKE_INSTALL_PREFIX=/opt/tritonserver \
      -D CMAKE_CUDA_ARCHITECTURES="75;80;86;90" \
      -D SM="75 80 86 90" \
      -D CXX_STD=17 \
      -D USE_NVTX=OFF \
      -D FT_DIR=/workspace/FasterTransformer \
      -D CMAKE_CUDA_FLAGS="-DCCCL_IGNORE_DEPRECATED_CPP_DIALECT" \
      -D TRITON_COMMON_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" \
      -D TRITON_CORE_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" \
      -D TRITON_BACKEND_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" \
      .. && \
    CUDAFLAGS="-include stdio.h" make -O -j"$(grep -c ^processor /proc/cpuinfo)" install && \
    rm /workspace/build/fastertransformer_backend/build/bin/*_example -rf && \
    rm /workspace/build/fastertransformer_backend/build/lib/lib*Backend.so -rf

# Removing redandant tools because of CVEs, no longer needed after build
RUN apt-get purge -y \
      git git-man vim vim-tiny wget openssh-client patch hwdata doca-sdk-common libfabric-aws-bin libfabric-aws-dev \
      binutils-x86-64-linux-gnu libpmi2-0-dev gpg gpgconf libcurl4-openssl-dev jq \
      autoconf \
      autogen \
      libb64-dev \
      libz-dev \
      rapidjson-dev && \
    apt-get autoremove --purge -y && \
    dpkg -l | awk '/^rc/ {print $2}' | xargs -r dpkg --purge && \
    apt-get clean && \
    echo "#!/usr/bin/sh\necho ssh: Could not resolve hostname mpi-is-disabled: nodename nor servname provided, or not known\nexit 255" >/usr/bin/ssh && chmod +x /usr/bin/ssh && \
    rm -rf /usr/local/cuda/{NsightSystems-cli*,bin/nsys,extras/CUPTI,gds} && \
    rm -rf /opt/nvidia/{nsight-systems-cli,nsight-compute} && \
    rm -rf /var/lib/apt/lists/*

ENV NCCL_LAUNCH_MODE=GROUP
ENV WORKSPACE=/workspace
WORKDIR /workspace
