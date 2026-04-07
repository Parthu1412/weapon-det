# syntax=docker/dockerfile:1
# Build:
#   DOCKER_BUILDKIT=1 docker build -t weapon-cpp:latest .
#
# Run (code baked in; start.sh compiles then launches all processes):
#   docker run --gpus all --rm --env-file .env weapon-cpp:latest
#
# Or mount code for live dev (start.sh rebuilds on change):
#   docker run --gpus all --rm -v $(pwd):/app --env-file .env weapon-cpp:latest

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ==============================================================================
# System dependencies
# ==============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    libopencv-dev \
    libboost-chrono-dev \
    libboost-system-dev \
    librdkafka-dev \
    librabbitmq-dev \
    libssl-dev \
    libzmq3-dev \
    ninja-build \
    pkg-config \
    python3 \
    tar \
    unzip \
    wget \
    zip \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# SimpleAmqpClient — not in Ubuntu 22.04 apt; build from source (same as fall-cpp)
# ==============================================================================
RUN git clone --depth 1 --branch v2.5.1 https://github.com/alanxz/SimpleAmqpClient.git /tmp/SimpleAmqpClient \
    && cmake -S /tmp/SimpleAmqpClient -B /tmp/SimpleAmqpClient/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr \
        -DENABLE_SSL_SUPPORT=ON \
        -DBUILD_SHARED_LIBS=ON \
    && cmake --build /tmp/SimpleAmqpClient/build --parallel $(nproc) \
    && cmake --install /tmp/SimpleAmqpClient/build \
    && rm -rf /tmp/SimpleAmqpClient

# ==============================================================================
# LibTorch — CUDA 11.8 (required by person_detection binary for YOLO TorchScript)
# ==============================================================================
RUN wget -q "https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcu118.zip" \
         -O /tmp/libtorch.zip \
    && unzip -q /tmp/libtorch.zip -d /opt \
    && rm /tmp/libtorch.zip

ENV LD_LIBRARY_PATH=/opt/libtorch/lib:${LD_LIBRARY_PATH:-}

# ==============================================================================
# ONNX Runtime — CUDA 11.8 build (onnxruntime port removed from vcpkg 2024.11.16)
# ORT 1.17.x is the last release series supporting CUDA 11.x
# ==============================================================================
ARG ORT_VERSION=1.17.3
RUN wget -q "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz" \
         -O /tmp/onnxruntime.tgz \
    && mkdir -p /opt/onnxruntime \
    && tar -xzf /tmp/onnxruntime.tgz -C /opt/onnxruntime --strip-components=1 \
    && rm /tmp/onnxruntime.tgz
# Create cmake config files for onnxruntime (tarball doesn't include them)
RUN mkdir -p /opt/onnxruntime/lib/cmake/onnxruntime && \
        cat > /opt/onnxruntime/lib/cmake/onnxruntime/onnxruntimeConfig.cmake << 'EOF'
get_filename_component(ONNXRUNTIME_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(ONNXRUNTIME_PREFIX "${ONNXRUNTIME_CMAKE_DIR}/../../.." ABSOLUTE)

set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_PREFIX}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_PREFIX}/lib")

if(NOT TARGET onnxruntime::onnxruntime)
    add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIRS}"
    )
endif()

set(onnxruntime_FOUND TRUE)
EOF

ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib:${LD_LIBRARY_PATH}

# ==============================================================================
# vcpkg — installs: cppzmq, nlohmann-json, redis-plus-plus, librdkafka, curl,
#          aws-sdk-cpp[s3]  (see vcpkg.json)
# ==============================================================================
ENV VCPKG_DEFAULT_TRIPLET=x64-linux-dynamic
RUN git clone --depth 1 --branch 2024.11.16 https://github.com/microsoft/vcpkg.git /opt/vcpkg \
    || git clone --depth 1 https://github.com/microsoft/vcpkg.git /opt/vcpkg \
    && /opt/vcpkg/bootstrap-vcpkg.sh

# ==============================================================================
# Application
# ==============================================================================
WORKDIR /app

# T4 GPU = compute capability 7.5 — pre-set so start.sh skips nvidia-smi detection
ENV TORCH_CUDA_ARCH_LIST=7.5

# Copy requirements first for better layer caching (vcpkg deps won't reinstall if vcpkg.json unchanged)
COPY vcpkg.json .
COPY CMakeLists.txt .

# Pre-install all vcpkg dependencies into the image so container startup is fast.
# BuildKit persistent cache keeps vcpkg binary archives on the host so packages
# already downloaded are never re-fetched from GitHub on subsequent builds,
# even when `docker build --no-cache` is used. The retry loop handles transient
# 5xx errors from GitHub (e.g. 502 Bad Gateway).
RUN --mount=type=cache,target=/root/.cache/vcpkg \
    for attempt in 1 2 3; do \
        /opt/vcpkg/vcpkg install --triplet x64-linux-dynamic && break; \
        echo "vcpkg install attempt $attempt failed, retrying in 15s..."; \
        sleep 15; \
    done

COPY app/ app/

COPY start.sh .
RUN chmod +x ./start.sh

CMD ["bash", "./start.sh"]
