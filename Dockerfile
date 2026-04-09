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
# SimpleAmqpClient — not in Ubuntu 22.04 apt; build from source
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
#          aws-sdk-cpp[s3]
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

# Copy requirements first for better layer caching
COPY vcpkg.json .
COPY CMakeLists.txt .

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
