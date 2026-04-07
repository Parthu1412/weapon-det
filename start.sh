#!/bin/bash
set -eu

# Repo root — always run from the directory containing this script
RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$RUN_DIR"

# STEP 1: Build (skip when WEAPON_SKIP_BUILD=1, e.g. inside Docker with prebuilts)

if [ "${WEAPON_SKIP_BUILD:-0}" != "1" ]; then
    echo "========================================"
    echo "      COMPILING C++ SOURCE CODE         "
    echo "========================================"

    mkdir -p build

    # Wipe stale cache if CMakeCache was generated from a different source path
    if [ -f "./build/CMakeCache.txt" ]; then
        cached_src=$(grep -m1 "^CMAKE_HOME_DIRECTORY" ./build/CMakeCache.txt 2>/dev/null | cut -d= -f2 || true)
        if [ -n "$cached_src" ] && [ "$cached_src" != "$RUN_DIR" ]; then
            echo "Stale CMakeCache (was: $cached_src). Cleaning..."
            rm -rf ./build/CMakeCache.txt ./build/CMakeFiles
        fi
    fi

    # Re-configure if: no cache, CMakeLists.txt changed, or build.ninja is missing
    need_configure=0
    [ ! -f "./build/CMakeCache.txt" ] && need_configure=1
    [ "./CMakeLists.txt" -nt "./build/CMakeCache.txt" ] && need_configure=1
    [ ! -f "./build/build.ninja" ] && need_configure=1

    if [ "$need_configure" -eq 1 ]; then
        echo "Configuring CMake..."

        # Auto-detect GPU compute capability for LibTorch+CUDA on CMake < 3.31
        if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ] && [ -z "${WEAPON_TORCH_CUDA_ARCH_LIST:-}" ]; then
            if command -v nvidia-smi >/dev/null 2>&1; then
                _cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' \r' || true)"
                if [ -n "${_cap:-}" ]; then
                    export TORCH_CUDA_ARCH_LIST="${_cap}"
                    echo "Auto-set TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
                fi
            fi
        fi

        _EXTRA_CMAKE=()
        [ -n "${WEAPON_TORCH_CUDA_ARCH_LIST:-}" ] && _EXTRA_CMAKE+=("-DWEAPON_TORCH_CUDA_ARCH_LIST=${WEAPON_TORCH_CUDA_ARCH_LIST}")

        cmake -B build -S . -G Ninja \
            -DCMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake \
            -DVCPKG_TARGET_TRIPLET=x64-linux-dynamic \
            -DCMAKE_PREFIX_PATH="/opt/libtorch;/opt/onnxruntime" \
            -DCMAKE_BUILD_TYPE=Release \
            "${_EXTRA_CMAKE[@]}"
    fi

    if [ ! -f "./build/build.ninja" ]; then
        echo "Error: configure failed — build/build.ninja not found. Fix CMake errors above."
        exit 1
    fi

    echo "Building..."
    cmake --build build --config Release -j"$(nproc)"

else
    echo "Skipping build (WEAPON_SKIP_BUILD=1) — using prebuilt binaries."
fi

# STEP 2: Set runtime environment

BUILD_DIR="${BUILD_DIR:-${RUN_DIR}/build}"

# Load vcpkg OpenSSL before any binary runs — needed by Kafka/RabbitMQ SSL stacks.
# Looks in (priority order): explicit VCPKG_SSL_DIR → local build dir → Docker install dir.
_vcpkg_lib="${RUN_DIR}/build/vcpkg_installed/x64-linux-dynamic/lib"
if   [ -n "${VCPKG_SSL_DIR:-}" ];                       then _ssl_dir="${VCPKG_SSL_DIR}"
elif [ -d "${_vcpkg_lib}" ];                             then _ssl_dir="${_vcpkg_lib}"
elif [ -d "/opt/weapon/vcpkg-libs" ];                    then _ssl_dir="/opt/weapon/vcpkg-libs"
else                                                          _ssl_dir=""
fi

if [ -n "${_ssl_dir}" ]; then
    if [ -f "${_ssl_dir}/libssl.so" ] && [ -f "${_ssl_dir}/libcrypto.so" ]; then
        export LD_PRELOAD="${_ssl_dir}/libssl.so:${_ssl_dir}/libcrypto.so"
        echo "LD_PRELOAD set from ${_ssl_dir}"
    elif [ -f "${_ssl_dir}/libssl.so.3" ] && [ -f "${_ssl_dir}/libcrypto.so.3" ]; then
        export LD_PRELOAD="${_ssl_dir}/libssl.so.3:${_ssl_dir}/libcrypto.so.3"
        echo "LD_PRELOAD set from ${_ssl_dir}"
    fi
fi

export OPENCV_FFMPEG_CAPTURE_OPTIONS="${OPENCV_FFMPEG_CAPTURE_OPTIONS:-rtsp_transport;tcp}"
export TORCH_XNNPACK_DISABLE="${TORCH_XNNPACK_DISABLE:-1}"


# STEP 3: Launch pipeline
# Start order: most-downstream first so each process is ready before its upstream sends.
#   msg_gen        — connects to weapon_inference output (ZMQ_WEAPON_TO_OUTPUT_PORT)
#   weapon_inference — binds output port; connects to camera port; downloads weapon model from S3
#   camera_reader  — binds camera port; auto-spawns person_detection subprocess for non-Redis cameras
#                    (person_detection downloads person model from S3 and runs YOLO TorchScript)

echo "========================================"
echo "       STARTING WEAPON PIPELINE         "
echo "========================================"

declare -A PIDS

cleanup() {
    echo "Stopping all processes..."
    for name in "${!PIDS[@]}"; do
        pid=${PIDS[$name]}
        kill "$pid" 2>/dev/null || true
        sleep 1
        kill -9 "$pid" 2>/dev/null || true
    done
    echo "Done."
}

trap cleanup SIGTERM SIGINT EXIT

# echo "Starting msg_gen..."
# "$BUILD_DIR/msg_gen" &
# PIDS["msg_gen"]=$!
# sleep 2

echo "Starting weapon_inference (RF-DETR ONNX; downloads weapon model from S3 on first run)..."
"$BUILD_DIR/weapon_inference" &
PIDS["weapon_inference"]=$!
sleep 2

echo "Starting camera_reader (will spawn person_detection subprocess for non-Redis cameras)..."
"$BUILD_DIR/camera_reader" &
PIDS["camera_reader"]=$!

echo "All services started. Monitoring..."

# Monitor: exit immediately if any process dies (lets the supervisor restart everything)
while true; do
    for name in "${!PIDS[@]}"; do
        if ! kill -0 "${PIDS[$name]}" 2>/dev/null; then
            echo "[$name] (PID ${PIDS[$name]}) died unexpectedly. Exiting."
            exit 1
        fi
    done
    sleep 5
done
