cmake_minimum_required(VERSION 3.20)
project(weapon_detection_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GPU arch for LibTorch + CMake < 3.31 (same pattern as Fall_C++_CURR)
set(WEAPON_TORCH_CUDA_ARCH_LIST "" CACHE STRING
    "GPU arch list for TORCH_CUDA_ARCH_LIST, e.g. 8.6")
if(WEAPON_TORCH_CUDA_ARCH_LIST)
    set(ENV{TORCH_CUDA_ARCH_LIST} "${WEAPON_TORCH_CUDA_ARCH_LIST}")
endif()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio dnn)
find_package(CURL CONFIG REQUIRED)
find_package(ZeroMQ CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(redis++ CONFIG REQUIRED)
find_package(RdKafka CONFIG REQUIRED)
find_package(aws-cpp-sdk-s3 CONFIG REQUIRED)
find_package(aws-cpp-sdk-core CONFIG REQUIRED)

# ONNX Runtime for RF-DETR weapon inference (RF-DETR only supports ONNX export, not TorchScript)
find_package(onnxruntime CONFIG REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(SAC REQUIRED IMPORTED_TARGET libSimpleAmqpClient)

if(DEFINED OpenCV_LIBS)
    set(WEAPON_OPENCV_DEPS ${OpenCV_LIBS})
else()
    set(WEAPON_OPENCV_DEPS ${OpenCV_LIBRARIES})
endif()

set(COMMON_SRC
    app/config.cpp
    app/utils/logger.cpp
    app/core/orchestrators/zmq_io.cpp
)

add_executable(camera_reader
    app/core/orchestrators/camera.cpp
    ${COMMON_SRC}
    app/utils/rtsp_camera.cpp
    app/core/inferences/person.cpp
)
target_include_directories(camera_reader PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(camera_reader PRIVATE
    ${TORCH_LIBRARIES}
    ${WEAPON_OPENCV_DEPS}
    libzmq
    nlohmann_json::nlohmann_json
    redis++::redis++
    CURL::libcurl
)

add_executable(weapon_inference
    app/core/orchestrators/weapon.cpp
    ${COMMON_SRC}
    app/core/inferences/person.cpp
    app/core/inferences/weapon.cpp
    app/core/services/weapon_service.cpp
    app/core/services/bbox_utils.cpp
)
target_include_directories(weapon_inference PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(weapon_inference PRIVATE
    ${TORCH_LIBRARIES}
    ${WEAPON_OPENCV_DEPS}
    libzmq
    nlohmann_json::nlohmann_json
    CURL::libcurl
    onnxruntime::onnxruntime
)

add_executable(msg_gen
    app/core/orchestrators/msg_gen.cpp
    ${COMMON_SRC}
    app/utils/aws.cpp
    app/kafka/msk_token.cpp
    app/kafka/kafka_producer.cpp
)
target_include_directories(msg_gen PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(msg_gen PRIVATE
    ${WEAPON_OPENCV_DEPS}
    libzmq
    nlohmann_json::nlohmann_json
    aws-cpp-sdk-s3
    aws-cpp-sdk-core
    RdKafka::rdkafka++
    PkgConfig::SAC
    CURL::libcurl
)
