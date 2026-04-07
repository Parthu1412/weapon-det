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

# ==============================================================================
# System libcurl workaround (same as fall-cpp)
# Ubuntu's system OpenCV may pull GDAL/NetCDF built against system libcurl4
# versioned symbols (CURL_OPENSSL_4). vcpkg libcurl lacks those versioned
# symbols → unresolved symbols at link. Fix: prefer system libcurl for any
# target that also links OpenCV.
# ==============================================================================
set(WEAPON_CURL_FOR_OPENCV CURL::libcurl)
if(UNIX AND NOT APPLE)
    find_library(WEAPON_SYSTEM_CURL NAMES curl
        PATHS /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu /usr/lib64 /usr/lib /lib64 /lib
        NO_DEFAULT_PATH NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH
        NO_CMAKE_INSTALL_PREFIX NO_CMAKE_SYSTEM_PATH)
    if(NOT WEAPON_SYSTEM_CURL OR WEAPON_SYSTEM_CURL MATCHES "vcpkg_installed")
        unset(WEAPON_SYSTEM_CURL CACHE)
        foreach(_c IN ITEMS
                /usr/lib/x86_64-linux-gnu/libcurl.so.4
                /lib/x86_64-linux-gnu/libcurl.so.4
                /usr/lib64/libcurl.so.4)
            if(EXISTS "${_c}")
                set(WEAPON_SYSTEM_CURL "${_c}" CACHE FILEPATH "" FORCE)
                break()
            endif()
        endforeach()
    endif()
    if(WEAPON_SYSTEM_CURL AND NOT WEAPON_SYSTEM_CURL MATCHES "vcpkg_installed")
        add_library(weapon_system_curl SHARED IMPORTED)
        set_target_properties(weapon_system_curl PROPERTIES IMPORTED_LOCATION "${WEAPON_SYSTEM_CURL}")
        set(WEAPON_CURL_FOR_OPENCV weapon_system_curl)
        message(STATUS "OpenCV: using system libcurl (${WEAPON_SYSTEM_CURL}) for GDAL/NetCDF symbol versions")
    endif()
endif()

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
)
target_include_directories(camera_reader PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(camera_reader PRIVATE
    ${WEAPON_OPENCV_DEPS}
    libzmq
    nlohmann_json::nlohmann_json
    redis++::redis++
    ${WEAPON_CURL_FOR_OPENCV}
)

add_executable(person_detection
    app/core/orchestrators/person_detection_main.cpp
    ${COMMON_SRC}
    app/core/inferences/person.cpp
    app/utils/aws.cpp
)
target_include_directories(person_detection PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(person_detection PRIVATE
    ${TORCH_LIBRARIES}
    ${WEAPON_OPENCV_DEPS}
    libzmq
    nlohmann_json::nlohmann_json
    aws-cpp-sdk-s3
    aws-cpp-sdk-core
    ${WEAPON_CURL_FOR_OPENCV}
)

add_executable(weapon_inference
    app/core/orchestrators/weapon.cpp
    ${COMMON_SRC}
    app/core/inferences/weapon.cpp
    app/core/services/weapon_service.cpp
    app/utils/bbox_utils.cpp
    app/utils/aws.cpp
)
target_include_directories(weapon_inference PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(weapon_inference PRIVATE
    ${TORCH_LIBRARIES}
    ${WEAPON_OPENCV_DEPS}
    libzmq
    nlohmann_json::nlohmann_json
    aws-cpp-sdk-s3
    aws-cpp-sdk-core
    onnxruntime::onnxruntime
    ${WEAPON_CURL_FOR_OPENCV}
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
    ${WEAPON_CURL_FOR_OPENCV}
)
