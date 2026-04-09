// ZMQ I/O helpers — interface for weapon pipeline inter-process communication.
// Declares ZmqWeaponFramePacket, ZmqWeaponOutPacket, send/recv functions, and make_trace_id().
#pragma once

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <zmq.hpp>

#include "../../utils/detection_json.hpp"

namespace app::core::orchestrators {

/** Camera / person-detection → weapon orchestrator */
struct ZmqWeaponFramePacket {
    int camera_id = 0;
    int store_id = 0;
    std::string timestamp;
    cv::Mat frame;
    std::vector<uchar> frame_jpg;
    std::vector<app::utils::PersonDetection> person_detections;
};

bool zmq_send_weapon_frame(zmq::socket_t& sock, const ZmqWeaponFramePacket& p);
bool zmq_recv_weapon_frame(zmq::socket_t& sock, ZmqWeaponFramePacket& p,
                           zmq::recv_flags flags = zmq::recv_flags::dontwait);

/** Weapon orchestrator → msg_gen */
struct ZmqWeaponOutPacket {
    int camera_id = 0;
    int store_id = 0;
    std::string timestamp;
    float confidence = 0.f;
    std::vector<std::vector<int>> weapon_boxes;
    std::vector<app::utils::PersonDetection> person_detections;
    cv::Mat annotated_frame;
    std::vector<uchar> annotated_jpg;
};

bool zmq_send_weapon_output(zmq::socket_t& sock, const ZmqWeaponOutPacket& p);
bool zmq_recv_weapon_output(zmq::socket_t& sock, ZmqWeaponOutPacket& p,
                            zmq::recv_flags flags = zmq::recv_flags::dontwait);

std::string make_trace_id();

}  // namespace app::core::orchestrators
