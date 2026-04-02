#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace app {
namespace config {

struct CameraConfig {
    int id = 0;
    std::string url;
    std::string client_type;
    int store_id = 0;
    std::string websocket_url;
};

/**
 * Global configuration (weapon detection pipeline) — env + optional .env in cwd.
 */
class AppConfig {
public:
    static AppConfig& getInstance();

    void load();

    int store_id = 123;
    int total_cameras = 0;
    std::string client_type = "webrtc";

    int buffer_size = 2;
    int fps = 1;

    /** ONNX path for RF-DETR weapon model (export via: model.export() in rfdetr Python). */
    std::string weapon_model_path = "weapon.onnx";
    std::string weapon_model_s3_key = "models/checkpoint_best_total.pth";
    float confidence_threshold = 0.5f;
    float iou_threshold = 0.1f;
    float person_bbox_expansion_percent = 0.2f;

    std::string person_detection_model = "yolo11n.pt";
    std::string person_detection_model_path = "yolo11n.torchscript";
    std::string person_detection_model_s3_key;
    int person_class_id = 0;
    /** Match Ultralytics predict defaults (see ultralytics/cfg/default.yaml). */
    float person_confidence_threshold = 0.25f;
    float person_iou_threshold = 0.7f;

    bool publish_weapon_without_person = true;

    std::string redis_host = "localhost";
    int redis_port = 6379;
    std::string redis_password;
    int redis_expiry = 200;
    int64_t longint_max = 30000LL;

    std::string aws_bucket;
    std::string aws_region = "us-east-2";

    std::vector<std::string> kafka_bootstrap_servers;
    std::string kafka_client_id;
    std::string kafka_topic = "weapon_topic";
    std::string kafka_aws_region;

    /** Loaded from RABBITMQ_*_PRODUCER with fallback to RABBITMQ_* (matches weapon Python config). */
    std::string rabbitmq_user;
    std::string rabbitmq_host;
    int rabbitmq_port = 5671;
    std::string rabbitmq_pass;
    bool rabbitmq_use_ssl = true;
    /** SO_RCVTIMEO/SO_SNDTIMEO in seconds; env RABBITMQ_SOCKET_TIMEOUT (default 10), like pika socket_timeout. */
    int rabbitmq_socket_timeout_sec = 10;

    bool use_generic_queue = true;
    std::string generic_queue_name = "weapon_queue";

    std::string zmq_camera_to_weapon_host = "127.0.0.1";
    int zmq_camera_to_weapon_port = 5558;
    /** PersonDetection process binds PULL here; camera_reader PUSHes RTSP/video frames (Python Queue IPC). */
    int zmq_person_frame_port = 5560;
    /** Empty = sibling binary `person_detection` next to this executable. */
    std::string person_detection_exe;
    /** Seconds to wait after spawning person_detection (model load + ZMQ bind). */
    int person_spawn_grace_sec = 15;
    std::string zmq_weapon_to_output_host = "127.0.0.1";
    int zmq_weapon_msg_gen_port = 5559;

    int max_retries = 3;
    int num_workers = 2;
    int queue_size = 1000;

    std::unordered_map<int, CameraConfig> load_camera_configs() const;

private:
    AppConfig() = default;
};

} // namespace config
} // namespace app
