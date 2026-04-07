#include "config.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <mutex>
#include <stdexcept>

namespace app {
namespace config {

namespace {

std::string trim(std::string s) {
    auto not_space = [](int ch) { return !std::isspace(ch); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

void load_dotenv_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) return;
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));
        if (!val.empty() && (val.front() == '"' || val.front() == '\'')) {
            char q = val.front();
            if (val.size() >= 2 && val.back() == q)
                val = val.substr(1, val.size() - 2);
        }
        if (::getenv(key.c_str()) == nullptr) {
#ifdef _WIN32
            _putenv_s(key.c_str(), val.c_str());
#else
            ::setenv(key.c_str(), val.c_str(), 0);
#endif
        }
    }
}

int getenv_int(const char* k, int def) {
    const char* v = std::getenv(k);
    if (!v || !*v) return def;
    try { return std::stoi(v); } catch (...) { return def; }
}

float getenv_float(const char* k, float def) {
    const char* v = std::getenv(k);
    if (!v || !*v) return def;
    try { return std::stof(v); } catch (...) { return def; }
}

std::string getenv_str(const char* k, const std::string& def = {}) {
    const char* v = std::getenv(k);
    return v ? std::string(v) : def;
}

bool getenv_bool(const char* k, bool def) {
    const char* v = std::getenv(k);
    if (!v) return def;
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s == "1" || s == "true" || s == "yes";
}

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim(item);
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

} // namespace

AppConfig& AppConfig::getInstance() {
    static AppConfig inst;
    static std::once_flag once;
    std::call_once(once, [](AppConfig* self) { self->load(); }, &inst);
    return inst;
}

void AppConfig::load() {
    load_dotenv_file(".env");

    store_id = getenv_int("STORE_ID", 123);
    total_cameras = getenv_int("TOTAL_CAMERAS", 0);
    client_type = getenv_str("CLIENT_TYPE", "webrtc");
    buffer_size = getenv_int("BUFFER_SIZE", 2);
    fps = getenv_int("FPS", 1);

    weapon_model_path = getenv_str("MODEL_PATH", "inference_model.onnx");
    weapon_model_s3_key = getenv_str("MODEL_S3_KEY", weapon_model_s3_key);
    confidence_threshold = getenv_float("CONFIDENCE_THRESHOLD", 0.2f);
    iou_threshold = getenv_float("IOU_THRESHOLD", 0.0f);
    person_bbox_expansion_percent = getenv_float("PERSON_BBOX_EXPANSION_PERCENT", 0.2f);

    person_detection_model = getenv_str("PERSON_DETECTION_MODEL", person_detection_model);
    person_detection_model_path = getenv_str("PERSON_DETECTION_MODEL_PATH", "yolov8m.torchscript");
    person_detection_model_s3_key = getenv_str("PERSON_DETECTION_MODEL_S3_KEY", "");
    person_class_id = getenv_int("PERSON_CLASS_ID", 0);
    person_confidence_threshold = getenv_float("PERSON_CONFIDENCE_THRESHOLD", 0.25f);
    person_iou_threshold = getenv_float("PERSON_IOU_THRESHOLD", 0.7f);

    publish_weapon_without_person = getenv_bool("PUBLISH_WEAPON_WITHOUT_PERSON", true);

    redis_host = getenv_str("REDIS_HOST", "localhost");
    redis_port = getenv_int("REDIS_PORT", 6379);
    redis_password = getenv_str("REDIS_PASSWORD", "moksa123");
    redis_expiry = getenv_int("REDIS_EXPIRY", 200);
    try {
        longint_max = static_cast<int64_t>(std::stoll(getenv_str("LONGINT_MAX", "30000")));
    } catch (...) {
        longint_max = 30000LL;
    }

    aws_bucket = getenv_str("AWS_BUCKET", "");
    aws_region = getenv_str("AWS_REGION", "us-east-2");

    kafka_bootstrap_servers = split_csv(getenv_str("KAFKA_BOOTSTRAP_SERVERS", ""));
    kafka_client_id = getenv_str("KAFKA_CLIENT_ID", "");
    kafka_topic = getenv_str("KAFKA_TOPIC", "weapon_topic");
    kafka_aws_region = getenv_str("KAFKA_AWS_REGION", "");
    if (kafka_aws_region.empty())
        kafka_aws_region = aws_region;

    rabbitmq_host = getenv_str("RABBITMQ_HOST_PRODUCER", getenv_str("RABBITMQ_HOST", ""));
    rabbitmq_port = getenv_int("RABBITMQ_PORT_PRODUCER", getenv_int("RABBITMQ_PORT", 5671));
    rabbitmq_user = getenv_str("RABBITMQ_USER_PRODUCER", getenv_str("RABBITMQ_USER", ""));
    rabbitmq_pass = getenv_str("RABBITMQ_PASS_PRODUCER", getenv_str("RABBITMQ_PASS", ""));
    rabbitmq_use_ssl = getenv_bool("RABBITMQ_USE_SSL", true);
    rabbitmq_socket_timeout_sec = getenv_int("RABBITMQ_SOCKET_TIMEOUT", 10);

    use_generic_queue = getenv_bool("USE_GENERIC_QUEUE", true);
    generic_queue_name = getenv_str("GENERIC_QUEUE_NAME", "weapon_queue");

    zmq_camera_to_weapon_host = getenv_str("ZMQ_CAMERA_TO_WEAPON_HOST", "127.0.0.1");
    zmq_camera_to_weapon_port = getenv_int("ZMQ_CAMERA_TO_WEAPON_PORT", 5558);
    zmq_person_frame_port = getenv_int("ZMQ_PERSON_FRAME_PORT", 5560);
    person_detection_exe = getenv_str("PERSON_DETECTION_EXE", "");
    person_spawn_grace_sec = getenv_int("PERSON_SPAWN_GRACE_SEC", 15);
    zmq_weapon_to_output_host = getenv_str("ZMQ_WEAPON_TO_OUTPUT_HOST", "127.0.0.1");
    zmq_weapon_msg_gen_port = getenv_int("ZMQ_WEAPON_TO_OUTPUT_PORT", 5559);

    max_retries = getenv_int("MAX_RETRIES", 3);
    num_workers = getenv_int("NUM_WORKERS", 2);
    queue_size = getenv_int("QUEUE_SIZE", 1000);
}

std::unordered_map<int, CameraConfig> AppConfig::load_camera_configs() const {
    std::unordered_map<int, CameraConfig> out;
    if (total_cameras <= 0) return out;

    for (int i = 1; i <= total_cameras; ++i) {
        std::string suffix = "_" + std::to_string(i);
        CameraConfig c;
        c.id = getenv_int(("CAMERA_ID" + suffix).c_str(), i);
        c.url = getenv_str(("CAMERA_URL" + suffix).c_str(), "");
        c.websocket_url = getenv_str(("WEBSOCKET_URL" + suffix).c_str(), "");
        c.store_id = getenv_int(("STORE_ID" + suffix).c_str(), store_id);
        c.client_type = getenv_str(("CLIENT_TYPE" + suffix).c_str(), client_type);

        if (c.url.empty())
            continue;

        static const std::vector<std::string> valid_types = {"webrtc", "rtsp", "redis", "video"};
        bool valid_type = std::find(valid_types.begin(), valid_types.end(), c.client_type) != valid_types.end();
        if (!valid_type)
            throw std::runtime_error("Invalid client type '" + c.client_type + "' for camera " + std::to_string(c.id));
        if (!c.store_id)
            throw std::runtime_error("Store ID cannot be empty for camera " + std::to_string(c.id));
        if (c.client_type == "webrtc" && c.websocket_url.empty())
            throw std::runtime_error("WebSocket URL required for WebRTC camera " + std::to_string(c.id));

        out[i] = c;
    }
    return out;
}

} // namespace config
} // namespace app
