#pragma once

#include <string>
#include <nlohmann/json.hpp>

namespace app {
namespace utils {

struct WeaponMessage {
    int store_id = 0;
    int moksa_camera_id = 0;
    nlohmann::json detections = nlohmann::json::array();
    std::string gcs_uri;
    std::string trace_id;
    std::string timestamp;
    std::string model_version = "v1.0";
};

// Use ordered_json to preserve insertion order — matches Python json.dumps dict order:
// store_id → moksa_camera_id → detections → gcs_uri → trace_id → timestamp → model_version
inline void to_json(nlohmann::json& j, const WeaponMessage& m) {
    nlohmann::ordered_json oj;
    oj["store_id"]        = m.store_id;
    oj["moksa_camera_id"] = m.moksa_camera_id;
    oj["detections"]      = m.detections;
    oj["gcs_uri"]         = m.gcs_uri;
    oj["trace_id"]        = m.trace_id;
    oj["timestamp"]       = m.timestamp;
    oj["model_version"]   = m.model_version;
    j = oj;
}

} // namespace utils
} // namespace app
