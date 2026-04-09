// WeaponMessage — Kafka payload struct for weapon-detection events.
// store_id, moksa_camera_id, confidence, s3_key, timestamp, bbox, etc.
#pragma once

#include <nlohmann/json.hpp>
#include <string>

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

// Preserve insertion order:
inline nlohmann::ordered_json to_ordered_json(const WeaponMessage& m)
{
    nlohmann::ordered_json oj;
    oj["store_id"] = m.store_id;
    oj["moksa_camera_id"] = m.moksa_camera_id;
    oj["detections"] = m.detections;
    oj["gcs_uri"] = m.gcs_uri;
    oj["trace_id"] = m.trace_id;
    oj["timestamp"] = m.timestamp;
    oj["model_version"] = m.model_version;
    return oj;
}

inline void to_json(nlohmann::json& j, const WeaponMessage& m)
{
    j = to_ordered_json(m);
}

}  // namespace utils
}  // namespace app
