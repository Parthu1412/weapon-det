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

inline void to_json(nlohmann::json& j, const WeaponMessage& m) {
    j = nlohmann::json{
        {"store_id", m.store_id},
        {"moksa_camera_id", m.moksa_camera_id},
        {"detections", m.detections},
        {"gcs_uri", m.gcs_uri},
        {"trace_id", m.trace_id},
        {"timestamp", m.timestamp},
        {"model_version", m.model_version},
    };
}

} // namespace utils
} // namespace app
