#pragma once

#include <nlohmann/json.hpp>
#include <vector>

namespace app::utils {

struct PersonDetection {
    std::vector<float               > box;   // [x1,y1,x2,y2]
    float score = 0.f;
};

/**
 * Parse Redis / ZMQ JSON person detections: detections[].box + optional score
 */
inline void parse_person_detections_json(const nlohmann::json& j,
                                         std::vector<PersonDetection>& out)
{
    out.clear();
    if (!j.is_object() || !j.contains("detections") || !j["detections"].is_array())
        return;
    for (const auto& det : j["detections"]) {
        if (!det.contains("box")) continue;
        const auto& bj = det["box"];
        if (!bj.is_array() || bj.size() != 4) continue;
        PersonDetection p;
        for (int i = 0; i < 4; ++i) {
            if (bj[i].is_number())
                p.box.push_back(static_cast<float>(bj[i].get<double>()));
        }
        if (p.box.size() != 4) continue;
        if (det.contains("score") && det["score"].is_number())
            p.score = static_cast<float>(det["score"].get<double>());
        out.push_back(std::move(p));
    }
}

inline nlohmann::json person_detections_to_json(const std::vector<PersonDetection>& dets) {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& p : dets) {
        nlohmann::json det;
        det["box"] = p.box;
        det["score"] = p.score;
        arr.push_back(std::move(det));
    }
    return arr;
}

}  // namespace app::utils
