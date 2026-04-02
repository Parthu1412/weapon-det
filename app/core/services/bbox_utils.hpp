#pragma once

#include "../../utils/detection_json.hpp"
#include <vector>
#include <utility>
#include <cstddef>
#include <tuple>

namespace app::core::services {

class BBoxUtils {
public:
    static float calculate_iou(const std::vector<float>& box1, const std::vector<float>& box2);

    static std::pair<bool, std::vector<std::tuple<std::size_t, std::size_t, float>>> has_iou_match(
        const std::vector<std::vector<float>>& person_boxes,
        const std::vector<std::vector<float>>& weapon_boxes,
        float iou_threshold);

    static std::vector<float> expand_bbox(const std::vector<float>& box, float expansion_percent);

    static std::vector<std::vector<float>> extract_person_boxes(
        const std::vector<app::utils::PersonDetection>& person_detections,
        float expansion_percent);
};

}  // namespace app::core::services
