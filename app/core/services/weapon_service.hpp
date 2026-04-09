// Weapon Detection Service - interface for WeaponService.
// WeaponService wraps WeaponModel (RF-DETR ONNX) and BBoxUtils to apply
// business logic: run inference, IOU-match weapons to persons, annotate frame.
#pragma once

#include <opencv2/core.hpp>
#include <optional>
#include <vector>

#include "../../utils/detection_json.hpp"
#include "../inferences/weapon.hpp"

namespace app::core::services {

struct WeaponProcessResult {
    std::vector<std::vector<int>> weapon_detections;
    float confidence = 0.f;
    std::vector<app::utils::PersonDetection> person_detections;
    cv::Mat annotated_frame;
};

class WeaponService
{
public:
    WeaponService();

    std::optional<WeaponProcessResult> process_frame(
        const cv::Mat& frame, const std::vector<app::utils::PersonDetection>* person_detections);

private:
    cv::Mat annotate_(const cv::Mat& frame, const std::vector<std::vector<int>>& weapon_boxes);

    app::core::inferences::WeaponModel weapon_;
};

}  // namespace app::core::services
