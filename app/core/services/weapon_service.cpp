#include "weapon_service.hpp"
#include "../../utils/bbox_utils.hpp"
#include "../../config.hpp"
#include "../../utils/logger.hpp"
#include <opencv2/imgproc.hpp>

namespace app::core::services {

WeaponService::WeaponService()
    : weapon_(app::config::AppConfig::getInstance().weapon_model_path,
              app::config::AppConfig::getInstance().confidence_threshold)
{}

cv::Mat WeaponService::annotate_(const cv::Mat& frame, const std::vector<std::vector<int>>& weapon_boxes) {
    cv::Mat out = frame.clone();
    for (const auto& box : weapon_boxes) {
        if (box.size() != 4) continue;
        int x1 = box[0], y1 = box[1], x2 = box[2], y2 = box[3];
        cv::rectangle(out, {x1, y1}, {x2, y2}, cv::Scalar(0, 0, 255), 2);
        cv::putText(out, "Weapon", {x1, y1 - 10}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 0, 255), 2);
    }
    return out;
}

std::optional<WeaponProcessResult> WeaponService::process_frame(
    const cv::Mat& frame,
    const std::vector<app::utils::PersonDetection>* person_detections)
{
    auto& cfg = app::config::AppConfig::getInstance();
    auto [weapon_boxes, conf] = weapon_.detect(frame);
    if (weapon_boxes.empty())
        return std::nullopt;

    std::vector<app::utils::PersonDetection> matched_persons;

    if (person_detections && !person_detections->empty()) {
        auto person_boxes = app::utils::BBoxUtils::extract_person_boxes(
            *person_detections, cfg.person_bbox_expansion_percent);
        std::vector<std::vector<float>> wfloat;
        for (const auto& b : weapon_boxes) {
            wfloat.push_back({static_cast<float>(b[0]), static_cast<float>(b[1]),
                              static_cast<float>(b[2]), static_cast<float>(b[3])});
        }
        auto [has_match, matches] =
            app::utils::BBoxUtils::has_iou_match(person_boxes, wfloat, cfg.iou_threshold);
        if (!has_match) {
            app::utils::Logger::info("Weapons detected but no IOU match with persons. Skipping.");
            return std::nullopt;
        }
        std::vector<bool> take(person_detections->size(), false);
        for (const auto& tup : matches) {
            take[std::get<0>(tup)] = true;
        }
        for (std::size_t i = 0; i < person_detections->size(); ++i) {
            if (take[i]) matched_persons.push_back((*person_detections)[i]);
        }
        app::utils::Logger::info("Weapon detected with IOU match | weapon_count=" +
            std::to_string(weapon_boxes.size()) + " matched_person_count=" + std::to_string(matched_persons.size()));
    } else {
        if (!cfg.publish_weapon_without_person) {
            return std::nullopt;
        }
        app::utils::Logger::info("Weapon detected (no person detected) | weapon_count=" +
            std::to_string(weapon_boxes.size()));
    }

    WeaponProcessResult r;
    r.weapon_detections = std::move(weapon_boxes);
    r.confidence = conf;
    r.person_detections = std::move(matched_persons);
    r.annotated_frame = annotate_(frame, r.weapon_detections);
    return r;
}

}  // namespace app::core::services
