// Bounding-box utilities — IoU, box expansion, best-person extraction.
// Implementation of BBoxUtils;
#include "bbox_utils.hpp"

#include <algorithm>
#include <sstream>
#include <string>

#include "logger.hpp"

namespace app::utils {

float BBoxUtils::calculate_iou(const std::vector<float>& box1, const std::vector<float>& box2)
{
    if (box1.size() < 4 || box2.size() < 4)
        return 0.f;
    float x1_1 = box1[0], y1_1 = box1[1], x2_1 = box1[2], y2_1 = box1[3];
    float x1_2 = box2[0], y1_2 = box2[1], x2_2 = box2[2], y2_2 = box2[3];
    float x1_i = std::max(x1_1, x1_2);
    float y1_i = std::max(y1_1, y1_2);
    float x2_i = std::min(x2_1, x2_2);
    float y2_i = std::min(y2_1, y2_2);
    if (x2_i <= x1_i || y2_i <= y1_i)
        return 0.f;
    float inter = (x2_i - x1_i) * (y2_i - y1_i);
    float a1 = (x2_1 - x1_1) * (y2_1 - y1_1);
    float a2 = (x2_2 - x1_2) * (y2_2 - y1_2);
    float uni = a1 + a2 - inter;
    if (uni <= 0.f)
        return 0.f;
    return inter / uni;
}

std::pair<bool, std::vector<std::tuple<std::size_t, std::size_t, float>>> BBoxUtils::has_iou_match(
    const std::vector<std::vector<float>>& person_boxes,
    const std::vector<std::vector<float>>& weapon_boxes, float iou_threshold)
{
    std::vector<std::tuple<std::size_t, std::size_t, float>> matches;
    if (person_boxes.empty() || weapon_boxes.empty())
        return {false, matches};
    for (std::size_t pi = 0; pi < person_boxes.size(); ++pi)
    {
        for (std::size_t wi = 0; wi < weapon_boxes.size(); ++wi)
        {
            float iou = calculate_iou(person_boxes[pi], weapon_boxes[wi]);
            if (iou >= iou_threshold)
                matches.emplace_back(pi, wi, iou);
        }
    }
    return {!matches.empty(), matches};
}

std::vector<float> BBoxUtils::expand_bbox(const std::vector<float>& box, float expansion_percent)
{
    if (box.size() < 4 || expansion_percent <= 0.f)
        return box;
    float x1 = box[0], y1 = box[1], x2 = box[2], y2 = box[3];
    float w = x2 - x1;
    float h = y2 - y1;
    float cx = (x1 + x2) / 2.f;
    float cy = (y1 + y2) / 2.f;
    float nw = w * (1.f + expansion_percent);
    float nh = h * (1.f + expansion_percent);
    return {cx - nw / 2.f, cy - nh / 2.f, cx + nw / 2.f, cy + nh / 2.f};
}

std::vector<std::vector<float>> BBoxUtils::extract_person_boxes(
    const std::vector<PersonDetection>& person_detections, float expansion_percent)
{
    std::vector<std::vector<float>> out;
    for (const auto& d : person_detections)
    {
        if (d.box.size() != 4)
        {
            std::ostringstream box_str;
            box_str << "[";
            for (std::size_t i = 0; i < d.box.size(); ++i)
            {
                if (i)
                    box_str << ",";
                box_str << d.box[i];
            }
            box_str << "]";
            Logger::warning(
                std::string("[BBoxUtils] Invalid person bounding box format expected_values=4 "
                            "got_values=") +
                std::to_string(d.box.size()) + " box=" + box_str.str());
            continue;
        }
        std::vector<float> b = d.box;
        if (expansion_percent > 0.f)
            b = expand_bbox(b, expansion_percent);
        out.push_back(std::move(b));
    }
    if (out.empty())
        Logger::debug("[BBoxUtils] No valid person boxes extracted from detections");
    return out;
}

}  // namespace app::utils
