#pragma once

#include <c10/core/ScalarType.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <torch/script.h>

#include "../../utils/detection_json.hpp"

namespace app::core::inferences {

/**
 * Ultralytics-style YOLO TorchScript: letterbox 640, BGR->RGB, pad 114/255.
 * Decode [1, 4+nc, N] (or tuple first tensor) with per-row argmax class + conf,
 * then filter to person_class_id (matches Ultralytics classes=[id]).
 */
class PersonInference {
public:
    PersonInference(const std::string& model_path, float conf_thresh, float iou_thresh, int person_class_id);

    std::vector<app::utils::PersonDetection> detect(const cv::Mat& bgr);

private:
    torch::jit::script::Module module_;
    torch::Device device_;
    c10::ScalarType model_elem_dtype_{c10::ScalarType::Float};
    float conf_thresh_;
    float iou_thresh_;
    int person_class_id_;
    int input_w_ = 640;
    int input_h_ = 640;

    torch::Tensor preprocess_(const cv::Mat& frame) const;
    void decode_and_nms_(const torch::Tensor& raw, int orig_w, int orig_h,
                         std::vector<app::utils::PersonDetection>& out);
};

/** Stable name for camera pipeline — delegates to PersonInference. */
class PersonModel {
public:
    PersonModel(const std::string& torchscript_path, float conf_thresh, float iou_thresh, int person_class_id)
        : impl_(torchscript_path, conf_thresh, iou_thresh, person_class_id)
    {}

    std::vector<app::utils::PersonDetection> detect(const cv::Mat& bgr) { return impl_.detect(bgr); }

private:
    PersonInference impl_;
};

}  // namespace app::core::inferences
