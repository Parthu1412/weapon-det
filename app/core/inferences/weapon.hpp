#pragma once

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace app::core::inferences {

/**
 * RF-DETR ONNX (same role as Python rfdetr): BGR→RGB, resize, ImageNet norm, NCHW.
 * Inputs/outputs: "input" / "dets" [1,N,4], "labels" [1,N,C] logits → sigmoid.
 */
class WeaponInference {
public:
    WeaponInference(const std::string& onnx_path, float conf_thresh, int input_h = 560, int input_w = 560);

    std::pair<std::vector<std::vector<int>>, float> detect(const cv::Mat& frame);

private:
    void preprocess_(const cv::Mat& bgr);

    Ort::Env env_;
    std::optional<Ort::Session> session_;
    float conf_thresh_;
    int input_h_;
    int input_w_;
    std::vector<float> input_buffer_;
    std::vector<int64_t> input_shape_;
};

/**
 * Weapon orchestrator facade (WeaponService) — thin wrapper over WeaponInference.
 */
class WeaponModel {
public:
    explicit WeaponModel(const std::string& onnx_path, float conf_thresh, int input_h = 560, int input_w = 560)
        : impl_(onnx_path, conf_thresh, input_h, input_w)
    {}

    std::pair<std::vector<std::vector<int>>, float> detect(const cv::Mat& bgr) { return impl_.detect(bgr); }

private:
    WeaponInference impl_;
};

}  // namespace app::core::inferences
