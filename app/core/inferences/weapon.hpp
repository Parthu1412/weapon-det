// Weapon Detection Inference — interface for WeaponInference and WeaponModel.
// WeaponInference: loads RF-DETR ONNX model, preprocesses BGR frames
//   (BGR→RGB, resize to 560x560, ImageNet normalise, NCHW), runs inference,
//   decodes sigmoid-filtered detections to pixel-space xyxy boxes.
// WeaponModel: thin facade used by WeaponService.
#pragma once

#include <onnxruntime_cxx_api.h>

#include <opencv2/core.hpp>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace app::core::inferences {

/**
 * RF-DETR ONNX : BGR→RGB, resize, ImageNet norm, NCHW.
 * Inputs/outputs: "input" / "dets" [1,N,4], "labels" [1,N,C] logits → sigmoid.
 */
class WeaponInference
{
public:
    WeaponInference(const std::string& onnx_path, float conf_thresh, int input_h = 616,
                    int input_w = 616, float nms_iou_thresh = 0.5f);

    std::pair<std::vector<std::vector<int>>, float> detect(const cv::Mat& frame);

private:
    void preprocess_(const cv::Mat& bgr);

    Ort::Env env_;
    std::optional<Ort::Session> session_;
    float conf_thresh_;
    float nms_iou_thresh_;
    int input_h_;
    int input_w_;
    std::vector<float> input_buffer_;
    std::vector<int64_t> input_shape_;
};

/**
 * Weapon orchestrator facade (WeaponService) — thin wrapper over WeaponInference.
 */
class WeaponModel
{
public:
    explicit WeaponModel(const std::string& onnx_path, float conf_thresh, int input_h = 616,
                         int input_w = 616, float nms_iou_thresh = 0.5f)
        : impl_(onnx_path, conf_thresh, input_h, input_w, nms_iou_thresh)
    {
    }

    std::pair<std::vector<std::vector<int>>, float> detect(const cv::Mat& bgr)
    {
        return impl_.detect(bgr);
    }

private:
    WeaponInference impl_;
};

}  // namespace app::core::inferences
