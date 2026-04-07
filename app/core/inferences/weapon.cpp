#include "weapon.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <torch/torch.h>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <algorithm>

#include "../../utils/logger.hpp"

namespace app::core::inferences {

// ImageNet normalization constants
static constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
static constexpr float kStd[3]  = {0.229f, 0.224f, 0.225f};

// --- WeaponInference ---------------------------------------------------------

WeaponInference::WeaponInference(const std::string& onnx_path, float conf_thresh,
                                 int input_h, int input_w, float nms_iou_thresh)
    : env_(ORT_LOGGING_LEVEL_WARNING, "rfdetr")
    , conf_thresh_(conf_thresh)
    , nms_iou_thresh_(nms_iou_thresh)
    , input_h_(input_h)
    , input_w_(input_w)
{
    if (input_h_ % 14 != 0 || input_w_ % 14 != 0) {
        app::utils::Logger::error(
            "[WeaponInference] input_h/input_w must be divisible by 14 (DINOv2 patch size). Got: " +
            std::to_string(input_h_) + "x" + std::to_string(input_w_));
        std::exit(EXIT_FAILURE);
    }

    try {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // Try CUDA EP first, silently fall back to CPU
        try {
            OrtCUDAProviderOptions cuda_opts{};
            // Force CuDNN to benchmark and find the fastest convolution algorithms for your hardware
            cuda_opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchHeuristic; 
            // Optimize memory arena allocation
            cuda_opts.arena_extend_strategy = 0; 
            // Enable CUDA graph if supported (massive speedup for static input sizes)
            cuda_opts.do_copy_in_default_stream = 1;

            opts.AppendExecutionProvider_CUDA(cuda_opts);
            app::utils::Logger::info("[WeaponInference] CUDA execution provider enabled.");
        } catch (...) {
            app::utils::Logger::info("[WeaponInference] CUDA unavailable; running on CPU.");
        }

        app::utils::Logger::info("[WeaponInference] Loading ONNX session from " + onnx_path +
            " (first CUDA run may take 1-5 min for kernel compilation, please wait...)");
        session_.emplace(env_, onnx_path.c_str(), opts);
        app::utils::Logger::info("[WeaponInference] Loaded: " + onnx_path);

        // --- OPTIMIZATION 1: Pre-allocate Input Buffer ---
        // Avoids allocating 3*H*W floats on the heap every frame
        input_buffer_.resize(3 * input_h_ * input_w_);
        input_shape_ = {1, 3, input_h_, input_w_};

        // --- OPTIMIZATION 2: Warmup Phase ---
        app::utils::Logger::debug("[WeaponInference] Warming up model with 100 dummy frames...");
        cv::Mat dummy(input_h_, input_w_, CV_8UC3, cv::Scalar(114, 114, 114));
        for (int i = 0; i < 100; ++i) {
            detect(dummy); 
        }
        app::utils::Logger::debug("[WeaponInference] Warmup complete.");

    } catch (const Ort::Exception& e) {
        app::utils::Logger::error(std::string("[WeaponInference] Failed to load model: ") + e.what());
        std::exit(EXIT_FAILURE);
    }
}


void WeaponInference::preprocess_(const cv::Mat& bgr) {
    // Move to GPU, BGR->RGB, HWC->NCHW, [0,1] float — all on GPU
    torch::Device device(torch::kCUDA);
    auto raw = torch::from_blob(bgr.data, {1, bgr.rows, bgr.cols, 3}, torch::kByte)
                   .to(device);
    auto t = raw.permute({0, 3, 1, 2}).flip(1).to(torch::kFloat).div_(255.0f);

    // Resize on GPU
    auto resized = torch::nn::functional::interpolate(
        t,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{input_h_, input_w_})
            .mode(torch::enumtype::kBilinear{})
            .align_corners(false));

    // ImageNet normalization on GPU
    resized[0][0] = (resized[0][0] - kMean[0]) / kStd[0];
    resized[0][1] = (resized[0][1] - kMean[1]) / kStd[1];
    resized[0][2] = (resized[0][2] - kMean[2]) / kStd[2];

    // Single contiguous copy GPU->CPU into pre-allocated ONNX input buffer
    auto cpu_tensor = resized.contiguous().cpu();
    std::memcpy(input_buffer_.data(), cpu_tensor.data_ptr<float>(),
                input_buffer_.size() * sizeof(float));
}

std::pair<std::vector<std::vector<int>>, float> WeaponInference::detect(const cv::Mat& frame) {
    if (frame.empty()) return {{}, 0.0f};

    try {
        // --- Preprocess ---
        auto t0 = std::chrono::high_resolution_clock::now();
        preprocess_(frame);
        auto t1 = std::chrono::high_resolution_clock::now();

        // Build input tensor linked to our pre-allocated buffer
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, input_buffer_.data(), input_buffer_.size(),
            input_shape_.data(), input_shape_.size());

        // --- Inference ---
        static const char* input_names[]  = {"input"};
        static const char* output_names[] = {"dets", "labels"};
        
        auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                     input_names, &input_tensor, 1,
                                     output_names, 2);
        auto t2 = std::chrono::high_resolution_clock::now();

        // --- Decode ---
        const auto dets_shape   = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        const auto labels_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();

        const int64_t num_queries  = dets_shape[1];
        const int64_t num_classes  = labels_shape[2];

        const float* dets_ptr   = outputs[0].GetTensorData<float>();
        const float* labels_ptr = outputs[1].GetTensorData<float>();

        std::vector<cv::Rect2d> raw_boxes;
        std::vector<float> scores;
        float max_conf = 0.0f;

        const float orig_w = static_cast<float>(frame.cols);
        const float orig_h = static_cast<float>(frame.rows);

       for (int64_t i = 0; i < num_queries; ++i) {
            float best_score = 0.f;

            // Labels are raw logits — apply sigmoid per class
            for (int64_t c = 0; c < num_classes; ++c) {
                const float logit = labels_ptr[i * num_classes + c];
                const float score = 1.0f / (1.0f + std::exp(-logit));
                if (score > best_score) best_score = score;
            }

            if (best_score < conf_thresh_) continue;

            // REVERTED: DETR natively outputs normalized [cx, cy, w, h]
            const float cx = dets_ptr[i * 4 + 0];
            const float cy = dets_ptr[i * 4 + 1];
            const float bw = dets_ptr[i * 4 + 2];
            const float bh = dets_ptr[i * 4 + 3];

            // Convert to top-left pixel coordinates
            const float x1 = (cx - bw * 0.5f) * orig_w;
            const float y1 = (cy - bh * 0.5f) * orig_h;
            const float pw = bw * orig_w;
            const float ph = bh * orig_h;

            raw_boxes.emplace_back(x1, y1, pw, ph);  // cv::Rect2d: x, y, width, height
            scores.push_back(best_score);
            if (best_score > max_conf) max_conf = best_score;
        }

        // --- NMS with containment suppression (mirrors rfdetr Python post-processing) ---
        // Standard IoU NMS misses cases where a small box is fully inside a large one
        // (IoU is low because union is large). We additionally suppress if one box
        // contains >= 80% of the smaller box's area.
        std::vector<int> nms_indices;
        if (!raw_boxes.empty()) {
            // Step 1: standard IoU NMS
            cv::dnn::NMSBoxes(raw_boxes, scores, conf_thresh_, nms_iou_thresh_, nms_indices);

            // Step 2: containment filtering — for each pair of surviving boxes,
            // if >= 80% of the SMALLER box's area is covered by the other box,
            // suppress the LOWER-SCORED one (regardless of which is smaller/larger).
            // This handles the case where the smaller box has higher confidence,
            // which the old one-directional check missed.
            std::vector<bool> suppressed2(nms_indices.size(), false);
            for (int i = 0; i < static_cast<int>(nms_indices.size()); ++i) {
                if (suppressed2[i]) continue;
                const auto& bi = raw_boxes[static_cast<size_t>(nms_indices[i])];
                const float area_i = static_cast<float>(bi.width * bi.height);
                for (int j = i + 1; j < static_cast<int>(nms_indices.size()); ++j) {
                    if (suppressed2[j]) continue;
                    const auto& bj = raw_boxes[static_cast<size_t>(nms_indices[j])];
                    const float area_j = static_cast<float>(bj.width * bj.height);
                    const float ix1 = std::max((float)bi.x, (float)bj.x);
                    const float iy1 = std::max((float)bi.y, (float)bj.y);
                    const float ix2 = std::min((float)(bi.x + bi.width),  (float)(bj.x + bj.width));
                    const float iy2 = std::min((float)(bi.y + bi.height), (float)(bj.y + bj.height));
                    if (ix2 <= ix1 || iy2 <= iy1) continue;
                    const float inter = (ix2 - ix1) * (iy2 - iy1);
                    const float min_area = std::min(area_i, area_j);
                    // If >= 80% of the smaller box is inside the other, suppress the lower-scored one
                    if (min_area > 0.f && (inter / min_area) >= 0.8f) {
                        if (scores[static_cast<size_t>(nms_indices[i])] >= scores[static_cast<size_t>(nms_indices[j])])
                            suppressed2[j] = true;
                        else
                            suppressed2[i] = true;
                    }
                }
            }
            std::vector<int> final_indices;
            final_indices.reserve(nms_indices.size());
            for (int i = 0; i < static_cast<int>(nms_indices.size()); ++i) {
                if (!suppressed2[i]) final_indices.push_back(nms_indices[i]);
            }
            nms_indices = std::move(final_indices);
        }

        const float fw = orig_w - 1.0f;
        const float fh = orig_h - 1.0f;
        std::vector<std::vector<int>> boxes;
        boxes.reserve(nms_indices.size());
        
        for (int idx : nms_indices) {
            const auto& r = raw_boxes[static_cast<size_t>(idx)];
            int ix1 = static_cast<int>(std::lround(std::max(0.f, std::min((float)r.x,           fw))));
            int iy1 = static_cast<int>(std::lround(std::max(0.f, std::min((float)r.y,           fh))));
            int ix2 = static_cast<int>(std::lround(std::max(0.f, std::min((float)(r.x + r.width),  fw))));
            int iy2 = static_cast<int>(std::lround(std::max(0.f, std::min((float)(r.y + r.height), fh))));
            
            boxes.push_back({ix1, iy1, ix2, iy2});
        }
        
        auto t3 = std::chrono::high_resolution_clock::now();

        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        app::utils::Logger::debug("[WeaponInference] Weapon detection completed"
            " | detection_count=" + std::to_string(boxes.size()) +
            " | confidence_threshold=" + std::to_string(conf_thresh_) +
            " | preprocess=" + std::to_string(ms(t0, t1)) +
            "ms | infer=" + std::to_string(ms(t1, t2)) +
            "ms | postprocess=" + std::to_string(ms(t2, t3)) + "ms");

        return {boxes, max_conf};

    } catch (const Ort::Exception& e) {
        app::utils::Logger::error(std::string("[WeaponInference] Inference error: ") + e.what());
        return {{}, 0.0f};
    }
}

}  // namespace app::core::inferences