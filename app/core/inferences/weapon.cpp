#include "weapon.hpp"

#include <opencv2/imgproc.hpp>
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
                                 int input_h, int input_w)
    : env_(ORT_LOGGING_LEVEL_WARNING, "rfdetr")
    , conf_thresh_(conf_thresh)
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
            opts.AppendExecutionProvider_CUDA(cuda_opts);
            app::utils::Logger::info("[WeaponInference] CUDA execution provider enabled.");
        } catch (...) {
            app::utils::Logger::info("[WeaponInference] CUDA unavailable; running on CPU.");
        }

        session_.emplace(env_, onnx_path.c_str(), opts);
        app::utils::Logger::info("[WeaponInference] Loaded: " + onnx_path);

        // --- OPTIMIZATION 1: Pre-allocate Input Buffer ---
        // Avoids allocating 3*H*W floats on the heap every frame
        input_buffer_.resize(3 * input_h_ * input_w_);
        input_shape_ = {1, 3, input_h_, input_w_};

        // --- OPTIMIZATION 2: Warmup Phase ---
        app::utils::Logger::info("[WeaponInference] Warming up model with 100 dummy frames...");
        cv::Mat dummy(input_h_, input_w_, CV_8UC3, cv::Scalar(114, 114, 114));
        for (int i = 0; i < 100; ++i) {
            detect(dummy); 
        }
        app::utils::Logger::info("[WeaponInference] Warmup complete.");

    } catch (const Ort::Exception& e) {
        app::utils::Logger::error(std::string("[WeaponInference] Failed to load model: ") + e.what());
        std::exit(EXIT_FAILURE);
    }
}

void WeaponInference::preprocess_(const cv::Mat& bgr) {
    // BGR -> RGB 
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    // Direct resize to model input size (RF-DETR does NOT use letterboxing)
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(input_w_, input_h_), 0, 0, cv::INTER_LINEAR);

    // uint8 -> float32, scale to [0, 1]
    cv::Mat flt;
    resized.convertTo(flt, CV_32FC3, 1.0f / 255.0f);

    // --- OPTIMIZATION 3: Fused Normalization & CHW Planarization ---
    // Drastically faster than using cv::split() and cv::merge()
    const float* src_ptr = reinterpret_cast<const float*>(flt.data);
    float* dst_r = input_buffer_.data();
    float* dst_g = dst_r + (input_h_ * input_w_);
    float* dst_b = dst_g + (input_h_ * input_w_);

    int num_pixels = input_h_ * input_w_;
    for (int i = 0; i < num_pixels; ++i) {
        dst_r[i] = (src_ptr[i * 3 + 0] - kMean[0]) / kStd[0];
        dst_g[i] = (src_ptr[i * 3 + 1] - kMean[1]) / kStd[1];
        dst_b[i] = (src_ptr[i * 3 + 2] - kMean[2]) / kStd[2];
    }
}

std::pair<std::vector<std::vector<int>>, float> WeaponInference::detect(const cv::Mat& frame) {
    if (frame.empty()) return {{}, 0.0f};

    const bool log_timing = (std::getenv("INFERENCE_TIMING_BREAKDOWN") != nullptr);

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

        std::vector<std::vector<int>> boxes;
        float max_conf = 0.0f;

        const float orig_w = static_cast<float>(frame.cols);
        const float orig_h = static_cast<float>(frame.rows);

        for (int64_t i = 0; i < num_queries; ++i) {
            float best_score = 0.f;
            
            // Labels are raw logits — apply sigmoid per class
            for (int64_t c = 0; c < num_classes; ++c) {
                const float logit = labels_ptr[i * num_classes + c];
                const float score = 1.0f / (1.0f + std::exp(-logit));
                if (score > best_score) {
                    best_score = score;
                }
            }
            
            if (best_score < conf_thresh_) continue;

            // Dets: normalized [cx, cy, w, h] in [0, 1] — convert to pixel xyxy
            const float cx = dets_ptr[i * 4 + 0];
            const float cy = dets_ptr[i * 4 + 1];
            const float bw = dets_ptr[i * 4 + 2];
            const float bh = dets_ptr[i * 4 + 3];

            float x1 = (cx - bw * 0.5f) * orig_w;
            float y1 = (cy - bh * 0.5f) * orig_h;
            float x2 = (cx + bw * 0.5f) * orig_w;
            float y2 = (cy + bh * 0.5f) * orig_h;

            // Clamp to image bounds and convert to int
            const float fw = orig_w - 1.0f;
            const float fh = orig_h - 1.0f;
            
            int ix1 = static_cast<int>(std::lround(std::max(0.f, std::min(x1, fw))));
            int iy1 = static_cast<int>(std::lround(std::max(0.f, std::min(y1, fh))));
            int ix2 = static_cast<int>(std::lround(std::max(0.f, std::min(x2, fw))));
            int iy2 = static_cast<int>(std::lround(std::max(0.f, std::min(y2, fh))));

            boxes.push_back({ix1, iy1, ix2, iy2});
            if (best_score > max_conf) max_conf = best_score;
        }
        
        auto t3 = std::chrono::high_resolution_clock::now();

        if (log_timing) {
            auto ms = [](auto a, auto b) {
                return std::chrono::duration<double, std::milli>(b - a).count();
            };
            app::utils::Logger::info("[WeaponInference] preprocess=" + std::to_string(ms(t0, t1)) +
                                     "ms  infer=" + std::to_string(ms(t1, t2)) +
                                     "ms  postprocess=" + std::to_string(ms(t2, t3)) + "ms");
        }

        return {boxes, max_conf};

    } catch (const Ort::Exception& e) {
        app::utils::Logger::error(std::string("[WeaponInference] Inference error: ") + e.what());
        return {{}, 0.0f};
    }
}

}  // namespace app::core::inferences