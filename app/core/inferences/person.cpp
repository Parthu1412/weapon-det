#include "person.hpp"

#include <ATen/Context.h>
#include <c10/core/ScalarType.h>
#include <torch/cuda.h>
#include <torch/nn/functional.h>
#include <opencv2/dnn.hpp>
#include <algorithm>
#include <chrono>
#include <cstdlib>

#include "../../utils/logger.hpp"

namespace app::core::inferences {

// --- PersonInference ---------------------------------------------------------

PersonInference::PersonInference(const std::string& model_path, float conf_thresh, 
                                 float iou_thresh, int person_class_id)
    : device_(at::hasCUDA() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU))
    , conf_thresh_(conf_thresh)
    , iou_thresh_(iou_thresh)
    , person_class_id_(person_class_id)
{
    try {
        module_ = torch::jit::load(model_path, device_);
        module_.eval();

        // Auto-detect model weight dtype (FP16/BF16/FP32)
        for (const auto& param : module_.parameters()) {
            const auto st = static_cast<c10::ScalarType>(param.scalar_type());
            if (c10::isFloatingType(st)) {
                model_elem_dtype_ = st;
                break;
            }
        }

        if (model_elem_dtype_ == c10::ScalarType::Half)
            app::utils::Logger::info("[PersonInference] FP16 model detected; inputs will be cast.");
        else if (model_elem_dtype_ == c10::ScalarType::BFloat16)
            app::utils::Logger::info("[PersonInference] BF16 model detected; inputs will be cast.");

        app::utils::Logger::info("[PersonInference] Loaded on " +
            std::string(device_.is_cuda() ? "GPU" : "CPU") + ": " + model_path);
        app::utils::Logger::info("[PersonInference] Person detection model loaded"
            " | model_path=" + model_path +
            " | person_class_id=" + std::to_string(person_class_id));

        // Warmup: Run 100 dummy frames to stabilize CUDA context
        app::utils::Logger::info("[PersonInference] Warming up model with 100 dummy frames...");
        {
            torch::NoGradGuard no_grad;
            cv::Mat dummy(640, 640, CV_8UC3, cv::Scalar(114, 114, 114));
            for (int i = 0; i < 100; ++i) {
                auto inp = preprocess_(dummy);
                module_.forward({inp});
            }
            if (device_.is_cuda()) torch::cuda::synchronize();
        }
        app::utils::Logger::info("[PersonInference] Warmup complete.");

    } catch (const std::exception& e) {
        app::utils::Logger::error(std::string("[PersonInference] Failed to load model: ") + e.what());
        std::exit(EXIT_FAILURE);
    }
}

torch::Tensor PersonInference::preprocess_(const cv::Mat& frame) const {
    auto raw = torch::from_blob(frame.data, {1, frame.rows, frame.cols, 3}, torch::kByte)
                   .clone()
                   .to(device_);

    auto t = raw.permute({0, 3, 1, 2}).to(torch::kFloat).div_(255.0f).flip(1);

    float r = std::min(static_cast<float>(input_h_) / frame.rows,
                       static_cast<float>(input_w_) / frame.cols);
    int new_h = static_cast<int>(std::round(frame.rows * r));
    int new_w = static_cast<int>(std::round(frame.cols * r));

    auto resized = torch::nn::functional::interpolate(
        t,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{new_h, new_w})
            .mode(torch::enumtype::kBilinear{})
            .align_corners(false));

    int dw = (input_w_ - new_w) / 2;
    int dh = (input_h_ - new_h) / 2;
    int pad_r = input_w_ - new_w - dw;
    int pad_b = input_h_ - new_h - dh;

    auto padded = torch::nn::functional::pad(
        resized,
        torch::nn::functional::PadFuncOptions({dw, pad_r, dh, pad_b})
            .value(114.0f / 255.0f));

    return padded.to(model_elem_dtype_);
}

void PersonInference::decode_and_nms_(const torch::Tensor& raw, int orig_w, int orig_h,
                                      std::vector<app::utils::PersonDetection>& out)
{
    out.clear();

    float r = std::min(static_cast<float>(input_h_) / orig_h,
                       static_cast<float>(input_w_) / orig_w);
    int new_w = static_cast<int>(std::round(orig_w * r));
    int new_h = static_cast<int>(std::round(orig_h * r));
    float pad_w = static_cast<float>((input_w_ - new_w) / 2);
    float pad_h = static_cast<float>((input_h_ - new_h) / 2);

    // [1, 4+nc, N] -> [N, 4+nc]; Ultralytics: assigned class = argmax(cls scores), conf = that score
    torch::Tensor t = raw.squeeze(0).transpose(0, 1);
    if (t.dim() != 2 || t.size(1) < 5) return;

    const int64_t ncol = t.size(1);
    const int num_classes = static_cast<int>(ncol - 4);
    if (num_classes <= 0 || person_class_id_ < 0 || person_class_id_ >= num_classes) return;

    torch::Tensor class_scores = t.narrow(1, 4, num_classes);
    auto mx = class_scores.max(1);
    torch::Tensor max_scores = std::get<0>(mx);
    torch::Tensor max_ids = std::get<1>(mx);
    torch::Tensor keep =
        max_ids.eq(static_cast<int64_t>(person_class_id_)) & max_scores.gt(conf_thresh_);

    torch::Tensor filtered = t.index({keep}).to(torch::kFloat).cpu().contiguous();
    if (filtered.size(0) == 0) return;

    const int rows = static_cast<int>(filtered.size(0));
    const int cols = static_cast<int>(filtered.size(1));
    const int class_col = 4 + person_class_id_;
    const float* data = filtered.data_ptr<float>();

    std::vector<cv::Rect2d> boxes;
    std::vector<float> scores;
    boxes.reserve(static_cast<size_t>(rows));
    scores.reserve(static_cast<size_t>(rows));

    for (int i = 0; i < rows; ++i) {
        const float* row = data + i * cols;
        float cx = row[0], cy = row[1], bw = row[2], bh = row[3];
        boxes.emplace_back(cx - bw * 0.5, cy - bh * 0.5, bw, bh);
        scores.push_back(row[class_col]);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_thresh_, iou_thresh_, indices);
    std::sort(indices.begin(), indices.end(),
              [&](int a, int b) { return scores[static_cast<size_t>(a)] > scores[static_cast<size_t>(b)]; });

    out.reserve(indices.size());
    for (int idx : indices) {
        const auto& b = boxes[idx];
        
        // Unscale from letterboxed space back to original frame coordinates
        double x1o = (b.x - pad_w) / r;
        double y1o = (b.y - pad_h) / r;
        double x2o = (b.x + b.width - pad_w) / r;
        double y2o = (b.y + b.height - pad_h) / r;

        // Clip to frame boundaries
        x1o = std::max(0.0, std::min(x1o, static_cast<double>(orig_w - 1)));
        x2o = std::max(0.0, std::min(x2o, static_cast<double>(orig_w - 1)));
        y1o = std::max(0.0, std::min(y1o, static_cast<double>(orig_h - 1)));
        y2o = std::max(0.0, std::min(y2o, static_cast<double>(orig_h - 1)));

        app::utils::PersonDetection p;
        p.box = {static_cast<float>(x1o), static_cast<float>(y1o), 
                 static_cast<float>(x2o), static_cast<float>(y2o)};
        p.score = scores[idx];
        out.push_back(std::move(p));
    }
}

std::vector<app::utils::PersonDetection> PersonInference::detect(const cv::Mat& frame) {
    if (frame.empty()) return {};



    try {
        torch::NoGradGuard no_grad;

        // --- Preprocess ---
        auto t0 = std::chrono::high_resolution_clock::now();
        auto input = preprocess_(frame);
        if (device_.is_cuda()) torch::cuda::synchronize();
        auto t1 = std::chrono::high_resolution_clock::now();

        // --- Inference ---
        auto raw_out = module_.forward({input});
        if (device_.is_cuda()) torch::cuda::synchronize();
        auto t2 = std::chrono::high_resolution_clock::now();

        torch::Tensor pred;
        if (raw_out.isTuple()) {
            pred = raw_out.toTuple()->elements()[0].toTensor();
        } else {
            pred = raw_out.toTensor();
        }

        // Always decode in FP32 regardless of model dtype
        if (pred.scalar_type() != c10::ScalarType::Float)
            pred = pred.to(c10::ScalarType::Float);

        // --- Decode + NMS ---
        std::vector<app::utils::PersonDetection> results;
        decode_and_nms_(pred, frame.cols, frame.rows, results);
        auto t3 = std::chrono::high_resolution_clock::now();

        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        app::utils::Logger::debug("[PersonInference] Person detection completed"
            " | detection_count=" + std::to_string(results.size()) +
            " | preprocess=" + std::to_string(ms(t0, t1)) +
            "ms | infer=" + std::to_string(ms(t1, t2)) +
            "ms | postprocess=" + std::to_string(ms(t2, t3)) + "ms");

        return results;
    } catch (const std::exception& e) {
        app::utils::Logger::error(std::string("[PersonInference] Inference error: ") + e.what());
        return {};
    }
}

}  // namespace app::core::inferences