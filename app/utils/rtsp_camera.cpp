#include "rtsp_camera.hpp"
#include "logger.hpp"
#include <cmath>
#include <stdexcept>

namespace app::utils {

RTSPCamera::RTSPCamera(std::string url, int fps, int buffer_size)
    : url_(std::move(url)), target_fps_(fps), buffer_size_(buffer_size)
{
    cap_.open(url_);
    if (!cap_.isOpened()) {
        Logger::error("[RTSPCamera] Failed to open RTSP stream | url=" + url_);
        throw std::runtime_error("RTSP Camera Initialization Failed");
    }

    auto now = std::chrono::system_clock::now();
    buffer_start_time_ = now;
    last_log_time_ = now;

    thread_finished_.store(false);
    // Python: threading.Thread(..., daemon=True) — no portable daemon in C++;
    // timeout + detach in release() matches “do not block past join(5)” / process-exit semantics.
    thread_ = std::thread(&RTSPCamera::readFramesLoop, this);

    Logger::info("[RTSPCamera] Initialized | url=" + url_ +
                 " | target_fps=" + std::to_string(target_fps_) +
                 " | buffer_size=" + std::to_string(buffer_size_));
}

RTSPCamera::~RTSPCamera() { release(); }

std::vector<cv::Mat> RTSPCamera::sampleFramesToTargetFps(const std::vector<FrameWithTs>& frames) const {
    if (frames.empty()) return {};
    int total = static_cast<int>(frames.size());
    std::vector<cv::Mat> sampled;
    if (total <= target_fps_) {
        sampled.reserve(static_cast<size_t>(total));
        for (const auto& e : frames) sampled.push_back(e.frame.clone());
        return sampled;
    }

    double sampling_interval = static_cast<double>(total) / static_cast<double>(target_fps_);
    sampled.reserve(static_cast<size_t>(target_fps_));
    for (int i = 0; i < target_fps_; ++i) {
        int frame_index = static_cast<int>(static_cast<double>(i) * sampling_interval);
        if (frame_index < total)
            sampled.push_back(frames[static_cast<size_t>(frame_index)].frame.clone());
    }
    return sampled;
}

void RTSPCamera::readFramesLoop() {
    while (running_.load()) {
        try {
            cv::Mat frame;
            bool ret = cap_.read(frame);

            if (!ret || frame.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            auto current_time = std::chrono::system_clock::now();
            FrameWithTs entry;
            entry.frame = frame.clone();
            entry.timestamp = current_time;
            frame_buffer_1s_.push_back(std::move(entry));
            frames_received_count_++;

            double elapsed =
                std::chrono::duration<double>(current_time - buffer_start_time_).count();
            if (elapsed >= 1.0) {
                auto sampled = sampleFramesToTargetFps(frame_buffer_1s_);
                {
                    std::lock_guard<std::mutex> lk(buffer_lock_);
                    for (auto& f : sampled) {
                        frame_buffer_.push_back(std::move(f));
                        while (static_cast<int>(frame_buffer_.size()) > buffer_size_)
                            frame_buffer_.pop_front();
                    }
                }
                frames_sampled_count_ += static_cast<int>(sampled.size());

                double time_elapsed =
                    std::chrono::duration<double>(current_time - last_log_time_).count();
                if (time_elapsed >= 1.0) {
                    double received_fps =
                        time_elapsed > 0 ? static_cast<double>(frames_received_count_) / time_elapsed
                                         : 0.0;
                    double sampled_fps =
                        time_elapsed > 0 ? static_cast<double>(frames_sampled_count_) / time_elapsed
                                         : 0.0;
                    Logger::debug("[RTSPCamera] RTSP FPS stats | url=" + url_ +
                                  " | received_fps=" +
                                  std::to_string(std::round(received_fps * 10.0) / 10.0).substr(0, 5) +
                                  " | frames_received=" + std::to_string(frames_received_count_) +
                                  " | sampled_fps=" +
                                  std::to_string(std::round(sampled_fps * 10.0) / 10.0).substr(0, 5) +
                                  " | frames_sampled=" + std::to_string(frames_sampled_count_) +
                                  " | target_fps=" + std::to_string(target_fps_));
                    frames_received_count_ = 0;
                    frames_sampled_count_ = 0;
                    last_log_time_ = current_time;
                }

                frame_buffer_1s_.clear();
                buffer_start_time_ = current_time;
            }

        } catch (const std::exception& e) {
            if (running_.load()) {
                Logger::error("[RTSPCamera] Error reading frame in background thread | url=" +
                              url_ + " | error=" + e.what());
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    thread_finished_.store(true);
}

bool RTSPCamera::read(cv::Mat& out) {
    std::lock_guard<std::mutex> lk(buffer_lock_);
    if (frame_buffer_.empty()) return false;
    out = frame_buffer_.front().clone();
    frame_buffer_.pop_front();
    return true;
}

bool RTSPCamera::isOpened() const { return cap_.isOpened(); }

void RTSPCamera::release() {
    if (released_.exchange(true)) return;

    running_.store(false);

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!thread_finished_.load() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

    if (thread_.joinable()) {
        if (thread_finished_.load()) {
            thread_.join();
        } else {
            Logger::warning(
                "[RTSPCamera] Background thread did not terminate, may cause resource leaks | url=" +
                url_);
            if (thread_finished_.load())
                thread_.join();
            else
                thread_.detach();
        }
    }

    cap_.release();
    {
        std::lock_guard<std::mutex> lk(buffer_lock_);
        frame_buffer_.clear();
    }
    frame_buffer_1s_.clear();
    Logger::info("[RTSPCamera] Released | url=" + url_);
}

}  // namespace app::utils
