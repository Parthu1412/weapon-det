// RTSP camera interface — FPS-controlled read loop, thread-safe frame delivery.

#pragma once

#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

namespace app::utils {

class RTSPCamera
{
public:
    explicit RTSPCamera(std::string url, int fps = 1, int buffer_size = 60);
    ~RTSPCamera();

    bool read(cv::Mat& out);
    bool isOpened() const;
    void release();

private:
    struct FrameWithTs {
        cv::Mat frame;
        std::chrono::system_clock::time_point timestamp{};
    };

    std::vector<cv::Mat> sampleFramesToTargetFps(const std::vector<FrameWithTs>& frames) const;
    void readFramesLoop();
    void openCapture();

    std::string url_;
    int target_fps_;
    int buffer_size_;

    cv::VideoCapture cap_;
    mutable std::mutex cap_mtx_;

    std::deque<cv::Mat> frame_buffer_;
    mutable std::mutex buffer_lock_;

    std::atomic<bool> running_{true};
    std::atomic<bool> thread_finished_{false};
    std::atomic<bool> released_{false};
    std::thread thread_;

    std::chrono::system_clock::time_point buffer_start_time_;
    std::chrono::system_clock::time_point last_log_time_;
    std::chrono::system_clock::time_point last_success_time_;

    int frames_received_count_{0};
    int frames_sampled_count_{0};
    int consecutive_failures_{0};

    std::vector<FrameWithTs> frame_buffer_1s_;
};

}  // namespace app::utils
