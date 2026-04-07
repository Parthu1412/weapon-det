// Redis consumer for weapon pipeline — reads JSON frame payloads (frame_base64 +
// detections) from the same key layout as centralized-yolo / Fall C++ producer.
// Consumer pacing matches weapon-detection Python app/utils/redis_client.py (RedisConsumer).

#pragma once

#include <sw/redis++/redis++.h>

#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <vector>

#include "app/config.hpp"
#include "detection_json.hpp"
#include "logger.hpp"

using namespace sw::redis;
using json = nlohmann::json;

namespace app {
namespace utils {

/**
 * WeaponRedisConsumer — same Redis read pacing as Python weapon RedisConsumer.
 *
 * Keys (C++ reads JSON format; aligns with Fall C++ / centralized-yolo producer):
 *   - {camera_id}_latest_str  -> plain string frame number
 *   - {camera_id}_{frame_number}_json -> JSON with frame_base64, detections, timestamp
 */
class WeaponRedisConsumer
{
public:
    WeaponRedisConsumer(const std::string& camera_id, float fps = 0.0f)
        : camera_id_(camera_id),
          frame_number_(0),
          last_read_time_(0.0),
          start_frame_number_(0),
          last_successful_frame_(-1),
          last_frame_number_(0),
          last_timestamp_(0.0)
    {
        auto& config = app::config::AppConfig::getInstance();
        // Python: fps if fps is not None else config.FPS — no forced minimum when <= 0
        fps_ = (fps > 0.0f) ? fps : static_cast<float>(config.fps);

        ConnectionOptions opts;
        opts.host = config.redis_host;
        opts.port = config.redis_port;
        if (!config.redis_password.empty())
        {
            opts.password = config.redis_password;
        }
        redis_ = std::make_unique<Redis>(opts);
        app::utils::Logger::info("[WeaponRedisConsumer] Connected for camera: " + camera_id_);
    }

    struct ReadResult {
        bool ok;
        cv::Mat frame;
        std::vector<PersonDetection> person_detections;
    };

    ReadResult read()
    {
        ReadResult result{false, cv::Mat(), {}};
        try
        {
            auto latest_opt = get_latest_frame_number();
            if (!latest_opt.has_value())
            {
                result.ok = true;
                return result;
            }
            int latest_frame_number = latest_opt.value();

            // Python: if self.fps and self.fps > 0: ... calculate_target_frame
            if (fps_ > 0.0f)
            {
                auto target_opt = calculate_target_frame(latest_frame_number);
                if (!target_opt.has_value())
                {
                    result.ok = true;
                    return result;
                }
                frame_number_ = target_opt.value();
            }

            std::string json_key = camera_id_ + "_" + std::to_string(frame_number_) + "_json";
            auto frame_data_opt = redis_->get(json_key);
            if (!frame_data_opt)
            {
                if (should_wait_or_reset())
                {
                    result.ok = true;
                    return result;
                }
                return result;
            }

            ReadResult parsed = parse_frame_payload(*frame_data_opt);

            // Python advances cursor whenever frame_data was present (even if frame is None).
            last_successful_frame_ = frame_number_;
            last_frame_number_ = frame_number_;
            frame_number_ += 1;
            if (frame_number_ >= static_cast<int>(app::config::AppConfig::getInstance().longint_max))
            {
                frame_number_ = 0;
                last_successful_frame_ = -1;
            }

            if (parsed.frame.empty())
            {
                result.ok = true;
                return result;
            }
            result.frame = std::move(parsed.frame);
            result.person_detections = std::move(parsed.person_detections);

            result.ok = true;
            return result;
        } catch (const Error& e)
        {
            app::utils::Logger::error("[WeaponRedisConsumer] Error reading from Redis: " +
                                      std::string(e.what()));
            try
            {
                reconnect();
            } catch (...)
            {
            }
            result.ok = false;
            return result;
        }
    }

    void reconnect()
    {
        try
        {
            auto& config = app::config::AppConfig::getInstance();
            ConnectionOptions opts;
            opts.host = config.redis_host;
            opts.port = config.redis_port;
            if (!config.redis_password.empty())
                opts.password = config.redis_password;
            redis_ = std::make_unique<Redis>(opts);
            redis_->ping();
            app::utils::Logger::info("[WeaponRedisConsumer] Reconnected for camera: " + camera_id_);
        } catch (const Error& e)
        {
            app::utils::Logger::error("[WeaponRedisConsumer] Reconnect failed: " +
                                      std::string(e.what()));
            throw;
        }
    }

private:
    std::string camera_id_;
    float fps_;
    int frame_number_;
    double last_read_time_;
    int start_frame_number_;
    int last_successful_frame_;
    int last_frame_number_;
    double last_timestamp_;
    std::unique_ptr<Redis> redis_;

    std::optional<int> get_latest_frame_number()
    {
        std::string key = camera_id_ + "_latest_str";
        auto val_opt = redis_->get(key);
        if (!val_opt)
        {
            app::utils::Logger::warning(
                "[WeaponRedisConsumer] No latest frame data set in Redis for camera: " + camera_id_);
            return std::nullopt;
        }
        try
        {
            int fn = std::stoi(*val_opt);
            if (frame_number_ == 0)
            {
                frame_number_ = fn;
            }
            return fn;
        } catch (...)
        {
            return std::nullopt;
        }
    }

    bool is_ahead(int frame_number) const
    {
        return frame_number_ > frame_number;
    }

    bool is_behind(int frame_number) const
    {
        if (frame_number_ < frame_number)
        {
            app::utils::Logger::warning("[WeaponRedisConsumer] Consumer is very behind producer for " +
                                        camera_id_);
            return true;
        }
        return false;
    }

    std::optional<int> calculate_target_frame(int latest_frame_number)
    {
        // Python: if not self.fps or self.fps <= 0: return self.frame_number
        if (fps_ <= 0.0f)
            return frame_number_;

        double current_time =
            std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
                .count();

        if (last_read_time_ == 0.0 || start_frame_number_ == 0)
        {
            last_read_time_ = current_time;
            start_frame_number_ = frame_number_;
            return frame_number_;
        }

        double elapsed_time = current_time - last_read_time_;
        int raw_per_target = (fps_ > 0.0f) ? static_cast<int>(15.0f / fps_) : 1;
        if (raw_per_target < 1)
            raw_per_target = 1;
        int frames_to_advance = static_cast<int>(elapsed_time * fps_) * raw_per_target;

        if (frames_to_advance > 0)
        {
            int target_frame = frame_number_ + frames_to_advance;
            if (target_frame > latest_frame_number)
            {
                target_frame = latest_frame_number;
            }
            if (last_successful_frame_ >= 0 && target_frame == last_successful_frame_)
            {
                app::utils::Logger::debug(
                    "[WeaponRedisConsumer] Waiting for new frame. Producer FPS < target FPS. Last: " +
                    std::to_string(last_successful_frame_) +
                    ", latest: " + std::to_string(latest_frame_number));
                return std::nullopt;
            }
            last_read_time_ = current_time;
            return target_frame;
        }
        return std::nullopt;
    }

    /** Matches Python should_wait_or_reset(): re-queries latest from Redis. */
    bool should_wait_or_reset()
    {
        auto latest_opt = get_latest_frame_number();
        if (!latest_opt.has_value())
            return true;
        int latest_frame_number = latest_opt.value();

        if (is_ahead(latest_frame_number))
            return true;
        if (is_behind(latest_frame_number))
        {
            frame_number_ = latest_frame_number;
            return true;
        }
        return (frame_number_ == last_frame_number_ || frame_number_ == latest_frame_number);
    }

    ReadResult parse_frame_payload(const std::string& data)
    {
        ReadResult r{true, cv::Mat(), {}};
        try
        {
            json payload = json::parse(data);
            if (!payload.contains("frame_base64"))
            {
                r.ok = true;
                return r;
            }
            parse_person_detections_json(payload, r.person_detections);
            last_timestamp_ =
                std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch())
                    .count();
            if (payload.contains("timestamp") && payload["timestamp"].is_number())
                last_timestamp_ = payload["timestamp"].get<double>();
            std::string b64 = payload["frame_base64"].get<std::string>();
            std::vector<uchar> img_bytes;
            size_t out_len = base64_decode(b64, img_bytes);
            if (out_len == 0)
                return r;
            r.frame = cv::imdecode(img_bytes, cv::IMREAD_COLOR);
            return r;
        } catch (const json::exception& e)
        {
            app::utils::Logger::warning("[WeaponRedisConsumer] JSON parse error: " +
                                        std::string(e.what()));
            return r;
        }
    }

    static size_t base64_decode(const std::string& in, std::vector<uchar>& out)
    {
        static const char tbl[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::vector<int> T(256, -1);
        for (int i = 0; i < 64; ++i)
            T[static_cast<uchar>(tbl[i])] = i;

        size_t len = in.size();
        out.resize(len * 3 / 4 + 4);
        size_t out_len = 0;
        int val = 0, bits = -8;
        for (size_t i = 0; i < len; ++i)
        {
            int c = T[static_cast<uchar>(in[i])];
            if (c < 0)
                continue;
            val = (val << 6) + c;
            bits += 6;
            if (bits >= 0)
            {
                if (out_len < out.size())
                    out[out_len++] = static_cast<uchar>((val >> bits) & 0xFF);
                bits -= 8;
            }
        }
        out.resize(out_len);
        return out_len;
    }
};

}  // namespace utils
}  // namespace app
