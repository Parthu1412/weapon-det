// Separate process: person detection for non-Redis cameras.
// Binds PULL for raw frames from camera_reader; PUSHes annotated packets to weapon_inference.

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <zmq.hpp>

#include "../../config.hpp"
#include "../../utils/aws.hpp"
#include "../../utils/logger.hpp"
#include "../inferences/person.hpp"
#include "zmq_io.hpp"

static std::atomic<bool> g_stop{false};

static void on_sig(int)
{
    g_stop = true;
}

static std::string utc_iso_now()
{
    std::time_t t = std::time(nullptr);
    std::tm tm_buf{};
#ifdef _WIN32
    gmtime_s(&tm_buf, &t);
#else
    gmtime_r(&t, &tm_buf);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

int main()
{
    using namespace app::core::orchestrators;
    app::utils::Logger::set_level_from_env();
    auto& cfg = app::config::AppConfig::getInstance();

    std::signal(SIGINT, on_sig);
    std::signal(SIGTERM, on_sig);

    try
    {
        zmq::context_t ctx(1);
        zmq::socket_t pull(ctx, zmq::socket_type::pull);
        const std::string bind_ep = "tcp://*:" + std::to_string(cfg.zmq_person_frame_port);
        pull.bind(bind_ep);
        app::utils::Logger::info("[PersonDetection] PULL bound " + bind_ep);

        app::utils::AwsApiManager aws_life;
        app::utils::S3Client s3;
        if (!cfg.person_detection_model_s3_key.empty())
        {
            try
            {
                s3.download_from_s3(cfg.person_detection_model_path,
                                    cfg.person_detection_model_s3_key);
            } catch (const std::exception& e)
            {
                app::utils::Logger::error(
                    std::string("[PersonDetection] S3 model download failed: ") + e.what());
                return 1;
            }
        }

        zmq::socket_t push(ctx, zmq::socket_type::push);
        const std::string w_ep = "tcp://" + cfg.zmq_camera_to_weapon_host + ":" +
                                 std::to_string(cfg.zmq_camera_to_weapon_port);
        push.connect(w_ep);
        push.set(zmq::sockopt::sndhwm, 300);
        app::utils::Logger::info("[PersonDetection] PUSH connected to weapon " + w_ep);

        app::core::inferences::PersonModel model(cfg.person_detection_model_path,
                                                 cfg.person_confidence_threshold,
                                                 cfg.person_iou_threshold, cfg.person_class_id);

        int frames_processed = 0;
        while (!g_stop)
        {
            ZmqWeaponFramePacket pkt;
            if (!zmq_recv_weapon_frame(pull, pkt, zmq::recv_flags::dontwait))
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            if (pkt.frame.empty())
                continue;

            frames_processed++;
            try
            {
                auto t0 = std::chrono::steady_clock::now();
                auto dets = model.detect(pkt.frame);
                const double inference_time_s =
                    std::chrono::duration_cast<std::chrono::duration<double>>(
                        std::chrono::steady_clock::now() - t0)
                        .count();

                std::ostringstream oss;
                oss << std::fixed << std::setprecision(3);
                oss << "Person inference took"
                    << " | inference_time=" << inference_time_s << " | camera_id=" << pkt.camera_id
                    << " | processed_frame_num=" << frames_processed;
                app::utils::Logger::info(oss.str());

                auto& ccfg = app::config::AppConfig::getInstance();
                if (!ccfg.publish_weapon_without_person && dets.empty())
                    continue;

                pkt.person_detections = std::move(dets);
                if (pkt.timestamp.empty())
                    pkt.timestamp = utc_iso_now();

                try
                {
                    if (!zmq_send_weapon_frame(push, pkt))
                    {
                        app::utils::Logger::warning(
                            "[PersonDetection] ZMQ buffer full, dropped frame camera_id=" +
                            std::to_string(pkt.camera_id));
                    }
                } catch (const zmq::error_t& e)
                {
                    app::utils::Logger::error(std::string("[PersonDetection] ZMQ send fatal: ") +
                                              e.what());
                    return 1;
                }
            } catch (const std::exception& e)
            {
                app::utils::Logger::error(std::string("[PersonDetection] Error: ") + e.what());
            }
        }
    } catch (const std::exception& e)
    {
        app::utils::Logger::error(std::string("[PersonDetection] Fatal: ") + e.what());
        return 1;
    }
    return 0;
}
