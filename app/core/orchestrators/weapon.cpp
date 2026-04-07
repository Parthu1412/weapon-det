#include "zmq_io.hpp"
#include "../../config.hpp"
#include "../../utils/aws.hpp"
#include "../../utils/logger.hpp"
#include "../services/weapon_service.hpp"
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <zmq.hpp>

static std::sig_atomic_t g_stop = 0;
static void on_sig(int) { g_stop = 1; }

int main() {
    using namespace app::core::orchestrators;
    app::utils::Logger::set_level_from_env();
    auto& cfg = app::config::AppConfig::getInstance();

    std::signal(SIGINT, on_sig);
    std::signal(SIGTERM, on_sig);

    try {
        zmq::context_t ctx(1);
        zmq::socket_t pull(ctx, zmq::socket_type::pull);
        std::string cam_ep = "tcp://*:" + std::to_string(cfg.zmq_camera_to_weapon_port);
        pull.bind(cam_ep);
        pull.set(zmq::sockopt::rcvhwm, 300);
        app::utils::Logger::info("Bound to port " + std::to_string(cfg.zmq_camera_to_weapon_port) +
            " for receiving from camera orchestrator and person_detection");

        zmq::socket_t push(ctx, zmq::socket_type::push);
        push.bind("tcp://*:" + std::to_string(cfg.zmq_weapon_msg_gen_port));
        push.set(zmq::sockopt::sndhwm, 300);
        app::utils::Logger::info("Bound to port " + std::to_string(cfg.zmq_weapon_msg_gen_port) +
            " for sending to msg gen orchestrator");

        // app::utils::AwsApiManager aws_life;
        // app::utils::S3Client s3_models;
        // if (!cfg.weapon_model_s3_key.empty()) {
        //     try {
        //         s3_models.download_from_s3(cfg.weapon_model_path, cfg.weapon_model_s3_key);
        //     } catch (const std::exception& e) {
        //         app::utils::Logger::error(std::string("[WeaponOrche] S3 weapon model download failed: ") + e.what());
        //         return 1;
        //     }
        // }

        app::core::services::WeaponService weapon_svc;

        app::utils::Logger::info("Initialized");

        int frames_processed = 0;
        int frames_detected = 0;
        auto last_stats_log = std::chrono::steady_clock::now();
        while (!g_stop) {
            ZmqWeaponFramePacket pkt;
            if (!zmq_recv_weapon_frame(pull, pkt, zmq::recv_flags::dontwait)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            if (pkt.frame.empty()) {
                app::utils::Logger::warning("Received None frame camera_id=" +
                    std::to_string(pkt.camera_id) + " store_id=" + std::to_string(pkt.store_id));
                continue;
            }

            frames_processed++;
            try {
                const std::vector<app::utils::PersonDetection>* pptr =
                    pkt.person_detections.empty() ? nullptr : &pkt.person_detections;
                auto t0 = std::chrono::steady_clock::now();
                auto opt = weapon_svc.process_frame(pkt.frame, pptr);
                const double inference_time_s =
                    std::chrono::duration_cast<std::chrono::duration<double>>(
                        std::chrono::steady_clock::now() - t0).count();

                std::ostringstream oss;
                oss << std::fixed << std::setprecision(3);
                oss << "Weapon inference took"
                    << " | inference_time=" << inference_time_s
                    << " | camera_id=" << pkt.camera_id
                    << " | processed_frame_num=" << frames_processed;
                app::utils::Logger::info(oss.str());

                if (!opt.has_value())
                    continue;

                ZmqWeaponOutPacket out;
                out.camera_id = pkt.camera_id;
                out.store_id = pkt.store_id;
                out.timestamp = pkt.timestamp;
                out.confidence = opt->confidence;
                out.weapon_boxes = std::move(opt->weapon_detections);
                out.person_detections = std::move(opt->person_detections);
                out.annotated_frame = std::move(opt->annotated_frame);

                try {
                    if (!zmq_send_weapon_output(push, out)) {
                        app::utils::Logger::warning("ZMQ buffer full, dropped detection");
                    } else {
                        app::utils::Logger::info("Sent detection to msg_gen orchestrator | camera_id=" +
                            std::to_string(out.camera_id) + " store_id=" + std::to_string(out.store_id));
                    }
                } catch (const zmq::error_t& e) {
                    app::utils::Logger::error(std::string("Error sending message: ") + e.what());
                    std::exit(1);
                }
            } catch (const std::exception& e) {
                app::utils::Logger::error(std::string("Error processing frame: ") + e.what());
            }
        }
    } catch (const std::exception& e) {
        app::utils::Logger::error(std::string("[WeaponOrche] Fatal: ") + e.what());
        return 1;
    }
    return 0;
}
