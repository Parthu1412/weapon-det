#include "zmq_io.hpp"
#include "../../config.hpp"
#include "../../kafka/kafka_producer.hpp"
#include "../../mqtt/rabbitmq.hpp"
#include "../../utils/aws.hpp"
#include "../../utils/logger.hpp"
#include "../../utils/message.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <ctime>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <future>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <optional>
#include <zmq.hpp>

static std::sig_atomic_t g_stop = 0;
static void on_sig(int) { g_stop = 1; }

static std::string utc_iso_timestamp() {
    std::time_t t = std::time(nullptr);
    std::tm tm_buf{};
#ifdef _WIN32
    gmtime_s(&tm_buf, &t);
#else
    gmtime_r(&t, &tm_buf);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_buf);
    return std::string(buf);
}

static void publish_once(app::utils::AwsApiManager& /*life*/,
                         app::utils::S3Client& s3,
                         app::kafka::KafkaProducer& kafka,
                         app::mqtt::RabbitMQClient* rmq,
                         const app::core::orchestrators::ZmqWeaponOutPacket& pkt)
{
    auto& cfg = app::config::AppConfig::getInstance();
    const std::string trace_id = app::core::orchestrators::make_trace_id();

    try {
        std::vector<uchar> jpg;
        if (!cv::imencode(".jpg", pkt.annotated_frame, jpg, {cv::IMWRITE_JPEG_QUALITY, 95})) {
            app::utils::Logger::error("[MsgGen] Failed to encode frame trace_id=" + trace_id +
                " camera_id=" + std::to_string(pkt.camera_id));
            return;
        }

        std::ostringstream confs;
        confs << std::fixed << std::setprecision(4) << pkt.confidence;
        std::string obj =
            "detections/" + std::to_string(pkt.store_id) + "/" + trace_id + "_" + confs.str() + ".jpg";

        std::optional<std::string> s3_url;
        try {
            s3_url = s3.upload_bytes_and_get_url(jpg.data(), jpg.size(), obj, "image/jpeg");
        } catch (const std::exception& e) {
            app::utils::Logger::error(std::string("[MsgGen] Error uploading to S3: ") + e.what());
            std::exit(1);
        }

        if (!s3_url || s3_url->empty()) {
            app::utils::Logger::error("[MsgGen] Failed to upload to S3 trace_id=" + trace_id +
                " camera_id=" + std::to_string(pkt.camera_id) +
                " store_id=" + std::to_string(pkt.store_id));
            return;
        }

        app::utils::WeaponMessage msg;
        msg.store_id = pkt.store_id;
        msg.moksa_camera_id = pkt.camera_id;
        msg.detections = nlohmann::json::array();
        nlohmann::json det;
        det["class"] = "Weapon";
        det["confidence"] = pkt.confidence;
        msg.detections.push_back(det);
        msg.gcs_uri = *s3_url;
        msg.trace_id = trace_id;
        msg.timestamp = pkt.timestamp.empty() ? utc_iso_timestamp() : pkt.timestamp;
        msg.model_version = "v1.0";

        try {
            kafka.produce(cfg.kafka_topic, msg);
            app::utils::Logger::info("Published to Kafka");
        } catch (const std::exception& e) {
            app::utils::Logger::error(std::string("Error publishing to Kafka: ") + e.what());
            std::exit(1);
        }

        if (rmq && rmq->is_connected()) {
            try {
                std::string queue = cfg.use_generic_queue ? cfg.generic_queue_name
                                                          : ("weapon_" + std::to_string(pkt.store_id));
                rmq->publish(queue, msg);
                app::utils::Logger::info("Published to RabbitMQ");
            } catch (const std::exception& e) {
                app::utils::Logger::error(std::string("Error publishing to RabbitMQ: ") + e.what());
                std::exit(1);
            }
        }
    } catch (const std::exception& e) {
        app::utils::Logger::error(std::string("[MsgGen] Error publishing detection trace_id=") + trace_id +
            " camera_id=" + std::to_string(pkt.camera_id) +
            " store_id=" + std::to_string(pkt.store_id) + " error=" + e.what());
    } catch (...) {
        app::utils::Logger::error("[MsgGen] Error publishing detection trace_id=" + trace_id +
            " camera_id=" + std::to_string(pkt.camera_id) +
            " store_id=" + std::to_string(pkt.store_id) + " error=(unknown)");
    }
}

int main() {
    using namespace app::core::orchestrators;
    app::utils::Logger::set_level_from_env();
    auto& cfg = app::config::AppConfig::getInstance();

    std::signal(SIGINT, on_sig);
    std::signal(SIGTERM, on_sig);

    app::utils::AwsApiManager aws_life;
    app::kafka::KafkaProducer kafka;
    kafka.start_with_retry();  // matches Python connect_kafka_with_retry(MAX_RETRIES), 1s backoff
    app::utils::Logger::info("[MsgGen] Kafka producer connected");

    app::utils::S3Client s3;

    zmq::context_t ctx(1);
    zmq::socket_t pull(ctx, zmq::socket_type::pull);
    std::string ep = "tcp://" + cfg.zmq_weapon_to_output_host + ":" +
                     std::to_string(cfg.zmq_weapon_msg_gen_port);
    pull.connect(ep);
    pull.set(zmq::sockopt::rcvhwm, 300);
    app::utils::Logger::info("Connected to port " + std::to_string(cfg.zmq_weapon_msg_gen_port) +
        " for receiving from weapon orchestrator");
    app::utils::Logger::info("Initialized");

    std::vector<std::future<void>> active_tasks;

    while (!g_stop) {
        // Prune completed futures — mirrors Python's active_tasks.discard()
        active_tasks.erase(
            std::remove_if(active_tasks.begin(), active_tasks.end(),
                [](std::future<void>& f) {
                    return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
                }),
            active_tasks.end());

        ZmqWeaponOutPacket pkt;
        if (!zmq_recv_weapon_output(pull, pkt, zmq::recv_flags::dontwait)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // Dispatch each detection concurrently — mirrors asyncio.create_task()
        // thread_local RabbitMQ: one connection per OS thread, reused across tasks
        active_tasks.push_back(
            std::async(std::launch::async,
                       [pkt = std::move(pkt), &aws_life, &s3, &kafka]() {
                           thread_local std::unique_ptr<app::mqtt::RabbitMQClient> tl_rmq;
                           if (!tl_rmq) {
                               auto& cfg = app::config::AppConfig::getInstance();
                               if (!cfg.rabbitmq_host.empty()) {
                                   tl_rmq = std::make_unique<app::mqtt::RabbitMQClient>();
                                   tl_rmq->connect_with_retry();
                               }
                           }
                           publish_once(aws_life, s3, kafka, tl_rmq.get(), pkt);
                       }));
    }

    // Drain in-flight tasks before shutdown — mirrors asyncio.gather(*active_tasks)
    for (auto& f : active_tasks) {
        try { f.get(); } catch (...) {}
    }

    kafka.stop();
    return 0;
}
