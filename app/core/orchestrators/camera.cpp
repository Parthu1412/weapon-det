#include "zmq_io.hpp"
#include "../../config.hpp"
#include "../../utils/logger.hpp"
#include "../../utils/redis_client.hpp"
#include "../../utils/rtsp_camera.hpp"
#include "../inferences/person.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <memory>
#include <vector>
#include <zmq.hpp>

using app::config::CameraConfig;

namespace {

struct QueuedFrame {
    int camera_id = 0;
    int store_id = 0;
    std::string timestamp;
    cv::Mat frame;
};

class FrameQueue {
public:
    explicit FrameQueue(size_t cap) : cap_(cap) {}

    bool push(QueuedFrame f) {
        std::unique_lock<std::mutex> lk(mtx_);
        if (q_.size() >= cap_) return false;
        q_.push_back(std::move(f));
        cv_.notify_one();
        return true;
    }

    bool pop_wait(QueuedFrame& out, int timeout_ms) {
        std::unique_lock<std::mutex> lk(mtx_);
        if (!cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                         [&] { return stop_ || !q_.empty(); }))
            return false;
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        return true;
    }

    void stop() {
        std::lock_guard<std::mutex> lk(mtx_);
        stop_ = true;
        cv_.notify_all();
    }

private:
    std::mutex mtx_;
    std::condition_variable cv_;
    std::deque<QueuedFrame> q_;
    size_t cap_ = 1000;
    bool stop_ = false;
};

std::atomic<bool> g_stop{false};
std::mutex g_zmq_send_mtx;

static void on_sig(int) { g_stop = true; }

std::string utc_iso_now() {
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

void person_worker(FrameQueue& q, const std::string& connect_ep, float person_conf, float person_iou,
                   int person_class, const std::string& torchscript_path)
{
    using namespace app::core::orchestrators;
    zmq::context_t pctx(1);
    zmq::socket_t push(pctx, zmq::socket_type::push);
    push.connect(connect_ep);
    push.set(zmq::sockopt::sndhwm, 300);
    app::utils::Logger::info("[PersonWorker] Connected push to " + connect_ep);

    app::core::inferences::PersonModel model(torchscript_path, person_conf, person_iou, person_class);

    while (!g_stop) {
        QueuedFrame qf;
        if (!q.pop_wait(qf, 100)) continue;  // match Python PersonDetectionService queue timeout ~0.1s
        if (qf.frame.empty()) continue;

        try {
            auto dets = model.detect(qf.frame);
            auto& cfg = app::config::AppConfig::getInstance();
            if (!cfg.publish_weapon_without_person && dets.empty())
                continue;

            ZmqWeaponFramePacket pkt;
            pkt.camera_id = qf.camera_id;
            pkt.store_id = qf.store_id;
            pkt.timestamp = qf.timestamp.empty() ? utc_iso_now() : qf.timestamp;
            pkt.frame = qf.frame;
            if (!dets.empty()) pkt.person_detections = std::move(dets);

            std::lock_guard<std::mutex> lk(g_zmq_send_mtx);
            try {
                if (!zmq_send_weapon_frame(push, pkt)) {
                    app::utils::Logger::warning("ZMQ buffer full, dropped frame camera_id=" +
                        std::to_string(pkt.camera_id));
                }
            } catch (const zmq::error_t& e) {
                app::utils::Logger::error(std::string("[PersonWorker] ZMQ send fatal: ") + e.what());
                std::exit(1);
            }
        } catch (const std::exception& e) {
            app::utils::Logger::error(std::string("[PersonWorker] Error: ") + e.what());
        }
    }
}

void run_camera_worker(const CameraConfig& cam, zmq::socket_t& push_sock, FrameQueue* frame_q,
                       std::atomic<bool>& alive)
{
    using namespace app::core::orchestrators;
    auto& cfg = app::config::AppConfig::getInstance();

    app::utils::Logger::info("[Camera] Thread starting camera_id=" + std::to_string(cam.id) +
        " store_id=" + std::to_string(cam.store_id) + " client_type=" + cam.client_type);

    if (cam.client_type == "redis") {
        app::utils::WeaponRedisConsumer redis(std::to_string(cam.id), static_cast<float>(cfg.fps));
        while (!g_stop) {
            auto r = redis.read();
            if (!r.ok) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }
            if (r.frame.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            auto& ccfg = app::config::AppConfig::getInstance();
            bool has_persons = !r.person_detections.empty();
            if (!ccfg.publish_weapon_without_person && !has_persons)
                continue;

            ZmqWeaponFramePacket p;
            p.camera_id = cam.id;
            p.store_id = cam.store_id;
            p.timestamp = utc_iso_now();
            p.frame = r.frame;
            if (!r.person_detections.empty()) p.person_detections = std::move(r.person_detections);

            std::lock_guard<std::mutex> lk(g_zmq_send_mtx);
            try {
                if (!zmq_send_weapon_frame(push_sock, p)) {
                    app::utils::Logger::warning("Camera " + std::to_string(cam.id) +
                        ": ZMQ buffer full, dropped frame");
                }
            } catch (const zmq::error_t& e) {
                app::utils::Logger::error(std::string("[Camera] Fatal ZMQ: ") + e.what());
                std::exit(1);
            }
        }
        alive = false;
        return;
    }

    if (cam.client_type == "rtsp" || cam.client_type == "video") {
        app::utils::RTSPCamera cap(cam.url, cfg.fps, cfg.buffer_size);
        while (!g_stop) {
            cv::Mat frame;
            if (!cap.read(frame)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));  // match CameraService 0.1s
                continue;
            }
            if (!frame_q) continue;
            QueuedFrame qf;
            qf.camera_id = cam.id;
            qf.store_id = cam.store_id;
            qf.timestamp = utc_iso_now();
            qf.frame = std::move(frame);
            if (!frame_q->push(std::move(qf))) {
                app::utils::Logger::warning("Camera " + std::to_string(cam.id) +
                    ": Frame queue full, dropped frame");
            }
        }
        cap.release();
        alive = false;
        return;
    }

    app::utils::Logger::error("[Camera] client_type=" + cam.client_type +
                              " not supported (use redis, rtsp, video).");
    alive = false;
}

}  // namespace

int main() {
    app::utils::Logger::set_level_from_env();
    auto& cfg = app::config::AppConfig::getInstance();
    std::signal(SIGINT, on_sig);
    std::signal(SIGTERM, on_sig);

    auto cameras = cfg.load_camera_configs();
    if (cameras.empty()) {
        app::utils::Logger::error("[CameraOrche] Failed to load camera configurations");
        return 1;
    }

    bool has_non_redis = false;
    for (const auto& kv : cameras) {
        if (kv.second.client_type != "redis") {
            has_non_redis = true;
            break;
        }
    }

    std::unique_ptr<FrameQueue> queue;
    std::thread person_thr;
    if (has_non_redis) {
        queue = std::make_unique<FrameQueue>(static_cast<size_t>(std::max(1, cfg.queue_size)));
        std::string ep = "tcp://" + cfg.zmq_camera_to_weapon_host + ":" +
                         std::to_string(cfg.zmq_camera_to_weapon_port);
        person_thr = std::thread(person_worker, std::ref(*queue), ep,
                                 cfg.person_confidence_threshold, cfg.person_iou_threshold,
                                 cfg.person_class_id, cfg.person_detection_model_path);
        app::utils::Logger::info("[CameraOrche] Starting person detection for non-Redis cameras (in-process thread)");
    } else {
        app::utils::Logger::info("All cameras are Redis type, skipping person detection thread");
    }

    std::vector<std::thread> threads;
    std::vector<std::unique_ptr<std::atomic<bool>>> alive_flags;
    try {
        zmq::context_t ctx(1);
        zmq::socket_t push(ctx, zmq::socket_type::push);
        push.bind("tcp://*:" + std::to_string(cfg.zmq_camera_to_weapon_port));
        push.set(zmq::sockopt::sndhwm, 300);
        app::utils::Logger::info("Bound to port " + std::to_string(cfg.zmq_camera_to_weapon_port) +
            " for sending to weapon orchestrator");

        app::utils::Logger::info("ZMQ sender ready - start weapon orchestrator separately");

        alive_flags.reserve(cameras.size());
        for (std::size_t i = 0; i < cameras.size(); ++i)
            alive_flags.push_back(std::make_unique<std::atomic<bool>>(true));

        std::size_t idx = 0;
        for (const auto& kv : cameras) {
            threads.emplace_back(run_camera_worker, kv.second, std::ref(push), queue.get(),
                                 std::ref(*alive_flags[idx]));
            app::utils::Logger::info("[CameraOrche] Started camera thread camera_id=" +
                std::to_string(kv.second.id));
            ++idx;
        }
        app::utils::Logger::info("[CameraOrche] All camera threads started (count=" +
            std::to_string(threads.size()) + ")");

        app::utils::Logger::info("[CameraOrche] All systems running, starting monitoring...");

        while (!g_stop) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            for (std::size_t i = 0; i < alive_flags.size(); ++i) {
                if (!alive_flags[i]->load()) {
                    app::utils::Logger::warning("[CameraOrche] A camera thread died");
                    g_stop = true;
                    break;
                }
            }
        }

        g_stop = true;
        if (queue) queue->stop();
        for (auto& t : threads) {
            if (t.joinable()) t.join();
        }
        if (person_thr.joinable()) person_thr.join();
    } catch (const std::exception& e) {
        app::utils::Logger::error(std::string("[CameraOrche] Fatal: ") + e.what());
        g_stop = true;
        if (queue) queue->stop();
        for (auto& t : threads) {
            if (t.joinable()) t.join();
        }
        if (person_thr.joinable()) person_thr.join();
        return 1;
    }
    return 0;
}
