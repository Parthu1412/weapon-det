#include "zmq_io.hpp"
#include "../../config.hpp"
#include "../../utils/logger.hpp"
#include "../../utils/redis_client.hpp"
#include "../../utils/rtsp_camera.hpp"
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <zmq.hpp>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

using app::config::CameraConfig;

namespace {

std::atomic<bool> g_stop{false};
std::mutex g_zmq_send_mtx;
std::mutex g_person_feed_mtx;

#ifdef _WIN32
HANDLE g_person_process = nullptr;
#else
pid_t g_person_pid = -1;
#endif

static void on_sig(int) { g_stop = true; }

void reap_person_child()
{
#ifdef _WIN32
    if (g_person_process) {
        TerminateProcess(g_person_process, 1);
        WaitForSingleObject(g_person_process, 8000);
        CloseHandle(g_person_process);
        g_person_process = nullptr;
    }
#else
    if (g_person_pid > 0) {
        kill(g_person_pid, SIGTERM);
        int st = 0;
        waitpid(g_person_pid, &st, 0);
        g_person_pid = -1;
    }
#endif
}

std::filesystem::path default_person_exe(const char* argv0)
{
    std::filesystem::path p = argv0 ? std::filesystem::path(argv0) : std::filesystem::path(".");
    p = std::filesystem::absolute(p).parent_path();
#ifdef _WIN32
    return p / "person_detection.exe";
#else
    return p / "person_detection";
#endif
}

bool spawn_person_detection(const std::string& exe)
{
#ifdef _WIN32
    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    std::string cmd = "\"" + exe + "\"";
    std::vector<char> buf(cmd.begin(), cmd.end());
    buf.push_back(0);
    if (!CreateProcessA(exe.c_str(), buf.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
        app::utils::Logger::error("[CameraOrche] CreateProcess failed for person_detection err=" +
                                  std::to_string(static_cast<unsigned>(GetLastError())));
        return false;
    }
    CloseHandle(pi.hThread);
    g_person_process = pi.hProcess;
    return true;
#else
    pid_t pid = fork();
    if (pid == -1) {
        app::utils::Logger::error("[CameraOrche] fork failed for person_detection");
        return false;
    }
    if (pid == 0) {
        execl(exe.c_str(), "person_detection", static_cast<char*>(nullptr));
        _exit(127);
    }
    g_person_pid = pid;
    return true;
#endif
}

std::string utc_iso_now()
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

void run_camera_worker(const CameraConfig& cam, zmq::socket_t& push_sock, zmq::socket_t* person_feed,
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
        if (!person_feed) {
            app::utils::Logger::error("[Camera] person_feed missing for non-Redis camera");
            alive = false;
            return;
        }
        app::utils::RTSPCamera cap(cam.url, cfg.fps, cfg.buffer_size);
        while (!g_stop) {
            cv::Mat frame;
            if (!cap.read(frame)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            ZmqWeaponFramePacket pkt;
            pkt.camera_id = cam.id;
            pkt.store_id = cam.store_id;
            pkt.timestamp = utc_iso_now();
            pkt.frame = std::move(frame);

            std::lock_guard<std::mutex> lk(g_person_feed_mtx);
            try {
                if (!zmq_send_weapon_frame(*person_feed, pkt)) {
                    app::utils::Logger::warning("Camera " + std::to_string(cam.id) +
                        ": person ZMQ buffer full, dropped frame");
                }
            } catch (const zmq::error_t& e) {
                app::utils::Logger::error(std::string("[Camera] person_feed ZMQ fatal: ") + e.what());
                std::exit(1);
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

int main(int argc, char** argv)
{
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

    if (has_non_redis) {
        std::filesystem::path child_path;
        if (!cfg.person_detection_exe.empty())
            child_path = cfg.person_detection_exe;
        else
            child_path = default_person_exe(argc > 0 ? argv[0] : nullptr);

        if (!std::filesystem::exists(child_path)) {
            app::utils::Logger::error("[CameraOrche] person_detection binary not found: " + child_path.string() +
                                      " (set PERSON_DETECTION_EXE)");
            return 1;
        }

        if (!spawn_person_detection(child_path.string())) {
            app::utils::Logger::error("[CameraOrche] Failed to spawn person_detection");
            return 1;
        }
        app::utils::Logger::info("[CameraOrche] Spawned person_detection subprocess (grace " +
                                 std::to_string(cfg.person_spawn_grace_sec) + "s)");
        std::this_thread::sleep_for(std::chrono::seconds(std::max(1, cfg.person_spawn_grace_sec)));
    } else {
        app::utils::Logger::info("All cameras are Redis type, skipping person_detection process");
    }

    std::vector<std::thread> threads;
    std::vector<std::unique_ptr<std::atomic<bool>>> alive_flags;
    try {
        zmq::context_t ctx(1);
        zmq::socket_t push(ctx, zmq::socket_type::push);
        const std::string w_ep = "tcp://" + cfg.zmq_camera_to_weapon_host + ":" +
            std::to_string(cfg.zmq_camera_to_weapon_port);
        push.connect(w_ep);
        push.set(zmq::sockopt::sndhwm, 300);
        app::utils::Logger::info("Connected to port " + std::to_string(cfg.zmq_camera_to_weapon_port) +
            " for sending to weapon orchestrator");

        zmq::socket_t person_feed(ctx, zmq::socket_type::push);
        zmq::socket_t* person_ptr = nullptr;
        if (has_non_redis) {
            const std::string ep =
                "tcp://127.0.0.1:" + std::to_string(cfg.zmq_person_frame_port);
            person_feed.connect(ep);
            person_feed.set(zmq::sockopt::sndhwm, 300);
            person_ptr = &person_feed;
            app::utils::Logger::info("[CameraOrche] Connected person_feed to " + ep);
        }

        app::utils::Logger::info("ZMQ sender ready - start weapon orchestrator separately");

        alive_flags.reserve(cameras.size());
        for (std::size_t i = 0; i < cameras.size(); ++i)
            alive_flags.push_back(std::make_unique<std::atomic<bool>>(true));

        std::size_t idx = 0;
        for (const auto& kv : cameras) {
            threads.emplace_back(run_camera_worker, kv.second, std::ref(push), person_ptr,
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
        for (auto& t : threads) {
            if (t.joinable()) t.join();
        }
    } catch (const std::exception& e) {
        app::utils::Logger::error(std::string("[CameraOrche] Fatal: ") + e.what());
        g_stop = true;
        for (auto& t : threads) {
            if (t.joinable()) t.join();
        }
        reap_person_child();
        return 1;
    }

    reap_person_child();
    return 0;
}
