#include "zmq_io.hpp"
#include "../../utils/logger.hpp"
#include <chrono>
#include <random>
#include <sstream>

namespace app::core::orchestrators {

namespace {

cv::Mat imdecode_jpeg_message(const zmq::message_t& msg) {
    if (msg.size() == 0) return {};
    const int n = static_cast<int>(msg.size());
    void* bytes = const_cast<void*>(static_cast<const void*>(msg.data()));
    cv::Mat buf(1, n, CV_8U, bytes);
    return cv::imdecode(buf, cv::IMREAD_COLOR);
}

void put_person_json(nlohmann::json& j, const std::vector<app::utils::PersonDetection>& persons) {
    if (persons.empty()) return;
    j["person_detections"] = app::utils::person_detections_to_json(persons);
}

void get_person_json(const nlohmann::json& j, std::vector<app::utils::PersonDetection>& persons) {
    persons.clear();
    if (!j.contains("person_detections") || !j["person_detections"].is_array()) return;
    nlohmann::json wrap;
    wrap["detections"] = j["person_detections"];
    app::utils::parse_person_detections_json(wrap, persons);
}

}  // namespace

bool zmq_send_weapon_frame(zmq::socket_t& sock, const ZmqWeaponFramePacket& p) {
    std::vector<uchar> jpg = p.frame_jpg;
    if (jpg.empty() && !p.frame.empty()) {
        if (!cv::imencode(".jpg", p.frame, jpg)) {
            app::utils::Logger::error("[weapon zmq] imencode frame failed");
            return false;
        }
    }
    nlohmann::json meta;
    meta["camera_id"] = p.camera_id;
    meta["store_id"] = p.store_id;
    meta["timestamp"] = p.timestamp;
    put_person_json(meta, p.person_detections);
    std::string meta_s = meta.dump();
    zmq::message_t m0(meta_s.size());
    memcpy(m0.data(), meta_s.data(), meta_s.size());
    zmq::message_t m1(jpg.size());
    memcpy(m1.data(), jpg.data(), jpg.size());
    try {
        sock.send(m0, zmq::send_flags::sndmore | zmq::send_flags::dontwait);
        sock.send(m1, zmq::send_flags::none);
    } catch (const zmq::error_t& e) {
        if (e.num() == EAGAIN) return false;
        throw;
    }
    return true;
}

bool zmq_recv_weapon_frame(zmq::socket_t& sock, ZmqWeaponFramePacket& p, zmq::recv_flags flags) {
    zmq::message_t m0, m1;
    try {
        auto r0 = sock.recv(m0, flags);
        if (!r0) return false;
        if (!sock.get(zmq::sockopt::rcvmore)) return false;
        auto r1 = sock.recv(m1, zmq::recv_flags::none);
        if (!r1) return false;
    } catch (const zmq::error_t&) {
        return false;
    }
    try {
        std::string meta_s(static_cast<char*>(m0.data()), m0.size());
        auto j = nlohmann::json::parse(meta_s);
        p.camera_id = j.value("camera_id", 0);
        p.store_id = j.value("store_id", 0);
        p.timestamp = j.value("timestamp", std::string());
        get_person_json(j, p.person_detections);
    } catch (const std::exception& e) {
        app::utils::Logger::error(std::string("[weapon zmq] bad frame meta: ") + e.what());
        return false;
    }
    p.frame_jpg.assign(static_cast<const uchar*>(m1.data()),
                       static_cast<const uchar*>(m1.data()) + m1.size());
    p.frame = imdecode_jpeg_message(m1);
    return !p.frame.empty();
}

bool zmq_send_weapon_output(zmq::socket_t& sock, const ZmqWeaponOutPacket& p) {
    std::vector<uchar> jpg = p.annotated_jpg;
    if (jpg.empty() && !p.annotated_frame.empty()) {
        if (!cv::imencode(".jpg", p.annotated_frame, jpg,
                          {cv::IMWRITE_JPEG_QUALITY, 95})) {
            app::utils::Logger::error("[weapon zmq] imencode annotated failed");
            return false;
        }
    }
    nlohmann::json meta;
    meta["camera_id"] = p.camera_id;
    meta["store_id"] = p.store_id;
    meta["timestamp"] = p.timestamp;
    meta["confidence"] = p.confidence;
    meta["weapon_detections"] = p.weapon_boxes;
    put_person_json(meta, p.person_detections);
    std::string meta_s = meta.dump();
    zmq::message_t m0(meta_s.size());
    memcpy(m0.data(), meta_s.data(), meta_s.size());
    zmq::message_t m1(jpg.size());
    memcpy(m1.data(), jpg.data(), jpg.size());
    try {
        sock.send(m0, zmq::send_flags::sndmore | zmq::send_flags::dontwait);
        sock.send(m1, zmq::send_flags::none);
    } catch (const zmq::error_t& e) {
        if (e.num() == EAGAIN) return false;
        throw;
    }
    return true;
}

bool zmq_recv_weapon_output(zmq::socket_t& sock, ZmqWeaponOutPacket& p, zmq::recv_flags flags) {
    zmq::message_t m0, m1;
    try {
        auto r0 = sock.recv(m0, flags);
        if (!r0) return false;
        if (!sock.get(zmq::sockopt::rcvmore)) return false;
        auto r1 = sock.recv(m1, zmq::recv_flags::none);
        if (!r1) return false;
    } catch (const zmq::error_t&) {
        return false;
    }
    try {
        std::string meta_s(static_cast<char*>(m0.data()), m0.size());
        auto j = nlohmann::json::parse(meta_s);
        p.camera_id = j.value("camera_id", 0);
        p.store_id = j.value("store_id", 0);
        p.timestamp = j.value("timestamp", std::string());
        p.confidence = j.value("confidence", 0.f);
        p.weapon_boxes.clear();
        if (j.contains("weapon_detections") && j["weapon_detections"].is_array()) {
            for (const auto& b : j["weapon_detections"]) {
                if (!b.is_array() || b.size() != 4) continue;
                std::vector<int> box(4);
                for (int i = 0; i < 4; ++i)
                    box[i] = b[i].is_number_integer() ? b[i].get<int>() : static_cast<int>(b[i].get<double>());
                p.weapon_boxes.push_back(std::move(box));
            }
        }
        get_person_json(j, p.person_detections);
    } catch (const std::exception& e) {
        app::utils::Logger::error(std::string("[weapon zmq] bad out meta: ") + e.what());
        return false;
    }
    p.annotated_jpg.assign(static_cast<const uchar*>(m1.data()),
                           static_cast<const uchar*>(m1.data()) + m1.size());
    p.annotated_frame = imdecode_jpeg_message(m1);
    return !p.annotated_frame.empty();
}

std::string make_trace_id() {
    auto now = std::chrono::system_clock::now().time_since_epoch().count();
    thread_local std::mt19937_64 gen(std::random_device{}());
    std::stringstream ss;
    ss << std::hex << now << "-" << gen();
    return ss.str();
}

}  // namespace app::core::orchestrators
