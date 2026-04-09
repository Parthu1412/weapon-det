// Kafka Producer - publishes WeaponMessage detections to an AWS MSK Kafka topic.
// Uses librdkafka with SASL/OAUTHBEARER (MSK IAM) authentication and SASL_SSL.
// Runs a background poll thread to serve delivery callbacks asynchronously.
// Retries producer creation up to AppConfig::max_retries with 1s backoff on failure.
#include "kafka_producer.hpp"

#include <librdkafka/rdkafkacpp.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <thread>
#include <vector>

#include "../config.hpp"
#include "../utils/logger.hpp"
#include "../utils/message.hpp"
#include "msk_token.hpp"

namespace app::kafka {

namespace {

class DeliveryReportCbImpl : public RdKafka::DeliveryReportCb
{
public:
    void dr_cb(RdKafka::Message& msg) override
    {
        if (msg.err() != RdKafka::ERR_NO_ERROR)
        {
            app::utils::Logger::error("[Kafka] Delivery failed: " + msg.errstr());
        } else
        {
            const std::string trace_id = (msg.key() && !msg.key()->empty()) ? *msg.key() : "";
            app::utils::Logger::info(
                "Message sent to Kafka"
                " | topic=" +
                msg.topic_name() + " | trace_id=" + trace_id + " | partition=" +
                std::to_string(msg.partition()) + " | offset=" + std::to_string(msg.offset()));
        }
    }
};

class OauthRefreshCb : public RdKafka::OAuthBearerTokenRefreshCb
{
public:
    void oauthbearer_token_refresh_cb(RdKafka::Handle* handle,
                                      const std::string& /*oauthbearer_config*/) override
    {
        if (!handle)
            return;
        auto& cfg = app::config::AppConfig::getInstance();
        std::string tok = generate_msk_iam_token(cfg.kafka_aws_region);
        if (tok.empty())
        {
            app::utils::Logger::error("[Kafka] MSK IAM token generation failed");
            return;
        }
        std::string errstr;
        auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();
        const RdKafka::ErrorCode code =
            handle->oauthbearer_set_token(tok, now_ms + 900000, "kafka-cluster", {}, errstr);
        if (code != RdKafka::ERR_NO_ERROR)
        {
            app::utils::Logger::error("[Kafka] oauthbearer_set_token: " + errstr);
        }
    }
};

std::string join_brokers(const std::vector<std::string>& servers)
{
    std::string s;
    for (size_t i = 0; i < servers.size(); ++i)
    {
        if (i)
            s += ',';
        s += servers[i];
    }
    return s;
}

}  // namespace

struct KafkaProducer::Impl {
    std::unique_ptr<RdKafka::Producer> producer;
    OauthRefreshCb oauth_cb;
    DeliveryReportCbImpl dr_cb;
    std::atomic<bool> stop_poll{false};
    std::thread poll_thread;
};

KafkaProducer::KafkaProducer() : impl_(std::make_unique<Impl>()) {}
KafkaProducer::~KafkaProducer()
{
    stop();
}

void KafkaProducer::start_with_retry()
{
    auto& cfg = app::config::AppConfig::getInstance();
    if (cfg.kafka_bootstrap_servers.empty())
    {
        app::utils::Logger::error("[Kafka] KAFKA_BOOTSTRAP_SERVERS not set");
        std::exit(1);
    }

    std::string errstr;
    std::unique_ptr<RdKafka::Conf> conf(RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));

    if (conf->set("bootstrap.servers", join_brokers(cfg.kafka_bootstrap_servers), errstr) !=
        RdKafka::Conf::CONF_OK)
    {
        app::utils::Logger::error("[Kafka] " + errstr);
        std::exit(1);
    }
    if (!cfg.kafka_client_id.empty())
        conf->set("client.id", cfg.kafka_client_id, errstr);

    if (conf->set("security.protocol", "SASL_SSL", errstr) != RdKafka::Conf::CONF_OK)
    {
        app::utils::Logger::error("[Kafka] security.protocol: " + errstr);
        std::exit(1);
    }
    if (conf->set("sasl.mechanism", "OAUTHBEARER", errstr) != RdKafka::Conf::CONF_OK)
    {
        app::utils::Logger::error("[Kafka] sasl.mechanism: " + errstr);
        std::exit(1);
    }
    if (conf->set("ssl.endpoint.identification.algorithm", "none", errstr) !=
        RdKafka::Conf::CONF_OK)
    {
        app::utils::Logger::error("[Kafka] ssl.endpoint.identification.algorithm: " + errstr);
        std::exit(1);
    }
    if (conf->set("enable.ssl.certificate.verification", "false", errstr) != RdKafka::Conf::CONF_OK)
    {
        app::utils::Logger::error("[Kafka] enable.ssl.certificate.verification: " + errstr);
        std::exit(1);
    }
    if (conf->set("message.max.bytes", "5000000", errstr) != RdKafka::Conf::CONF_OK)
    {
        app::utils::Logger::error("[Kafka] message.max.bytes: " + errstr);
        std::exit(1);
    }

    if (conf->set("oauthbearer_token_refresh_cb", &impl_->oauth_cb, errstr) !=
        RdKafka::Conf::CONF_OK)
    {
        app::utils::Logger::error("[Kafka] oauth cb: " + errstr);
        std::exit(1);
    }
    if (conf->set("dr_cb", &impl_->dr_cb, errstr) != RdKafka::Conf::CONF_OK)
    {
        app::utils::Logger::error("[Kafka] dr_cb: " + errstr);
        std::exit(1);
    }

    for (int attempt = 1; attempt <= cfg.max_retries; ++attempt)
    {
        std::string e2;
        impl_->producer.reset(RdKafka::Producer::create(conf.get(), e2));
        if (impl_->producer)
        {
            app::utils::Logger::info("[Kafka] Producer connected");
            impl_->stop_poll = false;
            impl_->poll_thread = std::thread([this]() {
                while (!impl_->stop_poll)
                    impl_->producer->poll(100);
            });
            return;
        }
        app::utils::Logger::error("[Kafka] create Producer failed (try " + std::to_string(attempt) +
                                  "): " + e2);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    app::utils::Logger::error("[Kafka] Failed to start producer after retries");
    std::exit(1);
}

void KafkaProducer::stop()
{
    if (impl_ && impl_->producer)
    {
        impl_->stop_poll = true;
        if (impl_->poll_thread.joinable())
            impl_->poll_thread.join();
        impl_->producer->flush(5000);
        impl_->producer.reset();
    }
}

void KafkaProducer::produce(const std::string& topic, const app::utils::WeaponMessage& message)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!impl_ || !impl_->producer)
    {
        app::utils::Logger::error("[Kafka] produce: not started");
        std::exit(1);
    }
    app::utils::Logger::info(
        "Sending message to Kafka"
        " | topic=" +
        topic + " | trace_id=" + message.trace_id +
        " | camera_id=" + std::to_string(message.moksa_camera_id) +
        " | store_id=" + std::to_string(message.store_id));

    nlohmann::ordered_json j = app::utils::to_ordered_json(message);
    std::string payload = j.dump();
    const std::string& key = message.trace_id;

    const int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::system_clock::now().time_since_epoch())
                               .count();
    RdKafka::ErrorCode resp = impl_->producer->produce(
        topic, RdKafka::Topic::PARTITION_UA, RdKafka::Producer::RK_MSG_COPY,
        const_cast<char*>(payload.data()), payload.size(), key.data(), key.size(), now_ms, nullptr);

    if (resp != RdKafka::ERR_NO_ERROR)
    {
        std::string errMsg = "[Kafka] produce failed: " + std::string(RdKafka::err2str(resp));
        app::utils::Logger::error(errMsg);
        stop();
        throw std::runtime_error(errMsg);
    }
    impl_->producer->poll(0);
}

}  // namespace app::kafka
