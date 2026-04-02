#pragma once

#include <SimpleAmqpClient/SimpleAmqpClient.h>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <chrono>
#include <cstdlib>
#include "../utils/message.hpp"
#include "../utils/logger.hpp"
#include "../config.hpp"

using json = nlohmann::json;

namespace app {
namespace mqtt {

class RabbitMQClient {
public:
    RabbitMQClient() {
        auto& config = app::config::AppConfig::getInstance();
        max_retries_ = config.max_retries;
    }

    void connect() {
        auto& config = app::config::AppConfig::getInstance();
        if (config.rabbitmq_host.empty()) {
            app::utils::Logger::info("[RabbitMQ] RABBITMQ host not set – publishing disabled.");
            return;
        }
        if (config.rabbitmq_use_ssl) {
            try {
                const char* ca_path = "/etc/ssl/certs/ca-certificates.crt";
                channel_ = AmqpClient::Channel::CreateSecure(
                    ca_path,
                    config.rabbitmq_host,
                    "", "",
                    config.rabbitmq_port,
                    config.rabbitmq_user,
                    config.rabbitmq_pass,
                    "/", 131072,
                    false);
            } catch (const std::exception& e) {
                std::string err(e.what());
                if (err.find("SSL") != std::string::npos) {
                    app::utils::Logger::error("[RabbitMQ] SSL failed, trying plain TCP.");
                    channel_ = AmqpClient::Channel::Create(
                        config.rabbitmq_host, config.rabbitmq_port,
                        config.rabbitmq_user, config.rabbitmq_pass);
                } else {
                    throw;
                }
            }
        } else {
            channel_ = AmqpClient::Channel::Create(
                config.rabbitmq_host,
                config.rabbitmq_port,
                config.rabbitmq_user,
                config.rabbitmq_pass
            );
        }
        app::utils::Logger::info("[RabbitMQ] Connected to " + config.rabbitmq_host + ":" +
            std::to_string(config.rabbitmq_port) + (config.rabbitmq_use_ssl ? " (SSL)" : ""));
    }

    void connect_with_retry() {
        auto& config = app::config::AppConfig::getInstance();
        if (config.rabbitmq_host.empty())
            return;
        int retry_count = 0;
        while (retry_count < max_retries_) {
            try {
                connect();
                app::utils::Logger::info("[RabbitMQ] Connection successful");
                return;
            } catch (const std::exception& e) {
                retry_count++;
                app::utils::Logger::error("[RabbitMQ] Connection failed (retry " +
                    std::to_string(retry_count) + "/" + std::to_string(max_retries_) +
                    "): " + e.what());
                if (retry_count < max_retries_)
                    std::this_thread::sleep_for(std::chrono::seconds(5));
            }
        }
        app::utils::Logger::error("[RabbitMQ] Max retries reached. Exiting application.");
        std::exit(1);
    }

    bool is_connected() const { return channel_ != nullptr; }

    void publish(const std::string& queue_name, const app::utils::WeaponMessage& message) {
        if (!channel_) {
            app::utils::Logger::warning("[RabbitMQ] Cannot publish, channel is null. Reconnecting...");
            connect_with_retry();
            if (!channel_) return;
        }

        json j;
        app::utils::to_json(j, message);
        std::string payload = j.dump();

        try {
            channel_->DeclareQueue(queue_name, false, true, false, false);
            auto amqp_msg = AmqpClient::BasicMessage::Create(payload);
            amqp_msg->DeliveryMode(AmqpClient::BasicMessage::dm_persistent);
            channel_->BasicPublish("", queue_name, amqp_msg);
            app::utils::Logger::info("[RabbitMQ] Published trace_id: " + message.trace_id +
                " to queue: " + queue_name);

        } catch (const std::exception& e) {
            app::utils::Logger::warning("[RabbitMQ] Publish failed, reconnecting | error=" +
                std::string(e.what()));
            channel_ = nullptr;
            connect_with_retry();
            channel_->DeclareQueue(queue_name, false, true, false, false);
            auto amqp_msg2 = AmqpClient::BasicMessage::Create(payload);
            amqp_msg2->DeliveryMode(AmqpClient::BasicMessage::dm_persistent);
            channel_->BasicPublish("", queue_name, amqp_msg2);
            app::utils::Logger::info("[RabbitMQ] Published after reconnect trace_id: " + message.trace_id);
        }
    }

    void close() {
        channel_ = nullptr;
        app::utils::Logger::info("[RabbitMQ] Connection closed");
    }

private:
    AmqpClient::Channel::ptr_t channel_;
    int max_retries_ = 3;
};

} // namespace mqtt
} // namespace app
